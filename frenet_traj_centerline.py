# -*- coding: utf-8 -*-
"""
frenet_transform.py — 固定カメラBEV軌跡のFrenet座標(s, d)変換

使い方（2段階）:
  1) 中心線の作成
       python frenet_transform.py auto
     → 軌跡CSVから基準車線（走行車線）の中心線を自動生成する。
       手順: 車単位フィルタ → 仮軸(PCA) → 車線振り分け →
             ビンごとの中央値+サンプル数重み → 低自由度多項式フィット →
             s輪切りでの作り直し(1回) → 曲率の物理チェック
     → CENTERLINE_NPY と 診断図 centerline_diagnostics.png を保存する

  2) 軌跡CSVへの変換
       python frenet_transform.py convert
     → CSV に s_meter, d_meter, d_dev, s_valid 列を追加した
       *_frenet.csv を保存する

出力列の意味:
  s_meter : 中心線に沿った道なり距離 [m]（中心線の向きが正）
  d_meter : 中心線からの符号付き横距離 [m]（進行方向に向かって左が正）
  d_dev   : 「最寄り車線中心」からのずれ [m] ← ふくらみ解析ではこれを使う
  s_valid : その点が中心線の有効範囲内か（範囲外は外挿になるので解析から除く）
"""

import os
import sys
import numpy as np
import pandas as pd

# ============== 設定（既存スクリプトと合わせる） ==============
VIDEO_PATH     = "20250413140000_20250413150000_out.mp4"
CSV_PATH       = "scene_A_track_recal.csv"
CENTERLINE_NPY = "centerline_meter_scene_A_track.npy"
DIAG_PNG       = "centerline_diagnostics.png"
OVERLAY_PNG    = "centerline_overlay.png"   # 中心線重ね描きの拡大版(縦長)
INVERT_Y       = True    # Trueで上が奥・下が手前の向きに合わせてY軸を反転

# 車線中心の d オフセット [m]。
# auto で中心線を作った場合、基準にした車線（走行車線）の中心が d≈0 になる。
# convert 実行時に表示される d ヒストグラムを見て、もう片方の車線の山の位置
# （およそ ±3.5 m）を追加する。例: [0.0, 3.5]
# 空リスト [] にすると「車両ごとの中央値」を基準にするフォールバックになる。
LANE_CENTERS_D = [0.0]

POLY_DEG   = 3      # convert側: 中心線点列を密サンプルへ再フィットする次数
N_SAMPLES  = 3000   # convert側: 中心線の密なサンプル数（≈数cm刻みになる）

# ---- auto モードの設定 ----
MIN_PTS        = 25     # 車単位フィルタ: 最低追跡点数
MIN_PATHLEN    = 10.0   # 車単位フィルタ: 弦長の最低値 [m]
BIN_M          = 0.5    # 輪切りビン幅 [m]
MIN_BIN_N      = 8      # このサンプル数未満のビンは基準線の有効範囲から外す
POLY_DEG_REF   = 2      # 基準線フィットの次数（2=曲率一定から開始）
CV_CHECK_DEG3  = True   # 車単位交差検証で2次と3次を比較して選ぶ
TARGET_DIR     = "majority"  # 進行方向フィルタ: "majority"=多数派の向き /
                             # +1 か -1 で仮軸方向の向きを直接指定
TARGET_LANE    = "larger"  # "larger"=台数が多いクラスタ(通常は走行車線) /
                           # 数値を入れるとその仮横位置[m]に近いクラスタを選ぶ
EDGE_TRIM_M    = 3.0    # フィット時に有効範囲の両端から除外する長さ [m]。0.0で無効化
S_VALID_MARGIN = 1.0    # convert時、有効範囲の端からこのマージン内も無効扱い [m]
# =============================================================


# ---------------------------------------------------------------
# 共通: 密な中心線点列から弧長と単位接線を計算する
# ---------------------------------------------------------------
def _arclen_tangent(cxy):
    dseg = np.hypot(*np.diff(cxy, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(dseg)])
    tan = np.gradient(cxy, axis=0)
    tan /= np.linalg.norm(tan, axis=1, keepdims=True) + 1e-12
    return s, tan


# ---------------------------------------------------------------
# auto モード本体
# ---------------------------------------------------------------
def _vehicle_filter(df):
    """車単位の入口検査（旧コードの MIN_PTS / MIN_PATHLEN を踏襲）。"""
    keep = []
    for vid, g in df.groupby("Vehicle_ID"):
        if len(g) < MIN_PTS:
            continue
        p = g[["X_meter", "Y_meter"]].to_numpy(dtype=float)
        if np.hypot(*(p[-1] - p[0])) < MIN_PATHLEN:
            continue
        keep.append(vid)
    out = df[df.Vehicle_ID.isin(keep)].copy()
    print(f"車単位フィルタ: {df.Vehicle_ID.nunique()}台 → {len(keep)}台")
    return out


def _split_two_clusters(vals):
    """1次元2クラスタ分割（簡易k-means）。しきい値と両クラスタ中心を返す。"""
    v = np.sort(np.asarray(vals, dtype=float))
    c = np.array([np.percentile(v, 25), np.percentile(v, 75)])
    for _ in range(100):
        lab = np.abs(v[:, None] - c[None, :]).argmin(axis=1)
        newc = np.array([v[lab == k].mean() if np.any(lab == k) else c[k]
                         for k in (0, 1)])
        if np.allclose(newc, c):
            break
        c = newc
    return c.mean(), np.sort(c)


def _bin_stats(coord, val, edges):
    """各ビンの中央値・MAD・サンプル数を返す。"""
    idx = np.digitize(coord, edges) - 1
    nb = len(edges) - 1
    med = np.full(nb, np.nan)
    mad = np.full(nb, np.nan)
    n   = np.zeros(nb, dtype=int)
    for b in range(nb):
        x = val[idx == b]
        n[b] = len(x)
        if len(x) > 0:
            m = np.median(x)
            med[b] = m
            mad[b] = np.median(np.abs(x - m))
    return med, mad, n


def _weighted_polyfit(u, v, w, deg):
    return np.polyfit(u, v, deg, w=w)


def _cv_select_degree(dfl, degs=(2, 3), n_rep=10, seed=0):
    """車単位で半分に分ける交差検証で多項式次数を選ぶ。
    片方の車たちのビン中央値でフィットし、もう片方の生点の残差(中央絶対値)で評価。"""
    vids = np.array(dfl.Vehicle_ID.unique())
    rng = np.random.default_rng(seed)
    edges = np.arange(dfl["_u"].min(), dfl["_u"].max() + BIN_M, BIN_M)
    scores = {d: [] for d in degs}
    for _ in range(n_rep):
        perm = rng.permutation(vids)
        half = len(perm) // 2
        A = dfl[dfl.Vehicle_ID.isin(perm[:half])]
        B = dfl[dfl.Vehicle_ID.isin(perm[half:])]
        med, mad, n = _bin_stats(A["_u"].to_numpy(), A["_v"].to_numpy(), edges)
        ok = (n >= max(3, MIN_BIN_N // 2)) & np.isfinite(med)
        if ok.sum() < max(degs) + 2:
            continue
        uc = (edges[:-1] + edges[1:])[ok] / 2
        for d in degs:
            coef = _weighted_polyfit(uc, med[ok], np.sqrt(n[ok]), d)
            res = B["_v"].to_numpy() - np.polyval(coef, B["_u"].to_numpy())
            scores[d].append(np.median(np.abs(res)))
    means = {d: (np.mean(s) if s else np.inf) for d, s in scores.items()}
    for d in degs:
        print(f"  交差検証 次数{d}: 検証側残差(中央絶対値) {means[d]*100:.2f} cm")
    best = min(means, key=means.get)
    # 3次の改善がごく僅か(<2mm)なら硬い2次を優先する
    if best != degs[0] and (means[degs[0]] - means[best]) < 0.002:
        best = degs[0]
        print(f"  改善幅が僅少のため硬い次数{best}を採用")
    else:
        print(f"  採用次数: {best}")
    return best


def build_centerline_auto():
    """軌跡CSVから基準車線の中心線を自動生成して CENTERLINE_NPY に保存する。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for fp in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
               "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf",
               "C:/Windows/Fonts/meiryo.ttc",
               "C:/Windows/Fonts/YuGothM.ttc",
               "C:/Windows/Fonts/msgothic.ttc"]:
        if os.path.exists(fp):
            font_manager.fontManager.addfont(fp)
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=fp).get_name()
            break
    plt.rcParams["axes.unicode_minus"] = False

    df = pd.read_csv(CSV_PATH)
    df = _vehicle_filter(df)
    xy = df[["X_meter", "Y_meter"]].to_numpy(dtype=float)

    # ---- (1) 仮軸: 全点のPCA主軸。輪切りの向きを揃えるだけの仮の物差し ----
    mu = xy.mean(axis=0)
    _, _, Vt = np.linalg.svd(xy - mu, full_matrices=False)
    e1 = Vt[0]                       # 進行方向に近い軸
    e2 = np.array([-e1[1], e1[0]])   # それと直交する軸
    df["_u"] = (xy - mu) @ e1
    df["_v"] = (xy - mu) @ e2

    # ---- (1.5) 進行方向フィルタ: 対向車線の車を除外する ----
    df = df.sort_values(["Vehicle_ID", "Frame"])
    du = df.groupby("Vehicle_ID")["_u"].agg(lambda g: g.iloc[-1] - g.iloc[0])
    if TARGET_DIR in (+1, -1):
        want = TARGET_DIR
    else:
        want = 1 if (du > 0).sum() >= (du < 0).sum() else -1
    same = du.index[np.sign(du) == want]
    n_before = df.Vehicle_ID.nunique()
    df = df[df.Vehicle_ID.isin(same)]
    print(f"進行方向フィルタ: {n_before}台 → {df.Vehicle_ID.nunique()}台 "
          f"(仮軸の向き {'+' if want>0 else '-'} を採用、対向車線を除外)")

    # ---- (2) 車線振り分け: 車ごとの仮横位置の中央値を2クラスタに分ける ----
    med_v = df.groupby("Vehicle_ID")["_v"].median()
    thr, centers = _split_two_clusters(med_v.to_numpy())
    lab = (med_v > thr).astype(int)          # 0=下側クラスタ, 1=上側クラスタ
    n0, n1 = (lab == 0).sum(), (lab == 1).sum()
    print(f"車線振り分け: クラスタ中心 v≈{centers[0]:+.2f}m ({n0}台) / "
          f"v≈{centers[1]:+.2f}m ({n1}台), しきい値 {thr:+.2f}m")
    sep = centers[1] - centers[0]
    if not (2.5 <= sep <= 4.5):
        print(f"警告: クラスタ間隔 {sep:.1f}m が車線幅(≈3.5m)と合いません。"
              "車線以外(対向線や誤追跡)で割れている可能性があります")
    if isinstance(TARGET_LANE, (int, float)):
        pick = int(np.argmin(np.abs(centers - float(TARGET_LANE))))
    else:
        pick = int(n1 > n0)
    target_ids = med_v.index[lab == pick]
    dfl = df[df.Vehicle_ID.isin(target_ids)].copy()
    print(f"基準車線として v≈{centers[pick]:+.2f}m のクラスタ({len(target_ids)}台)を採用")
    print("※台数の多い側=走行車線のはず。動画と見比べて逆なら TARGET_LANE に"
          " 期待する仮横位置[m]を数値で指定してください。")

    # ---- (3) 1回目: 仮軸uで輪切り → ビン中央値 → 重み付きフィット ----
    edges = np.arange(dfl["_u"].min(), dfl["_u"].max() + BIN_M, BIN_M)
    uc_all = (edges[:-1] + edges[1:]) / 2
    med, mad, n = _bin_stats(dfl["_u"].to_numpy(), dfl["_v"].to_numpy(), edges)
    ok = (n >= MIN_BIN_N) & np.isfinite(med)

    # 有効範囲 = 条件を満たすビンの最大連続区間（飛び地は使わない）
    runs, start = [], None
    for i, o in enumerate(ok):
        if o and start is None:
            start = i
        if (not o or i == len(ok) - 1) and start is not None:
            end = i if o else i - 1
            runs.append((start, end))
            start = None
    if not runs:
        raise RuntimeError("有効ビンがありません。MIN_BIN_N か BIN_M を見直してください")
    r0, r1 = max(runs, key=lambda r: r[1] - r[0])
    valid = np.zeros_like(ok)
    valid[r0:r1 + 1] = ok[r0:r1 + 1]
    print(f"有効範囲: u = {edges[r0]:.1f} 〜 {edges[r1+1]:.1f} m "
          f"(有効ビン {valid.sum()}/{len(ok)})")

    core = (uc_all >= edges[r0] + EDGE_TRIM_M) & (uc_all <= edges[r1 + 1] - EDGE_TRIM_M)
    valid &= core

    deg = POLY_DEG_REF
    if CV_CHECK_DEG3:
        deg = _cv_select_degree(dfl, degs=(POLY_DEG_REF, POLY_DEG_REF + 1))
    coef1 = _weighted_polyfit(uc_all[valid], med[valid], np.sqrt(n[valid]), deg)
    res1 = med[valid] - np.polyval(coef1, uc_all[valid])
    print(f"1回目フィット(次数{deg}): ビン中央値残差 RMS {np.sqrt(np.mean(res1**2))*100:.2f} cm")

    # ---- (4) 2回目: 出来た曲線に沿った弧長sで輪切りし直して作り直す ----
    ug = np.arange(edges[r0], edges[r1 + 1], BIN_M / 5)
    cxy1 = mu + ug[:, None] * e1 + np.polyval(coef1, ug)[:, None] * e2
    s1, tan1 = _arclen_tangent(cxy1)
    sp, dp = cart_to_frenet(dfl[["X_meter", "Y_meter"]].to_numpy(dtype=float),
                            cxy1, s1, tan1)
    inside = (sp > 0) & (sp < s1[-1])
    edges_s = np.arange(0, s1[-1] + BIN_M, BIN_M)
    sc = (edges_s[:-1] + edges_s[1:]) / 2
    med2, mad2, n2 = _bin_stats(sp[inside], dp[inside], edges_s)
    ok2 = (n2 >= MIN_BIN_N) & np.isfinite(med2)
    ok2 &= (sc >= EDGE_TRIM_M) & (sc <= s1[-1] - EDGE_TRIM_M)
    coef2 = _weighted_polyfit(sc[ok2], med2[ok2], np.sqrt(n2[ok2]), deg)
    res2 = med2[ok2] - np.polyval(coef2, sc[ok2])
    print(f"2回目フィット(s輪切り): ビン中央値残差 RMS {np.sqrt(np.mean(res2**2))*100:.2f} cm")

    # 1回目の曲線を、s輪切りで求めた横補正ぶんだけ法線方向へずらして最終形にする
    nrm = np.stack([-tan1[:, 1], tan1[:, 0]], axis=1)   # d>0 の向き（進行方向左）
    dfix = np.polyval(coef2, s1)
    cxy2 = cxy1 + dfix[:, None] * nrm

    # ---- (5) 物理チェック: 曲率から暗黙の曲線半径を逆算 ----
    sf, tanf = _arclen_tangent(cxy2)
    dx  = np.gradient(cxy2[:, 0], sf)
    dy  = np.gradient(cxy2[:, 1], sf)
    ddx = np.gradient(dx, sf)
    ddy = np.gradient(dy, sf)
    kappa = np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2)**1.5
    core = slice(len(sf) // 10, -len(sf) // 10)   # 端は微分が荒れるので除外
    kmax = np.nanmax(kappa[core])
    if kmax > 1e-9:
        print(f"物理チェック: 最小曲線半径 ≈ {1.0/kmax:.0f} m "
              f"(高速道路本線なら数百m以上のはず。小さすぎたらノイズへの過剰適合を疑う)")
    else:
        print("物理チェック: ほぼ直線 (曲率 ≈ 0)")

    np.save(CENTERLINE_NPY, cxy2)
    print(f"保存しました: {CENTERLINE_NPY} ({len(cxy2)}点, 全長 {sf[-1]:.1f} m)")

    # ---- (6) 診断図 ----
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    ax = axes[0, 0]
    ax.scatter(df["X_meter"], df["Y_meter"], s=1, c="0.8", label="全軌跡点")
    ax.scatter(dfl["X_meter"], dfl["Y_meter"], s=1, c="tab:blue", alpha=0.3,
               label="基準車線の軌跡点")
    ax.plot(cxy2[:, 0], cxy2[:, 1], "r-", lw=2, label="生成した中心線")
    ax.set_title("BEV上の軌跡と生成中心線")
    ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]")
    ax.set_aspect("equal"); ax.legend(markerscale=6, fontsize=9)

    ax = axes[0, 1]
    ax.bar(sc, n2, width=BIN_M * 0.9, color="tab:blue")
    ax.axhline(MIN_BIN_N, color="red", ls="--", lw=1, label=f"MIN_BIN_N={MIN_BIN_N}")
    ax.set_title("カバレッジ: 各sビンのサンプル数")
    ax.set_xlabel("s [m]"); ax.set_ylabel("点数"); ax.legend()

    ax = axes[1, 0]
    ax.plot(sc[ok2], mad2[ok2] * 100, "-o", ms=3)
    ax.set_title("ばらつき: 各sビンのMAD (追跡が汚い区間はここが太る)")
    ax.set_xlabel("s [m]"); ax.set_ylabel("MAD [cm]")

    ax = axes[1, 1]
    ax.axhline(0, color="0.7", lw=1)
    ax.plot(sc[ok2], res2 * 100, "-o", ms=3)
    ax.set_title("残差: ビン中央値 − フィット (sに沿った構造が残れば次数を上げる)")
    ax.set_xlabel("s [m]"); ax.set_ylabel("残差 [cm]")
    plt.tight_layout()
    plt.savefig(DIAG_PNG, dpi=130)
    print(f"診断図を保存: {DIAG_PNG}")

    # ---- (7) 中心線重ね描きの拡大版(横軸だけ拡大した縦長図) ----
    fig2, ax = plt.subplots(figsize=(8, 14))
    for vid, g in df.groupby("Vehicle_ID"):
        g = g.sort_values("Frame")
        ax.plot(g.X_meter, g.Y_meter, color="0.45", lw=0.8, alpha=0.8, zorder=1)
    ax.plot(cxy2[:, 0], cxy2[:, 1], "r-", lw=2.5, zorder=3, label="生成した中心線")
    xr0, xr1 = df.X_meter.min(), df.X_meter.max()
    pad = 0.15 * (xr1 - xr0) + 1.0
    ax.set_xlim(xr0 - pad, xr1 + pad)
    yr0, yr1 = dfl.Y_meter.min(), dfl.Y_meter.max()
    ypad = 0.05 * (yr1 - yr0) + 2.0
    ax.set_ylim(yr0 - ypad, yr1 + ypad)
    if INVERT_Y:
        ax.invert_yaxis()
        ax.set_title("BEV座標と生成中心線\n上が奥・下が手前")
    else:
        ax.set_title("BEV座標と生成中心線")
    ax.set_xlabel("X_meter 横方向[m]")
    ax.set_ylabel("Y_meter 進行方向[m]")
    ax.grid(True, color="0.9"); ax.set_axisbelow(True)
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(OVERLAY_PNG, dpi=130)
    print(f"中心線重ね描き(拡大版)を保存: {OVERLAY_PNG}")


# ---------------------------------------------------------------
# 変換側（既存のまま。fit_centerlineはautoの密な点列も問題なく扱える）
# ---------------------------------------------------------------
def fit_centerline(raw_pts, deg=POLY_DEG, n=N_SAMPLES):
    """点列を滑らかな曲線にフィットし、密なサンプル・弧長・接線を返す。"""
    p = np.asarray(raw_pts, dtype=float)
    seg = np.hypot(*np.diff(p, axis=0).T)
    t = np.concatenate([[0.0], np.cumsum(seg)])
    t = t / t[-1]

    deg = min(deg, len(p) - 1)
    cx = np.polyfit(t, p[:, 0], deg)
    cy = np.polyfit(t, p[:, 1], deg)

    tt = np.linspace(0.0, 1.0, n)
    cxy = np.stack([np.polyval(cx, tt), np.polyval(cy, tt)], axis=1)
    s, tan = _arclen_tangent(cxy)
    return cxy, s, tan


def cart_to_frenet(xy, cxy, s, tan):
    """点群 xy (N,2) を (s, d) に変換する。d>0 は進行方向に向かって左。"""
    from scipy.spatial import cKDTree
    tree = cKDTree(cxy)
    _, idx = tree.query(xy, k=1)
    near = cxy[idx]
    tg = tan[idx]
    rel = xy - near
    s_out = s[idx] + np.einsum("ij,ij->i", rel, tg)
    d_out = tg[:, 0] * rel[:, 1] - tg[:, 1] * rel[:, 0]
    return s_out, d_out


def convert_csv():
    if not os.path.exists(CENTERLINE_NPY):
        raise RuntimeError("先に `python frenet_transform.py auto` を実行してください")
    raw = np.load(CENTERLINE_NPY)
    cxy, s, tan = fit_centerline(raw)
    print(f"中心線フィット完了: 全長 {s[-1]:.1f} m, サンプル間隔 {s[-1]/len(s)*100:.1f} cm")

    df = pd.read_csv(CSV_PATH)
    xy = df[["X_meter", "Y_meter"]].to_numpy(dtype=float)
    s_out, d_out = cart_to_frenet(xy, cxy, s, tan)
    df["s_meter"] = s_out
    df["d_meter"] = d_out
    # 有効範囲フラグ: 中心線の端より外は外挿なので解析から除けるようにする
    df["s_valid"] = (s_out >= S_VALID_MARGIN) & (s_out <= s[-1] - S_VALID_MARGIN)
    n_out = (~df["s_valid"]).sum()
    print(f"有効範囲: s = {S_VALID_MARGIN:.1f} 〜 {s[-1]-S_VALID_MARGIN:.1f} m "
          f"(範囲外の点 {n_out} / {len(df)})")

    # --- ふくらみ解析用の「ずれ」列 d_dev ---
    if LANE_CENTERS_D:
        lanes = np.asarray(LANE_CENTERS_D, dtype=float)
        nearest = lanes[np.argmin(np.abs(d_out[:, None] - lanes[None, :]), axis=1)]
        df["d_dev"] = d_out - nearest
        print(f"d_dev = d - 最寄り車線中心 {LANE_CENTERS_D} で計算")
    else:
        df["d_dev"] = df["d_meter"] - df.groupby("Vehicle_ID")["d_meter"].transform("median")
        print("d_dev = d - 車両ごとの中央値 で計算（LANE_CENTERS_D 未設定）")

    # 車線中心の設定を助けるための d ヒストグラム(テキスト表示)
    dv = d_out[df["s_valid"].to_numpy()]
    hist, be = np.histogram(dv, bins=np.arange(np.floor(dv.min()), np.ceil(dv.max()) + 0.25, 0.25))
    print("d の分布 (LANE_CENTERS_D 設定の参考):")
    top = hist.max()
    for h, b0, b1 in zip(hist, be[:-1], be[1:]):
        if h > top * 0.05:
            print(f"  {b0:+5.2f}〜{b1:+5.2f} m | {'#' * int(40*h/top)}")

    out = os.path.splitext(CSV_PATH)[0] + "_frenet_auto.csv"
    df.to_csv(out, index=False)
    print(f"保存しました: {out}")
    print(df[["s_meter", "d_meter", "d_dev"]].describe().round(2))


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "auto":
        build_centerline_auto()
    elif mode == "convert":
        convert_csv()
    else:
        print(__doc__)