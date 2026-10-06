# -*- coding: utf-8 -*-
"""
frenet_transform.py — 固定カメラBEV軌跡のFrenet座標(s, d)変換

使い方（2段階）:
  1) 中心線の作成
       python frenet_transform.py auto CSV CENTERLINE_NPY
     → 軌跡CSVから基準車線（走行車線）の中心線を自動生成する。
       手順: 車単位フィルタ → 中心線に使うクラスの絞り込み → 仮軸(PCA) →
             車線振り分け → ビンごとの中央値+サンプル数重み →
             低自由度多項式フィット → s輪切りでの作り直し(1回) → 曲率の物理チェック
     → CENTERLINE_NPY と 診断図を保存する

  2) 軌跡CSVへの変換
       python frenet_transform.py convert CSV CENTERLINE_NPY
     → CSV に s_meter, d_meter, s_valid 列を追加した *_frenet_auto.csv を保存する

出力列の意味:
  s_meter : 中心線に沿った道なり距離 [m]（中心線の向きが正）
  d_meter : 中心線からの符号付き横距離 [m]（進行方向に向かって左が正）
  s_valid : その点が中心線の有効範囲内か（範囲外は外挿になるので解析から除く）
  ※ 車線中心からのずれ d_dev は evaluate_refpoint.py が普通車基準で作る
"""

import os
import sys
import numpy as np
import pandas as pd

# ============== 設定 ==============
CSV_PATH       = None    # コマンドライン引数（またはインポート側）で指定する
CENTERLINE_NPY = None
INVERT_Y       = True    # Trueで上が奥・下が手前の向きに合わせてY軸を反転（重ね描き図のみ）

POLY_DEG   = 3      # convert側: 中心線点列を密サンプルへ再フィットする次数
N_SAMPLES  = 3000   # convert側: 中心線の密なサンプル数（≈数cm刻みになる）

# ---- auto モードの設定 ----
CENTERLINE_CLASSES = [2]  # 中心線の作成に使う Class_ID（車両単位の多数決）。None で全車両
                          # 評価指標の前提「すべて普通車基準」に合わせて普通車(2)のみにする
MIN_PTS        = 25     # 車単位フィルタ: 最低追跡点数
MIN_PATHLEN    = 10.0   # 車単位フィルタ: 弦長の最低値 [m]
BIN_M          = 0.5    # 輪切りビン幅 [m]
MIN_BIN_N      = 4      # このサンプル数未満のビンは基準線の有効範囲から外す
                        # （普通車のみで作るため 8 → 4 に下げた。8 だと有効範囲が半分になる）
POLY_DEG_REF   = 2      # 基準線フィットの次数（2=曲率一定から開始）
CV_CHECK_DEG3  = True   # 車単位交差検証で2次と3次を比較して選ぶ
TARGET_DIR     = "majority"  # 進行方向フィルタ: "majority"=多数派の向き / +1 か -1 で指定
# TARGET_LANE    = "larger"    # "larger"=台数が多いクラスタ(通常は走行車線) / 数値で仮横位置[m]指定
N_LANES = 5
# _v が小さい順に 0,1,2,3,4
# 中央の車線を基準線にするなら 2
TARGET_LANE_INDEX = 2
EDGE_TRIM_M    = 3.0    # フィット時に有効範囲の両端から除外する長さ [m]。0.0で無効化
S_VALID_MARGIN = 1.0    # convert時、有効範囲の端からこのマージン内も無効扱い [m]
# ==================================


def _out_png(suffix):
    base = os.path.splitext(CENTERLINE_NPY)[0]
    return f"{base}_{suffix}.png"


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
    """車単位の入口検査（MIN_PTS / MIN_PATHLEN）。"""
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


def _class_filter(df):
    """中心線の材料を CENTERLINE_CLASSES の車両（多数決クラス）に絞る。"""
    if CENTERLINE_CLASSES is None:
        return df
    maj = df.groupby("Vehicle_ID")["Class_ID"].agg(lambda s: s.mode().iloc[0])
    keep = maj.index[maj.isin(CENTERLINE_CLASSES)]
    n0 = df.Vehicle_ID.nunique()
    out = df[df.Vehicle_ID.isin(keep)].copy()
    print(f"中心線クラスフィルタ: Class_ID∈{CENTERLINE_CLASSES} → {n0}台 → {out.Vehicle_ID.nunique()}台")
    return out


# def _split_two_clusters(vals):
#     """1次元2クラスタ分割（簡易k-means）。しきい値と両クラスタ中心を返す。"""
#     v = np.sort(np.asarray(vals, dtype=float))
#     c = np.array([np.percentile(v, 25), np.percentile(v, 75)])
#     for _ in range(100):
#         lab = np.abs(v[:, None] - c[None, :]).argmin(axis=1)
#         newc = np.array([v[lab == k].mean() if np.any(lab == k) else c[k]
#                          for k in (0, 1)])
#         if np.allclose(newc, c):
#             break
#         c = newc
#     return c.mean(), np.sort(c)

def _split_k_clusters(vals, k=5):
    """
    1次元k-means。
    車両ごとの横位置中央値を k 個の車線クラスタに分ける。

    戻り値
    -------
    labels  : 各入力値のクラスタ番号
    centers : 小さい順に並べたクラスタ中心
    """
    v = np.asarray(vals, dtype=float)

    if len(v) < k:
        raise RuntimeError(
            f"車両数 {len(v)} 台では {k} クラスタに分割できません"
        )

    # 初期中心
    # 5車線なら 10,30,50,70,90 percentile 付近から開始
    q = np.linspace(10, 90, k)
    centers = np.percentile(v, q)

    for _ in range(100):
        # 最も近い中心へ割り当て
        labels = np.abs(
            v[:, None] - centers[None, :]
        ).argmin(axis=1)

        new_centers = centers.copy()

        for j in range(k):
            x = v[labels == j]

            if len(x) > 0:
                new_centers[j] = np.mean(x)

        if np.allclose(new_centers, centers):
            break

        centers = new_centers

    # クラスタ番号を横位置の小さい順に揃える
    order = np.argsort(centers)
    centers = centers[order]

    remap = np.empty(k, dtype=int)

    for new_id, old_id in enumerate(order):
        remap[old_id] = new_id

    labels = remap[labels]

    return labels, centers

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


def _cv_select_degree(dfl, degs=(2, 3), n_rep=10, seed=0):
    """車単位で半分に分ける交差検証で多項式次数を選ぶ。"""
    vids = np.array(dfl.Vehicle_ID.unique())
    rng = np.random.default_rng(seed)
    edges = np.arange(dfl["_u"].min(), dfl["_u"].max() + BIN_M, BIN_M)
    scores = {d: [] for d in degs}
    for _ in range(n_rep):
        perm = rng.permutation(vids)
        half = len(perm) // 2
        A = dfl[dfl.Vehicle_ID.isin(perm[:half])]
        B = dfl[dfl.Vehicle_ID.isin(perm[half:])]
        med, _, n = _bin_stats(A["_u"].to_numpy(), A["_v"].to_numpy(), edges)
        ok = (n >= max(3, MIN_BIN_N // 2)) & np.isfinite(med)
        if ok.sum() < max(degs) + 2:
            continue
        uc = (edges[:-1] + edges[1:])[ok] / 2
        for d in degs:
            coef = np.polyfit(uc, med[ok], d, w=np.sqrt(n[ok]))
            res = B["_v"].to_numpy() - np.polyval(coef, B["_u"].to_numpy())
            scores[d].append(np.median(np.abs(res)))
    means = {d: (np.mean(s) if s else np.inf) for d, s in scores.items()}
    for d in degs:
        print(f"  交差検証 次数{d}: 検証側残差(中央絶対値) {means[d]*100:.2f} cm")
    best = min(means, key=means.get)
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

    df_all = pd.read_csv(CSV_PATH)
    df = _vehicle_filter(df_all)
    df = _class_filter(df)                       # 普通車だけで中心線を作る
    xy = df[["X_meter", "Y_meter"]].to_numpy(dtype=float)

    # ---- (1) 仮軸: 全点のPCA主軸 ----
    mu = xy.mean(axis=0)
    _, _, Vt = np.linalg.svd(xy - mu, full_matrices=False)
    e1 = Vt[0]
    e2 = np.array([-e1[1], e1[0]])
    df["_u"] = (xy - mu) @ e1
    df["_v"] = (xy - mu) @ e2

    # ---- (1.5) 進行方向フィルタ ----
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

    # ---- (2) 車線振り分け ----
    # med_v = df.groupby("Vehicle_ID")["_v"].median()
    # thr, centers = _split_two_clusters(med_v.to_numpy())
    # lab = (med_v > thr).astype(int)
    # n0, n1 = (lab == 0).sum(), (lab == 1).sum()
    # print(f"車線振り分け: クラスタ中心 v≈{centers[0]:+.2f}m ({n0}台) / "
    #       f"v≈{centers[1]:+.2f}m ({n1}台), しきい値 {thr:+.2f}m")
    # sep = centers[1] - centers[0]
    # if not (2.5 <= sep <= 4.5):
    #     print(f"警告: クラスタ間隔 {sep:.1f}m が車線幅(≈3.5m)と合いません。")
    # if isinstance(TARGET_LANE, (int, float)):
    #     pick = int(np.argmin(np.abs(centers - float(TARGET_LANE))))
    # else:
    #     pick = int(n1 > n0)
    # target_ids = med_v.index[lab == pick]
    # dfl = df[df.Vehicle_ID.isin(target_ids)].copy()
    # print(f"基準車線として v≈{centers[pick]:+.2f}m のクラスタ({len(target_ids)}台)を採用")
    # ---- (2) 5車線に振り分け ----
    med_v = df.groupby("Vehicle_ID")["_v"].median()

    labels, centers = _split_k_clusters(
        med_v.to_numpy(),
        k=N_LANES
    )

    # Vehicle_ID をindexとしてクラスタ番号を持つ
    lane_label = pd.Series(
        labels,
        index=med_v.index
    )

    print("車線クラスタ:")

    for j in range(N_LANES):
        n_j = int((lane_label == j).sum())

        print(
            f"  lane{j}: "
            f"v≈{centers[j]:+.2f} m, "
            f"{n_j}台"
        )

    # 隣接クラスタ間隔の確認
    sep = np.diff(centers)

    print(
        "隣接クラスタ間隔:",
        " / ".join(f"{x:.2f}m" for x in sep)
    )

    for j, x in enumerate(sep):
        if not (2.0 <= x <= 5.0):
            print(
                f"警告: lane{j} - lane{j+1} の間隔 "
                f"{x:.2f}m が不自然です"
            )

    # 指定した1車線だけを中心線生成に使用
    pick = TARGET_LANE_INDEX

    target_ids = lane_label.index[
        lane_label == pick
    ]

    dfl = df[
        df.Vehicle_ID.isin(target_ids)
    ].copy()

    print(
        f"基準車線: lane{pick} "
        f"(v≈{centers[pick]:+.2f}m, "
        f"{len(target_ids)}台)"
    )

    # ---- (3) 1回目: 仮軸uで輪切り → ビン中央値 → 重み付きフィット ----
    edges = np.arange(dfl["_u"].min(), dfl["_u"].max() + BIN_M, BIN_M)
    uc_all = (edges[:-1] + edges[1:]) / 2
    med, _, n = _bin_stats(dfl["_u"].to_numpy(), dfl["_v"].to_numpy(), edges)
    ok = (n >= MIN_BIN_N) & np.isfinite(med)

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
    coef1 = np.polyfit(uc_all[valid], med[valid], deg, w=np.sqrt(n[valid]))
    res1 = med[valid] - np.polyval(coef1, uc_all[valid])
    print(f"1回目フィット(次数{deg}): ビン中央値残差 RMS {np.sqrt(np.mean(res1**2))*100:.2f} cm")

    # ---- (4) 2回目: 弧長sで輪切りし直して作り直す ----
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
    coef2 = np.polyfit(sc[ok2], med2[ok2], deg, w=np.sqrt(n2[ok2]))
    res2 = med2[ok2] - np.polyval(coef2, sc[ok2])
    print(f"2回目フィット(s輪切り): ビン中央値残差 RMS {np.sqrt(np.mean(res2**2))*100:.2f} cm")

    nrm = np.stack([-tan1[:, 1], tan1[:, 0]], axis=1)
    dfix = np.polyval(coef2, s1)
    cxy2 = cxy1 + dfix[:, None] * nrm

    # ---- (5) 物理チェック ----
    sf, _ = _arclen_tangent(cxy2)
    dx  = np.gradient(cxy2[:, 0], sf)
    dy  = np.gradient(cxy2[:, 1], sf)
    ddx = np.gradient(dx, sf)
    ddy = np.gradient(dy, sf)
    kappa = np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2)**1.5
    core = slice(len(sf) // 10, -len(sf) // 10)
    kmax = np.nanmax(kappa[core])
    if kmax > 1e-9:
        print(f"物理チェック: 最小曲線半径 ≈ {1.0/kmax:.0f} m")
    else:
        print("物理チェック: ほぼ直線 (曲率 ≈ 0)")

    np.save(CENTERLINE_NPY, cxy2)
    print(f"保存しました: {CENTERLINE_NPY} ({len(cxy2)}点, 全長 {sf[-1]:.1f} m)")

    # ---- (6) 診断図 ----
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    ax = axes[0, 0]
    ax.scatter(df_all["X_meter"], df_all["Y_meter"], s=1, c="0.8", label="全軌跡点")
    ax.scatter(dfl["X_meter"], dfl["Y_meter"], s=1, c="tab:blue", alpha=0.3,
               label="基準車線の普通車軌跡点")
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
    ax.set_title("ばらつき: 各sビンのMAD")
    ax.set_xlabel("s [m]"); ax.set_ylabel("MAD [cm]")
    ax = axes[1, 1]
    ax.axhline(0, color="0.7", lw=1)
    ax.plot(sc[ok2], res2 * 100, "-o", ms=3)
    ax.set_title("残差: ビン中央値 − フィット")
    ax.set_xlabel("s [m]"); ax.set_ylabel("残差 [cm]")
    plt.tight_layout()
    diag = _out_png("diag")
    plt.savefig(diag, dpi=130); plt.close(fig)
    print(f"診断図を保存: {diag}")

    # ---- (7) 中心線重ね描き ----
    fig2, ax = plt.subplots(figsize=(8, 14))
    for _, g in df_all.groupby("Vehicle_ID"):
        g = g.sort_values("Frame")
        ax.plot(g.X_meter, g.Y_meter, color="0.45", lw=0.8, alpha=0.8, zorder=1)
    ax.plot(cxy2[:, 0], cxy2[:, 1], "r-", lw=2.5, zorder=3, label="生成した中心線")
    xr0, xr1 = df_all.X_meter.min(), df_all.X_meter.max()
    pad = 0.15 * (xr1 - xr0) + 1.0
    ax.set_xlim(xr0 - pad, xr1 + pad)
    yr0, yr1 = dfl.Y_meter.min(), dfl.Y_meter.max()
    ypad = 0.05 * (yr1 - yr0) + 2.0
    ax.set_ylim(yr0 - ypad, yr1 + ypad)
    if INVERT_Y:
        ax.invert_yaxis()
    ax.set_title("BEV座標と生成中心線")
    ax.set_xlabel("X_meter 横方向[m]"); ax.set_ylabel("Y_meter 進行方向[m]")
    ax.grid(True, color="0.9"); ax.set_axisbelow(True)
    ax.legend(loc="upper right")
    plt.tight_layout()
    ov = _out_png("overlay")
    plt.savefig(ov, dpi=130); plt.close(fig2)
    print(f"中心線重ね描きを保存: {ov}")


# ---------------------------------------------------------------
# 変換側
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

    df = pd.read_csv(CSV_PATH)                      # 変換は全車両（クラスで絞らない）
    xy = df[["X_meter", "Y_meter"]].to_numpy(dtype=float)
    s_out, d_out = cart_to_frenet(xy, cxy, s, tan)
    df["s_meter"] = s_out
    df["d_meter"] = d_out
    df["s_valid"] = (s_out >= S_VALID_MARGIN) & (s_out <= s[-1] - S_VALID_MARGIN)
    n_out = (~df["s_valid"]).sum()
    print(f"有効範囲: s = {S_VALID_MARGIN:.1f} 〜 {s[-1]-S_VALID_MARGIN:.1f} m "
          f"(範囲外の点 {n_out} / {len(df)})")

    out = os.path.splitext(CSV_PATH)[0] + "_frenet_auto.csv"
    df.to_csv(out, index=False)
    print(f"保存しました: {out}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if len(sys.argv) > 3:
        CSV_PATH = sys.argv[2]
        CENTERLINE_NPY = sys.argv[3]
    if mode in ("auto", "convert") and (CSV_PATH is None or CENTERLINE_NPY is None):
        sys.exit("CSV と CENTERLINE_NPY を引数で指定してください\n"
                 "  例: python frenet_transform.py auto scene_f0.500.csv centerline_f0.500.npy")
    if mode == "auto":
        build_centerline_auto()
    elif mode == "convert":
        convert_csv()
    else:
        print(__doc__)