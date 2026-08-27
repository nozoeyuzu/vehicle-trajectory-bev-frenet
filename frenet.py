# -*- coding: utf-8 -*-
"""
frenet_transform.py — 固定カメラBEV軌跡のFrenet座標(s, d)変換

使い方（2段階）:
  1) 中心線のクリック取得（1回だけやればよい）
       python frenet_transform.py click
     → 動画の1フレーム目が表示される。道路の中心線（または基準にしたい
       車線の中心）に沿って、奥から手前へ順にクリックしていく。
       白線が見えるなら白線の間の中央を狙う。5〜15点で十分。
       [u] 1つ戻す / [Enter or q] 確定して保存
     → centerline_meter.npy が保存される（BEVメートル座標）

  2) 軌跡CSVへの変換
       python frenet_transform.py convert
     → CSV に s_meter, d_meter, d_dev 列を追加した
       *_frenet.csv を保存する

出力列の意味:
  s_meter : 中心線に沿った道なり距離 [m]（クリックした向きが正）
  d_meter : 中心線からの符号付き横距離 [m]（進行方向に向かって左が正）
  d_dev   : 「最寄り車線中心」からのずれ [m] ← ふくらみ解析ではこれを使う
"""

import os
import sys
import numpy as np
import pandas as pd

# ============== 設定（既存スクリプトと合わせる） ==============
VIDEO_PATH     = "rakkabutu_highway.mp4"
CSV_PATH       = "traffic_trajectory_data2_recal.csv"
CENTERLINE_NPY = "centerline_meter1.npy"

# 既存コードと同じホモグラフィ
# SRC_PTS = np.float32([
#     [635, 180],
#     [725, 180],
#     [407, 348],
#     [704, 345],
# ])
# DST_PTS = np.float32([[0.0, 0.0], [3.5, 0.0], [0.0, 8.0], [3.5, 8.0]])

# 車線中心の d オフセット [m]。
# 例: クリックした中心線が2車線の境界なら [-1.75, +1.75]
#     クリックした線自体が1つの車線中心なら [0.0]（隣の車線があれば +3.5 等を追加）
# 空リスト [] にすると「車両ごとの中央値」を基準にするフォールバックになる。
LANE_CENTERS_D = [-4.6, -1.1, 2.4]

POLY_DEG   = 3      # 中心線フィットの多項式次数（緩いカーブなら3で十分）
N_SAMPLES  = 3000   # 中心線の密なサンプル数（≈数cm刻みになる）
# =============================================================


def click_centerline():
    """動画1フレーム目に中心線の点をクリックし、BEVメートル座標で保存する。"""
    import cv2
    cap = cv2.VideoCapture(VIDEO_PATH)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"動画を読めません: {VIDEO_PATH}")

    matrix = np.load("homography.npy")
    pts_px = []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            pts_px.append((x, y))

    win = "centerline: click along road center (far -> near), u=undo, q=done"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)

    while True:
        disp = frame.copy()
        for i, p in enumerate(pts_px):
            cv2.circle(disp, p, 4, (0, 0, 255), -1)
            if i > 0:
                cv2.line(disp, pts_px[i - 1], p, (0, 255, 0), 1)
        cv2.imshow(win, disp)
        k = cv2.waitKey(30) & 0xFF
        if k == ord("u") and pts_px:
            pts_px.pop()
        elif k in (ord("q"), 13):  # q or Enter
            break
    cv2.destroyAllWindows()

    if len(pts_px) < 3:
        raise RuntimeError("3点以上クリックしてください")

    px = np.float32(pts_px).reshape(-1, 1, 2)
    mt = cv2.perspectiveTransform(px, matrix).reshape(-1, 2)
    np.save(CENTERLINE_NPY, mt)
    print(f"保存しました: {CENTERLINE_NPY}  ({len(mt)}点)")
    print(mt.round(2))


def fit_centerline(raw_pts, deg=POLY_DEG, n=N_SAMPLES):
    """クリック点列を滑らかな曲線にフィットし、密なサンプル・弧長・接線を返す。

    x(t), y(t) をそれぞれ弦長パラメータ t の多項式でフィットする。
    次数を低く保つことで、局所的な凹凸は表現できない＝道路の大局形状だけを表す。
    """
    p = np.asarray(raw_pts, dtype=float)
    # 弦長パラメータ（0..1）
    seg = np.hypot(*np.diff(p, axis=0).T)
    t = np.concatenate([[0.0], np.cumsum(seg)])
    t = t / t[-1]

    deg = min(deg, len(p) - 1)
    cx = np.polyfit(t, p[:, 0], deg)
    cy = np.polyfit(t, p[:, 1], deg)

    tt = np.linspace(0.0, 1.0, n)
    cxy = np.stack([np.polyval(cx, tt), np.polyval(cy, tt)], axis=1)  # (n,2)

    # 弧長 s と単位接線
    dseg = np.hypot(*np.diff(cxy, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(dseg)])
    tan = np.gradient(cxy, axis=0)
    tan /= np.linalg.norm(tan, axis=1, keepdims=True) + 1e-12
    return cxy, s, tan


def cart_to_frenet(xy, cxy, s, tan):
    """点群 xy (N,2) を (s, d) に変換する。

    d の符号: 進行方向（クリックの向き）に向かって左が正。
    """
    from scipy.spatial import cKDTree
    tree = cKDTree(cxy)
    _, idx = tree.query(xy, k=1)
    near = cxy[idx]
    tg = tan[idx]
    rel = xy - near
    # 接線方向の残差で s を微修正（最近傍サンプルの離散誤差を補正）
    s_out = s[idx] + np.einsum("ij,ij->i", rel, tg)
    # 外積の z 成分で符号付き横距離
    d_out = tg[:, 0] * rel[:, 1] - tg[:, 1] * rel[:, 0]
    return s_out, d_out


def convert_csv():
    if not os.path.exists(CENTERLINE_NPY):
        raise RuntimeError("先に `python frenet_transform.py click` を実行してください")
    raw = np.load(CENTERLINE_NPY)
    cxy, s, tan = fit_centerline(raw)
    print(f"中心線フィット完了: 全長 {s[-1]:.1f} m, サンプル間隔 {s[-1]/len(s)*100:.1f} cm")

    df = pd.read_csv(CSV_PATH)
    xy = df[["X_meter", "Y_meter"]].to_numpy(dtype=float)
    s_out, d_out = cart_to_frenet(xy, cxy, s, tan)
    df["s_meter"] = s_out
    df["d_meter"] = d_out

    # --- ふくらみ解析用の「ずれ」列 d_dev ---
    if LANE_CENTERS_D:
        lanes = np.asarray(LANE_CENTERS_D, dtype=float)
        nearest = lanes[np.argmin(np.abs(d_out[:, None] - lanes[None, :]), axis=1)]
        df["d_dev"] = d_out - nearest
        print(f"d_dev = d - 最寄り車線中心 {LANE_CENTERS_D} で計算")
    else:
        # フォールバック: 車両ごとの d の中央値を基準にする
        # （回避が軌跡の半分未満なら中央値は車線追従値に留まる、という頑健性を利用）
        df["d_dev"] = df["d_meter"] - df.groupby("Vehicle_ID")["d_meter"].transform("median")
        print("d_dev = d - 車両ごとの中央値 で計算（LANE_CENTERS_D 未設定）")

    out = os.path.splitext(CSV_PATH)[0] + "_frenet.csv"
    df.to_csv(out, index=False)
    print(f"保存しました: {out}")
    print(df[["s_meter", "d_meter", "d_dev"]].describe().round(2))


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "click":
        click_centerline()
    elif mode == "convert":
        convert_csv()
    else:
        print(__doc__)