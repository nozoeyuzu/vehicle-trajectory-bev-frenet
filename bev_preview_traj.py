# # -*- coding: utf-8 -*-
# """
# bev_preview_traj.py — 実写BEV + メートルグリッド + 軌跡 + 中心線オーバーレイ

# bev_preview_centerline.py の拡張版。
# CSV の X_meter / Y_meter（=すでにBEV変換済みの世界座標）を、
# 実写BEV画像と同じ「世界座標→画素」の式で写して重ね描きする。
# ホモグラフィを再適用しないので、軌跡と背景は定義上ぴったり同じ座標系に乗る。

# 使い方:
#     python bev_preview_traj.py
# 出力:
#     bev_preview_traj.png

# 見るポイント（重要）:
#   A) 軌跡の束が白線ペンキと平行に流れているか
#        平行  → BEV空間は健全。カーブのずれは中心線側の問題
#        非平行(ペンキに対して斜行) → 本物の交通挙動 or カメラドリフト
#   B) 青の中心線がペンキと平行か（ど真ん中を通る必要はない、平行かだけ見る）
#   C) 軌跡の束と中心線の開き方が場所によって変わるか
#        → 変わる区間が、Y≈-25で見えていた「2mのずれ」の正体の場所
# """

# import cv2
# import numpy as np
# import pandas as pd

# # ==================== 設定 ====================
# VIDEO_PATH     = "rakkabutu_highway.mp4"
# H_NPY          = "homography.npy"
# CENTERLINE_NPY = "centerline_meter1.npy"    # 11点版の中心線
# CSV_PATH       = "traffic_trajectory_data2_recal.csv"
# OUT_PNG        = "bev_preview_traj.png"

# FRAME_IDX = 0            # 背景に使うフレーム番号（0=1枚目）

# # 表示する世界座標の範囲 [m] と解像度 [px/m]
# X0, X1 = -12.0, 16.0
# Y0, Y1 = -80.0, 66.0
# PPM    = 14.0            # 1mあたりの画素数

# # 背景の暗さ（0.0=真っ黒〜1.0=そのまま）。線を見やすくするため少し落とす
# BG_GAIN = 0.75

# # --- 軌跡の描画設定 ---
# DRAW_TRAJ    = True
# TRAJ_ALPHA   = 0.85      # 軌跡レイヤの不透明度（1.0で完全上書き）
# TRAJ_THICK   = 1
# MIN_POINTS   = 30        # これ未満の点数のtrackは描かない（ノイズ除去）
# TRACK_FILTER = None      # 例: [6, 302, 45] と書くとそのIDだけ描画。Noneで全部
# FRAME_RANGE  = None      # 例: (0, 800) と書くとその区間だけ描画。Noneで全部
# X_RANGE_TRAJ = None      # 例: (0.0, 16.0) でクリック側道路だけに限定。Noneで全部
# COLOR_MODE   = "track"   # "track"=IDごとに色分け / "single"=全部同色
# SINGLE_COLOR = (0, 0, 255)   # BGR（COLOR_MODE="single"のとき）
# DRAW_ENDPOINT = True     # 各軌跡の終点に小さな丸を打つ（進行方向の確認用）

# # --- 中心線の描画設定 ---
# DRAW_CENTERLINE = True
# OFFSETS_D       = [+4.5, +1.0, -2.5]   # 中心線からの法線方向オフセット [m]

# # --- CSVの列名（違っていればここを直す） ---
# COL_X, COL_Y = "X_meter", "Y_meter"
# COL_ID       = "Vehicle_ID"
# COL_FRAME    = "Frame"              # 無ければ自動で行順を使う
# # =============================================


# # ---------- 世界座標 → BEV画素 ----------
# W  = int(round((X1 - X0) * PPM))
# Hg = int(round((Y1 - Y0) * PPM))

# def wx2u(x):
#     return int(round((x - X0) * PPM))

# def wy2v(y):
#     return int(round((Y1 - y) * PPM))   # y上向き=画像上方向


# # ---------- 背景の実写BEVを作る ----------
# cap = cv2.VideoCapture(VIDEO_PATH)
# if FRAME_IDX > 0:
#     cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME_IDX)
# ok, frame = cap.read()
# cap.release()
# if not ok:
#     raise RuntimeError(f"動画が読めません: {VIDEO_PATH} (frame={FRAME_IDX})")

# H = np.load(H_NPY)                       # 画像画素 → 世界メートル
# A = np.array([[PPM, 0.0, -X0 * PPM],     # 世界メートル → BEV画素
#               [0.0, -PPM,  Y1 * PPM],
#               [0.0, 0.0,   1.0]], dtype=np.float64)
# M = A @ H
# bev = cv2.warpPerspective(frame, M, (W, Hg))
# bev = cv2.convertScaleAbs(bev, alpha=BG_GAIN, beta=0)


# # ---------- メートルグリッド ----------
# for x in range(int(np.ceil(X0)), int(np.floor(X1)) + 1):
#     u = wx2u(x)
#     major = (x % 5 == 0)
#     cv2.line(bev, (u, 0), (u, Hg - 1), (0, 220, 255), 2 if major else 1)
#     if major:
#         cv2.putText(bev, f"{x}", (u + 3, Hg - 6),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1, cv2.LINE_AA)

# for y in range(int(np.ceil(Y0)), int(np.floor(Y1)) + 1):
#     v = wy2v(y)
#     major = (y % 5 == 0)
#     cv2.line(bev, (0, v), (W - 1, v), (0, 220, 255), 2 if major else 1)
#     if major:
#         cv2.putText(bev, f"{y}", (4, v - 4),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1, cv2.LINE_AA)


# # ---------- 中心線とオフセット線 ----------
# def fit_centerline(raw, n_samples=3000):
#     """frenet_transform.py と同じ弦長パラメータ化＋多項式フィット。"""
#     seg = np.linalg.norm(np.diff(raw, axis=0), axis=1)
#     t = np.concatenate([[0.0], np.cumsum(seg)])
#     t = t / t[-1]
#     deg = min(3, len(raw) - 1)
#     px = np.polyfit(t, raw[:, 0], deg)
#     py = np.polyfit(t, raw[:, 1], deg)
#     ts = np.linspace(0.0, 1.0, n_samples)
#     cxy = np.stack([np.polyval(px, ts), np.polyval(py, ts)], axis=1)
#     d = np.gradient(cxy, axis=0)
#     tan = d / np.linalg.norm(d, axis=1, keepdims=True)
#     return cxy, tan


# def draw_world_polyline(img, pts_world, color, thick=2):
#     """世界座標の点列を画素に写してポリライン描画（範囲外は自動で切れる）。"""
#     prev = None
#     for x, y in pts_world:
#         u, v = wx2u(x), wy2v(y)
#         inside = (0 <= u < W) and (0 <= v < Hg)
#         if prev is not None and inside and prev[2]:
#             cv2.line(img, (prev[0], prev[1]), (u, v), color, thick, cv2.LINE_AA)
#         prev = (u, v, inside)


# cxy = None
# if DRAW_CENTERLINE:
#     raw = np.load(CENTERLINE_NPY)
#     cxy, tan = fit_centerline(raw)
#     normal = np.stack([-tan[:, 1], tan[:, 0]], axis=1)   # dの正方向


# # ---------- 軌跡 ----------
# if DRAW_TRAJ:
#     df = pd.read_csv(CSV_PATH)
#     for c in (COL_X, COL_Y, COL_ID):
#         if c not in df.columns:
#             raise KeyError(f"CSVに列 '{c}' がありません。列名: {list(df.columns)}")

#     if FRAME_RANGE is not None and COL_FRAME in df.columns:
#         f0, f1 = FRAME_RANGE
#         df = df[(df[COL_FRAME] >= f0) & (df[COL_FRAME] <= f1)]
#     if X_RANGE_TRAJ is not None:
#         xa, xb = X_RANGE_TRAJ
#         df = df[(df[COL_X] >= xa) & (df[COL_X] <= xb)]

#     rng = np.random.default_rng(0)
#     layer = bev.copy()
#     n_drawn = 0

#     for tid, g in df.groupby(COL_ID):
#         if len(g) < MIN_POINTS:
#             continue
#         if TRACK_FILTER is not None and tid not in TRACK_FILTER:
#             continue
#         if COL_FRAME in g.columns:
#             g = g.sort_values(COL_FRAME)

#         if COLOR_MODE == "track":
#             color = tuple(int(c) for c in rng.integers(70, 256, size=3))
#         else:
#             color = SINGLE_COLOR

#         pts = g[[COL_X, COL_Y]].to_numpy(dtype=float)
#         draw_world_polyline(layer, pts, color, TRAJ_THICK)

#         if DRAW_ENDPOINT:
#             u, v = wx2u(pts[-1, 0]), wy2v(pts[-1, 1])
#             if 0 <= u < W and 0 <= v < Hg:
#                 cv2.circle(layer, (u, v), 3, color, -1, cv2.LINE_AA)
#         n_drawn += 1

#     bev = cv2.addWeighted(layer, TRAJ_ALPHA, bev, 1.0 - TRAJ_ALPHA, 0)
#     print(f"軌跡を描画: {n_drawn} 本（{MIN_POINTS}点未満は除外）")

# # 中心線は軌跡より上に描く（比較したい線なので隠れないように）
# if DRAW_CENTERLINE:
#     for d in OFFSETS_D:
#         draw_world_polyline(bev, cxy + d * normal, (0, 200, 0), 1)
#         mid = cxy[len(cxy) // 2] + d * normal[len(cxy) // 2]
#         u, v = wx2u(mid[0]), wy2v(mid[1])
#         if 0 <= u < W and 0 <= v < Hg:
#             cv2.putText(bev, f"d={d:+.1f}", (u + 4, v),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1, cv2.LINE_AA)
#     draw_world_polyline(bev, cxy, (255, 80, 0), 2)      # 中心線（青・太）
#     for x, y in raw:                                    # クリック生点（赤）
#         u, v = wx2u(x), wy2v(y)
#         if 0 <= u < W and 0 <= v < Hg:
#             cv2.circle(bev, (u, v), 4, (0, 0, 255), -1)


# cv2.imwrite(OUT_PNG, bev)
# print(f"保存しました: {OUT_PNG}  ({W}x{Hg}px)")
# print("チェック:")
# print("  A) 軌跡の束が白線ペンキと平行か（平行ならBEV空間は健全）")
# print("  B) 青の中心線がペンキと平行か（ど真ん中でなくてよい）")
# print("  C) 軌跡束と中心線の開き方が場所で変わるか（変わる区間=ずれの発生源）")

# -*- coding: utf-8 -*-
"""
bev_preview_traj2.py — 実写BEV + 軌跡 + 中心線（読みやすさ改善版）

前版(bev_preview_traj.py)の問題を修正:
  - 表示範囲が道路とずれていた → X範囲を軌跡から自動決定
  - 縦長すぎて角度が判定できない → 90度回転して道路を横向きに
  - 1mグリッドが路面を塗りつぶす → 10m間隔のみ / オフ可
  - 軌跡を全部描いて壁になる → 車線ごとの代表線 or 指定IDのみ
  - 全体図では細部が見えない → 指定Y位置の拡大パネルを別途出力

出力:
  bev_traj_overview.png   全体図（回転済み・間引き済み）
  bev_traj_zoom_Y+40.png  など、ZOOM_CENTERS の数だけ拡大パネル

使い方:
    python bev_preview_traj2.py
"""

import cv2
import numpy as np
import pandas as pd

# ==================== 入力 ====================
VIDEO_PATH     = "rakkabutu_highway.mp4"
H_NPY          = "homography.npy"
CENTERLINE_NPY = "centerline_meter1.npy"
CSV_PATH       = "traffic_trajectory_data2_recal.csv"

COL_X, COL_Y = "X_meter", "Y_meter"
COL_ID       = "Vehicle_ID"
COL_FRAME    = "Frame"
COL_CLASS    = "Class_ID"
CLASS_KEEP   = None          # 例: [2, 5, 7] で車両クラスのみ。Noneで全部

FRAME_IDX = 0                # 背景に使うフレーム

# ==================== 全体図の設定 ====================
AUTO_X_RANGE = True          # 軌跡の分布からX範囲を自動決定
X_MARGIN     = 3.0           # 自動決定時の左右余白 [m]
X0, X1       = -14.0, 2.0    # AUTO_X_RANGE=False のときだけ使う
Y0, Y1       = -80.0, 66.0

PPM_OVERVIEW = 6.0           # 全体図の解像度 [px/m]（低めで十分）
ROTATE       = True          # 90度回転して道路を横向きにする
BG_GAIN      = 1.15          # 背景の明るさ（1.0=そのまま。ペンキを見たいので明るめ）

GRID_MODE    = "major"       # "major"=10mのみ / "none"=なし / "fine"=1m+5m
GRID_STEP    = 10            # major時の間隔 [m]

# ==================== 軌跡の描画 ====================
TRAJ_MODE    = "representative"
# "representative": 車線帯ごとに中央値的な1本を選んで描く（推奨・見やすい）
# "sample"        : ランダムにN本だけ描く
# "ids"           : TRACK_IDS で指定したIDだけ描く
# "all"           : 全部（前版と同じ。非推奨）

N_SAMPLE   = 8               # "sample" のときの本数
TRACK_IDS  = [6]             # "ids" のときの対象（6=停止車両）
MIN_POINTS = 60              # これ未満の点数のtrackは対象外
MIN_SPAN_Y = 40.0            # Y方向にこれ以上走っていない軌跡は除外（横切り等を排除）
LANE_EDGES = None            # 例: [-9.0, -6.0, -3.0, 0.0] で車線帯を手動指定。Noneで自動3帯

TRAJ_THICK = 2

# ==================== 中心線 ====================
DRAW_CENTERLINE = True
OFFSETS_D       = [+4.5, +1.0, -2.5]

# ==================== 拡大パネル ====================
ZOOM_CENTERS = [40.0, 0.0, -25.0, -60.0]   # 切り出す中心Y [m]（Noneで無効）
ZOOM_SIZE_Y  = 24.0                        # 切り出す縦幅 [m]
PPM_ZOOM     = 26.0                        # 拡大パネルの解像度 [px/m]
# =============================================


# ---------- 共通ユーティリティ ----------
def make_bev(frame, H, x0, x1, y0, y1, ppm, bg_gain):
    W  = int(round((x1 - x0) * ppm))
    Hg = int(round((y1 - y0) * ppm))
    A = np.array([[ppm, 0.0, -x0 * ppm],
                  [0.0, -ppm,  y1 * ppm],
                  [0.0, 0.0,   1.0]], dtype=np.float64)
    img = cv2.warpPerspective(frame, A @ H, (W, Hg))
    img = cv2.convertScaleAbs(img, alpha=bg_gain, beta=0)
    return img, W, Hg


def mk_mapper(x0, y1, ppm):
    def wx2u(x): return int(round((x - x0) * ppm))
    def wy2v(y): return int(round((y1 - y) * ppm))
    return wx2u, wy2v


def draw_polyline(img, pts, wx2u, wy2v, W, Hg, color, thick=2):
    prev = None
    for x, y in pts:
        u, v = wx2u(x), wy2v(y)
        inside = (0 <= u < W) and (0 <= v < Hg)
        if prev is not None and inside and prev[2]:
            cv2.line(img, (prev[0], prev[1]), (u, v), color, thick, cv2.LINE_AA)
        prev = (u, v, inside)


def fit_centerline(raw, n_samples=3000):
    seg = np.linalg.norm(np.diff(raw, axis=0), axis=1)
    t = np.concatenate([[0.0], np.cumsum(seg)]); t /= t[-1]
    deg = min(3, len(raw) - 1)
    px, py = np.polyfit(t, raw[:, 0], deg), np.polyfit(t, raw[:, 1], deg)
    ts = np.linspace(0.0, 1.0, n_samples)
    cxy = np.stack([np.polyval(px, ts), np.polyval(py, ts)], axis=1)
    d = np.gradient(cxy, axis=0)
    return cxy, d / np.linalg.norm(d, axis=1, keepdims=True)


def draw_grid(img, x0, x1, y0, y1, wx2u, wy2v, W, Hg, mode, step):
    if mode == "none":
        return
    col = (0, 200, 240)
    if mode == "fine":
        xs = range(int(np.ceil(x0)), int(np.floor(x1)) + 1)
        ys = range(int(np.ceil(y0)), int(np.floor(y1)) + 1)
        majors = 5
    else:
        xs = range(int(np.ceil(x0 / step) * step), int(np.floor(x1)) + 1, step)
        ys = range(int(np.ceil(y0 / step) * step), int(np.floor(y1)) + 1, step)
        majors = step
    for x in xs:
        u = wx2u(x)
        cv2.line(img, (u, 0), (u, Hg - 1), col, 2 if x % majors == 0 else 1)
        if x % majors == 0:
            cv2.putText(img, f"{x}", (u + 3, Hg - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
    for y in ys:
        v = wy2v(y)
        cv2.line(img, (0, v), (W - 1, v), col, 2 if y % majors == 0 else 1)
        if y % majors == 0:
            cv2.putText(img, f"{y}", (4, v - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)


# ---------- 読み込み ----------
cap = cv2.VideoCapture(VIDEO_PATH)
if FRAME_IDX > 0:
    cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME_IDX)
ok, frame = cap.read(); cap.release()
if not ok:
    raise RuntimeError(f"動画が読めません: {VIDEO_PATH}")
H = np.load(H_NPY)

df = pd.read_csv(CSV_PATH)
if CLASS_KEEP is not None and COL_CLASS in df.columns:
    df = df[df[COL_CLASS].isin(CLASS_KEEP)]

# 軌跡の候補を作る
tracks = []
for tid, g in df.groupby(COL_ID):
    if len(g) < MIN_POINTS:
        continue
    g = g.sort_values(COL_FRAME) if COL_FRAME in g.columns else g
    pts = g[[COL_X, COL_Y]].to_numpy(dtype=float)
    if pts[:, 1].max() - pts[:, 1].min() < MIN_SPAN_Y:
        continue
    tracks.append((int(tid), pts))
print(f"候補軌跡: {len(tracks)} 本")

all_x = np.concatenate([p[:, 0] for _, p in tracks]) if tracks else np.array([0.0])
if AUTO_X_RANGE:
    X0 = float(np.floor(np.percentile(all_x, 0.5) - X_MARGIN))
    X1 = float(np.ceil(np.percentile(all_x, 99.5) + X_MARGIN))
print(f"X範囲: {X0} 〜 {X1} m")

# 描画対象の選抜
if TRAJ_MODE == "all":
    sel = tracks
elif TRAJ_MODE == "ids":
    sel = [t for t in tracks if t[0] in TRACK_IDS]
elif TRAJ_MODE == "sample":
    rng = np.random.default_rng(0)
    idx = rng.choice(len(tracks), size=min(N_SAMPLE, len(tracks)), replace=False)
    sel = [tracks[i] for i in idx]
else:  # representative
    med = np.array([np.median(p[:, 0]) for _, p in tracks])
    if LANE_EDGES is None:
        qs = np.percentile(med, [0, 33, 66, 100])
        edges = list(qs)
    else:
        edges = LANE_EDGES
    sel = []
    for a, b in zip(edges[:-1], edges[1:]):
        band = [i for i in range(len(tracks)) if a <= med[i] <= b]
        if not band:
            continue
        # 帯の中で最も「中央値的」かつ長い軌跡を選ぶ
        center = np.median(med[band])
        band.sort(key=lambda i: (abs(med[i] - center), -len(tracks[i][1])))
        sel += [tracks[i] for i in band[:2]]
print(f"描画する軌跡: {[t[0] for t in sel]}")

cxy = normal = raw = None
if DRAW_CENTERLINE:
    raw = np.load(CENTERLINE_NPY)
    cxy, tan = fit_centerline(raw)
    normal = np.stack([-tan[:, 1], tan[:, 0]], axis=1)

PALETTE = [(0, 0, 255), (0, 255, 255), (255, 0, 255),
           (0, 165, 255), (255, 255, 0), (128, 0, 255),
           (0, 255, 128), (255, 128, 0)]


def render(x0, x1, y0, y1, ppm, grid_mode, grid_step, rotate, out_path):
    img, W, Hg = make_bev(frame, H, x0, x1, y0, y1, ppm, BG_GAIN)
    wx2u, wy2v = mk_mapper(x0, y1, ppm)

    draw_grid(img, x0, x1, y0, y1, wx2u, wy2v, W, Hg, grid_mode, grid_step)

    for k, (tid, pts) in enumerate(sel):
        draw_polyline(img, pts, wx2u, wy2v, W, Hg,
                      PALETTE[k % len(PALETTE)], TRAJ_THICK)

    if DRAW_CENTERLINE:
        for d in OFFSETS_D:
            draw_polyline(img, cxy + d * normal, wx2u, wy2v, W, Hg, (0, 200, 0), 1)
        draw_polyline(img, cxy, wx2u, wy2v, W, Hg, (255, 80, 0), 2)
        for x, y in raw:
            u, v = wx2u(x), wy2v(y)
            if 0 <= u < W and 0 <= v < Hg:
                cv2.circle(img, (u, v), 4, (60, 60, 255), -1)

    if rotate:
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    cv2.imwrite(out_path, img)
    print(f"保存: {out_path}  ({img.shape[1]}x{img.shape[0]}px)")


# 全体図
render(X0, X1, Y0, Y1, PPM_OVERVIEW, GRID_MODE, GRID_STEP, ROTATE,
       "bev_traj_overview.png")

# 拡大パネル
if ZOOM_CENTERS:
    for yc in ZOOM_CENTERS:
        render(X0, X1, yc - ZOOM_SIZE_Y / 2, yc + ZOOM_SIZE_Y / 2,
               PPM_ZOOM, "major", 5, ROTATE,
               f"bev_traj_zoom_Y{yc:+.0f}.png")

print("\n見るポイント:")
print("  1) 拡大パネルで白線ペンキが見えるか（見えない場所=判定不能な領域）")
print("  2) 軌跡(カラー)がペンキと平行か → 平行ならBEVは健全")
print("  3) 青の中心線がペンキと平行か → 非平行なら中心線の形が原因")