# -*- coding: utf-8 -*-
"""
bev_preview.py — 推定したホモグラフィの目視検証

homography.npy を使って動画の1フレーム目を真上から見た画像(BEV)に変換し、
1m/5mのメートルグリッドを重ねて保存する。

使い方:
    python bev_preview.py
出力:
    bev_preview.png

正しく校正できていれば:
  - 破線ペンキ1本の長さ = グリッド8m分
  - 破線どうしの間隔 = どこでも3.5m
  - 白線が滑らかな曲線(ガタつき・広がり・すぼまりがない)
どこかが崩れていれば、その場所の仮定(規格・線の選択・クリック)が間違っている。
"""

import cv2
import numpy as np

VIDEO_PATH = "rakkabutu_highway.mp4"
H_NPY      = "homography.npy"
# 表示する世界座標の範囲 [m] と解像度
X0, X1 = -12.0, 16.0     # 横方向
Y0, Y1 = -50.0, 70.0     # 道なり方向(校正原点から)
SCALE  = 15              # 1mあたりのピクセル数

H = np.load(H_NPY)
cap = cv2.VideoCapture(VIDEO_PATH)
ok, frame = cap.read()
cap.release()
if not ok:
    raise RuntimeError(f"動画を読めません: {VIDEO_PATH}")

W  = int((X1 - X0) * SCALE)
Hg = int((Y1 - Y0) * SCALE)

# 出力画素(u,v) → 世界座標(x,y)。yは上向きが奥になるように反転
S = np.array([[1.0 / SCALE, 0.0, X0],
              [0.0, -1.0 / SCALE, Y1],
              [0.0, 0.0, 1.0]])
# 出力画素 → 世界 → 画像画素
M = np.linalg.inv(H) @ S
bev = cv2.warpPerspective(frame, M, (W, Hg), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)

# ---- メートルグリッドを重ねる ----
def wx2u(x):  return int(round((x - X0) * SCALE))
def wy2v(y):  return int(round((Y1 - y) * SCALE))

for x in np.arange(np.ceil(X0), X1 + 0.01, 1.0):
    thick = 2 if abs(x % 5) < 1e-6 else 1
    col = (0, 220, 255) if thick == 2 else (0, 120, 140)
    cv2.line(bev, (wx2u(x), 0), (wx2u(x), Hg - 1), col, thick)
    if thick == 2:
        cv2.putText(bev, f"{x:.0f}", (wx2u(x) + 3, Hg - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1, cv2.LINE_AA)

for y in np.arange(np.ceil(Y0), Y1 + 0.01, 1.0):
    thick = 2 if abs(y % 5) < 1e-6 else 1
    col = (0, 220, 255) if thick == 2 else (0, 120, 140)
    cv2.line(bev, (0, wy2v(y)), (W - 1, wy2v(y)), col, thick)
    if thick == 2:
        cv2.putText(bev, f"{y:.0f}", (4, wy2v(y) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1, cv2.LINE_AA)

cv2.imwrite("bev_preview1.png", bev)
print("保存しました: bev_preview.png")
print("チェック: ペンキ長=8マス / 線間隔=3.5マス / 線が滑らかか")