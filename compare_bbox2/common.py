# -*- coding: utf-8 -*-
"""
common.py — compare_bbox2 の共通設定と関数

compare_bbox からの主な変更点:
  1. ホモグラフィは 4K(3840x2160) の画素座標を前提に作られているため、
     1080p の追跡座標を SCALE(=2) 倍してから変換する
  2. 中心線は f=0.5 で1本だけ作り、全 f で共通に使う
  3. 大型車はクラスIDではなく bbox の大きさ（同じ車線の普通車比）で定義する
  4. 大型車の車線は、評価する f に依存しない「占有」で判定する
     （同じ車線の、トラック車体と重なる位置に普通車は同時にいられない）

出力フォルダ: out_<CSV名>2  （例: out_P33C01_5min2）
"""

import os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # リポジトリ直下

# ============== 設定 ==============
# 評価に使うホモグラフィは「評価する車道」の路面用を指定する
CAMERAS = {
    "P33C01": dict(csv="P33C01_5min.csv", H="H_P33C01_WB.npy"),
    "P33C02": dict(csv="P33C02_5min.csv", H="H_P33C02_WB.npy"),
    "P33C03": dict(csv="P33C03_5min.csv", H="H_P33C03_EB.npy"),   # 中央分離帯越しに EB 車道を撮影（WB用HはNaN）
    "P33C04": dict(csv="P33C04_5min.csv", H="H_P33C04_WB.npy"),   # 手前の WB 車道を撮影（EB用HはNaN）
    "P33C05": dict(csv="P33C05_5min.csv", H="H_P33C05_WB.npy"),
    "P33C06": dict(csv="P33C06_5min.csv", H="H_P33C06_WB.npy"),
    # 15分版（IMGSZ=1280 で追跡）。キー名で指定して実行する: python compare_bbox2/run_all.py P33C06_15min
    "P33C06_15min": dict(csv="P33C06_15min.csv", H="H_P33C06_WB.npy"),
    "P33C05_15min": dict(csv="P33C05_15min.csv", H="H_P33C05_WB.npy"),
    "P33C02_15min": dict(csv="P33C02_15min.csv", H="H_P33C02_WB.npy"),
    "P33C01_15min": dict(csv="P33C01_15min.csv", H="H_P33C01_WB.npy"),
    "P33C04_15min": dict(csv="P33C04_15min.csv", H="H_P33C04_WB.npy"),
    "P33C03_15min": dict(csv="P33C03_15min.csv", H="H_P33C03_EB.npy"),
    # ポール P10（15分版, IMGSZ=1280）。P33 と同じくポールは WB 側、C03 だけ EB 車道を撮影
    "P10C01_15min": dict(csv="P10C01_15min.csv", H="H_P10C01_WB.npy"),
    "P10C02_15min": dict(csv="P10C02_15min.csv", H="H_P10C02_WB.npy"),
    "P10C03_15min": dict(csv="P10C03_15min.csv", H="H_P10C03_EB.npy"),
    "P10C04_15min": dict(csv="P10C04_15min.csv", H="H_P10C04_WB.npy"),
    "P10C05_15min": dict(csv="P10C05_15min.csv", H="H_P10C05_WB.npy"),
    "P10C06_15min": dict(csv="P10C06_15min.csv", H="H_P10C06_WB.npy"),
}
DEFAULT_CAMERAS = ["P33C01", "P33C02", "P33C03", "P33C04", "P33C05", "P33C06"]   # 引数なしで実行するカメラ
H_IMAGE_W = 3840          # ホモグラフィが作られた画像の幅（I-24 MOTION は 4K）
H_IMAGE_H = 2160
VIDEO_W   = 1920          # 追跡に使った動画の幅
SCALE     = H_IMAGE_W / VIDEO_W   # 追跡座標 → ホモグラフィの画素座標 の倍率（= 2.0）
FT_TO_M   = 12.0 / 39.37  # US survey foot → m（Hの出力は Tennessee State Plane [ft]）
F_VALUES  = [i / 8 for i in range(9)]   # 基準点の位置 f（0=下辺左端, 0.5=中央, 1=右端）
# ==================================


def base_name(cam):
    return os.path.splitext(os.path.basename(CAMERAS[cam]["csv"]))[0]


def out_dir(cam):
    d = os.path.join(ROOT, "out_" + base_name(cam) + "2")
    os.makedirs(d, exist_ok=True)
    return d


def out_path(cam, name):
    return os.path.join(out_dir(cam), f"{base_name(cam)}_{name}")


def load_H(cam):
    H = np.load(os.path.join(ROOT, CAMERAS[cam]["H"]))
    if not np.isfinite(H).all():
        raise ValueError(f"{CAMERAS[cam]['H']} に NaN が含まれています（この車道のHは使えません）")
    return H


def col(prefix, f):
    """f ごとの列名（例: d_0.500）"""
    return f"{prefix}_{f:.3f}"


def apply_homography(H, x, y):
    """追跡の画素座標 (x, y) [1080p] を路面座標 [m] に変換する。
    H は 4K 画素 → State Plane [ft] なので、画素を SCALE 倍してから変換し、ft → m にする。"""
    x = np.asarray(x, float) * SCALE
    y = np.asarray(y, float) * SCALE
    p = np.stack([x, y, np.ones_like(x)], axis=1) @ H.T
    return p[:, 0] / p[:, 2] * FT_TO_M, p[:, 1] / p[:, 2] * FT_TO_M


def camera_position(H, origin_m):
    """ホモグラフィからカメラの路面上の位置 [m] と高さ [m] を逆算する。
    画像中心を主点・正方画素と仮定し、回転行列の直交条件から焦点距離を求める。
    origin_m は数値誤差を避けるための局所原点（中心線の平均位置など）。"""
    cx, cy = H_IMAGE_W / 2, H_IMAGE_H / 2
    T = np.array([[1 / FT_TO_M, 0, origin_m[0] / FT_TO_M],
                  [0, 1 / FT_TO_M, origin_m[1] / FT_TO_M],
                  [0, 0, 1]])
    G = np.linalg.inv(H) @ T                      # 局所路面座標[m] → 4K画素
    G = G / np.linalg.norm(G[:, 0])
    g1, g2 = G[:, 0], G[:, 1]
    num = (g1[0] - cx * g1[2]) * (g2[0] - cx * g2[2]) + (g1[1] - cy * g1[2]) * (g2[1] - cy * g2[2])
    f2 = -num / (g1[2] * g2[2])
    if f2 <= 0:   # 直交条件が使えないときは長さ条件で求める
        a1 = (g1[0] - cx * g1[2]) ** 2 + (g1[1] - cy * g1[2]) ** 2
        a2 = (g2[0] - cx * g2[2]) ** 2 + (g2[1] - cy * g2[2]) ** 2
        f2 = (a1 - a2) / (g2[2] ** 2 - g1[2] ** 2)
    f = np.sqrt(abs(f2))
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
    B = np.linalg.inv(K) @ G
    lam = 1 / np.linalg.norm(B[:, 0])
    r1, r2, t = lam * B[:, 0], lam * B[:, 1], lam * B[:, 2]
    for sg in (1, -1):
        R = np.c_[sg * r1, sg * r2, np.cross(sg * r1, sg * r2)]
        C = -R.T @ (sg * t)
        if C[2] > 0:
            break
    return np.array(origin_m) + C[:2], C[2], f
