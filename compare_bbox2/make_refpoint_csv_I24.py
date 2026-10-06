# -*- coding: utf-8 -*-
"""
make_refpoint_csv_I24.py — bbox下辺上の基準点 f ごとに路面座標を作る（4K換算あり）

使い方:
    python compare_bbox2/make_refpoint_csv_I24.py P33C01

処理:
    bbox基準点 (1080p画素) → ×SCALE(=2) で 4K画素 → Homography → State Plane [ft] → [m]

出力（out_<CSV名>2/）:
    <CSV名>_refpoints.csv  元の列 + f ごとの X_meter_f, Y_meter_f
    <CSV名>_f0.500.csv     X_meter, Y_meter = 下辺中央（中心線の自動生成に使う）
"""

import sys
import numpy as np
import pandas as pd

from common import CAMERAS, DEFAULT_CAMERAS, ROOT, F_VALUES, SCALE, base_name, out_path, load_H, apply_homography, col
import os


def main(cam):
    H = load_H(cam)
    df = pd.read_csv(os.path.join(ROOT, CAMERAS[cam]["csv"]))
    print(f"[{cam}] 入力 {CAMERAS[cam]['csv']} ({len(df)}行, {df.Vehicle_ID.nunique()}台)  H={CAMERAS[cam]['H']}  SCALE={SCALE}")

    # X_pixel / Y_pixel が下辺中央かの確認（元コードと同じ）
    err = np.abs((df.BBox_x1 + df.BBox_x2) / 2 - df.X_pixel).max()
    print(f"  bbox下辺中央 vs X_pixel 最大差: {err:.4f} px")

    out = df.copy()
    for f in F_VALUES:
        px = df.BBox_x1 + f * (df.BBox_x2 - df.BBox_x1)     # 下辺上を左端から f の位置
        X, Y = apply_homography(H, px, df.BBox_y2)
        out[col("X_meter", f)] = X
        out[col("Y_meter", f)] = Y
    out.to_csv(out_path(cam, "refpoints.csv"), index=False, float_format="%.4f")

    c = df.copy()
    c["X_meter"], c["Y_meter"] = out[col("X_meter", 0.5)], out[col("Y_meter", 0.5)]
    c.to_csv(out_path(cam, "f0.500.csv"), index=False, float_format="%.4f")
    print(f"  保存: {out_path(cam, 'refpoints.csv')}")


if __name__ == "__main__":
    for cam in (sys.argv[1:] or DEFAULT_CAMERAS):
        main(cam)
