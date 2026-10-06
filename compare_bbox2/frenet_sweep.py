# -*- coding: utf-8 -*-
"""
frenet_sweep.py — f=0.5 で中心線を1本作り、全 f の基準点を同じ中心線で Frenet 変換する

使い方:
    python compare_bbox2/frenet_sweep.py P33C01      （make_refpoint_csv_I24.py の後）

f ごとに中心線を作り直すと、基準線が別の車線に移ることがある（旧 C02 の f=0 など）ため、
中心線は下辺中央 (f=0.5) の軌跡から1本だけ作る。クラス間差は車線ごとの普通車中心からの
ずれで測るので、中心線を共通にしても評価は変わらない。

出力（out_<CSV名>2/）:
    centerline_f0.500.npy (+ 診断図)   frenet_transform.py の auto と同じ
    <CSV名>_frenet.csv                 元の列 + f ごとの s_f, d_f + s_valid
"""

import sys
import numpy as np
import pandas as pd

import frenet_transform as ft
from common import CAMERAS, DEFAULT_CAMERAS, F_VALUES, out_dir, out_path, col
import os

S_VALID_MARGIN = 1.0   # 中心線の端からこの距離以内は外挿扱い [m]


def main(cam):
    npy = os.path.join(out_dir(cam), "centerline_f0.500.npy")
    ft.CSV_PATH = out_path(cam, "f0.500.csv")
    ft.CENTERLINE_NPY = npy
    print(f"[{cam}] 中心線を作成（f=0.5）")
    ft.build_centerline_auto()

    cxy, s, tan = ft.fit_centerline(np.load(npy))
    df = pd.read_csv(out_path(cam, "refpoints.csv"))
    keep = ["Frame", "Time_sec", "Vehicle_ID", "Class_ID", "Conf", "X_pixel", "Y_pixel",
            "BBox_x1", "BBox_y1", "BBox_x2", "BBox_y2"]
    out = df[[c for c in keep if c in df.columns]].copy()
    for f in F_VALUES:
        xy = df[[col("X_meter", f), col("Y_meter", f)]].to_numpy(float)
        so, do = ft.cart_to_frenet(xy, cxy, s, tan)
        out[col("s", f)] = so
        out[col("d", f)] = do
    s05 = out[col("s", 0.5)]
    out["s_valid"] = (s05 >= S_VALID_MARGIN) & (s05 <= s[-1] - S_VALID_MARGIN)
    out.to_csv(out_path(cam, "frenet.csv"), index=False, float_format="%.4f")
    print(f"  中心線長 {s[-1]:.1f} m, 有効点 {out.s_valid.mean():.1%}  保存: {out_path(cam, 'frenet.csv')}")


if __name__ == "__main__":
    for cam in (sys.argv[1:] or DEFAULT_CAMERAS):
        main(cam)
