# -*- coding: utf-8 -*-
"""
make_refpoint_csv.py — bboxの基準点を変えて X_meter, Y_meter を作り直す

使い方:
    python make_refpoint_csv.py

入力: 追跡CSV（BBox_x1, BBox_x2, BBox_y2 列を持つもの）とホモグラフィ .npy
出力: 基準点の位置 f ごとに X_meter, Y_meter を差し替えたCSV
        <元の名前>_f0.000.csv   下辺左端  (x1, y2)
        <元の名前>_f0.500.csv   下辺中央  ((x1+x2)/2, y2)  ← 既存と同じ
        <元の名前>_f1.000.csv   下辺右端  (x2, y2)
        ほか掃引用の f

このあと run_sweep.py（または frenet_transform.py を f ごとに auto → convert）で
Frenet変換し、evaluate_refpoint.py に渡す。
"""

import os
import numpy as np
import pandas as pd

# ============== 設定 ==============
CSV_PATH = "scene_A_track2_recal2.csv"
H_PATH   = "homography_scene_A_track3.npy"
BASE_NAME = os.path.splitext(os.path.basename(CSV_PATH))[0]
OUT_DIR   = "out_" + BASE_NAME       # 生成物をまとめて置くフォルダ。CSV名から自動で決まる（例: out_scene_A_track_recal）
# 基準点の位置 f（0=下辺左端, 0.5=中央, 1=右端）。主評価は 0, 0.5, 1、残りは掃引用（等間隔にして範囲を偏らせない）
F_VALUES = [i / 8 for i in range(9)]   # 0, 0.125, ..., 1 の対称な9点
# ==================================


def apply_homography(H, x, y):
    """画素座標 (x, y) を H で路面座標に変換する。"""
    p = np.stack([x, y, np.ones_like(x)], axis=1) @ H.T
    return p[:, 0] / p[:, 2], p[:, 1] / p[:, 2]


def main():
    H = np.load(H_PATH)
    df = pd.read_csv(CSV_PATH)
    os.makedirs(OUT_DIR, exist_ok=True)
    base = os.path.join(OUT_DIR, BASE_NAME)

    # 既存の X_meter/Y_meter が「下辺中央 × このH」で作られているか確認する
    Xc, Yc = apply_homography(H, df["X_pixel"].to_numpy(float), df["BBox_y2"].to_numpy(float))
    err = np.hypot(Xc - df["X_meter"], Yc - df["Y_meter"]).max()
    print(f"整合性チェック: 下辺中央をHで変換 vs 既存 X_meter/Y_meter の最大差 {err:.2e} m")
    if err > 1e-3:
        print("  警告: 既存のX_meter/Y_meterはこのHで作られていません。H_PATHを確認してください")

    for f in F_VALUES:
        px = df["BBox_x1"] + f * (df["BBox_x2"] - df["BBox_x1"])   # 下辺上を左端から f の位置
        py = df["BBox_y2"]
        X, Y = apply_homography(H, px.to_numpy(float), py.to_numpy(float))
        out = df.copy()
        out["X_meter"] = X
        out["Y_meter"] = Y
        out["ref_f"] = f
        path = f"{base}_f{f:.3f}.csv"
        out.to_csv(path, index=False)
        print(f"保存: {path}  (f={f}, X_meter 中央値 {np.median(X):+.2f} m)")


if __name__ == "__main__":
    main()