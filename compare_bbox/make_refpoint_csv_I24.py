# -*- coding: utf-8 -*-
"""
make_refpoint_csv_I24.py
I-24 MOTION用：bbox下辺上の基準点を変えて X_meter, Y_meter を作る。

処理:
    bbox基準点(pixel)
        ↓
    Homography
        ↓
    Tennessee State Plane座標 [US survey ft]
        ↓ × 12 / 39.37
    X_meter, Y_meter [m]

使い方:
    python make_refpoint_csv_I24.py

入力:
    追跡CSV
    ホモグラフィ .npy

出力:
    <元の名前>_f0.000.csv
    <元の名前>_f0.125.csv
    ...
    <元の名前>_f1.000.csv

f:
    0.0 = bbox下辺左端
    0.5 = bbox下辺中央
    1.0 = bbox下辺右端
"""

import os
import numpy as np
import pandas as pd


# ====================================================
# 設定
# ====================================================

CSV_PATH = "P33C06_5min.csv"

# カメラに近い側の道路面に対応するHを指定する
# 例：近い側がEBなら EB用
H_PATH = "H_P33C06_WB.npy"
BASE_NAME = os.path.splitext(os.path.basename(CSV_PATH))[0]
OUT_DIR = "out_" + BASE_NAME

# 0, 0.125, ..., 1.0 の9点
F_VALUES = [i / 8 for i in range(9)]

# US survey foot → meter
FT_TO_M = 12.0 / 39.37


# ====================================================
# Homography
# ====================================================

def apply_homography(H, x, y):
    """
    画素座標 (x, y) を
    Tennessee State Plane座標 [US survey ft] に変換する。
    """

    p = np.stack(
        [x, y, np.ones_like(x)],
        axis=1
    ) @ H.T

    X_TN_ft = p[:, 0] / p[:, 2]
    Y_TN_ft = p[:, 1] / p[:, 2]

    return X_TN_ft, Y_TN_ft


# ====================================================
# main
# ====================================================

def main():

    H = np.load(H_PATH)
    df = pd.read_csv(CSV_PATH)

    os.makedirs(OUT_DIR, exist_ok=True)

    base = os.path.join(
        OUT_DIR,
        BASE_NAME
    )

    print(f"入力CSV : {CSV_PATH}")
    print(f"Homography: {H_PATH}")
    print(f"行数    : {len(df)}")

    if "Vehicle_ID" in df.columns:
        print(f"車両数  : {df.Vehicle_ID.nunique()}")

    # ------------------------------------------------
    # 既存 X_pixel がbbox下辺中央か確認
    # ------------------------------------------------

    center_x = (
        df["BBox_x1"].to_numpy(float)
        + df["BBox_x2"].to_numpy(float)
    ) / 2.0

    center_y = df["BBox_y2"].to_numpy(float)

    if "X_pixel" in df.columns:
        err_x = np.abs(
            center_x
            - df["X_pixel"].to_numpy(float)
        ).max()

        print(
            f"bbox下辺中央 vs X_pixel 最大差: "
            f"{err_x:.6f} px"
        )

    if "Y_pixel" in df.columns:
        err_y = np.abs(
            center_y
            - df["Y_pixel"].to_numpy(float)
        ).max()

        print(
            f"BBox_y2 vs Y_pixel 最大差: "
            f"{err_y:.6f} px"
        )

    # ------------------------------------------------
    # 各基準点 f を変換
    # ------------------------------------------------

    for f in F_VALUES:

        # bbox下辺上の基準点
        px = (
            df["BBox_x1"]
            + f * (
                df["BBox_x2"]
                - df["BBox_x1"]
            )
        ).to_numpy(float)

        py = df["BBox_y2"].to_numpy(float)

        # pixel → Tennessee State Plane [ft]
        X_TN_ft, Y_TN_ft = apply_homography(
            H,
            px,
            py
        )

        # ft → meter
        X_meter = X_TN_ft * FT_TO_M
        Y_meter = Y_TN_ft * FT_TO_M

        # ------------------------------------------------
        # 保存
        # ------------------------------------------------

        out = df.copy()

        out["X_ref_pixel"] = px
        out["Y_ref_pixel"] = py

        out["X_TN_ft"] = X_TN_ft
        out["Y_TN_ft"] = Y_TN_ft

        out["X_meter"] = X_meter
        out["Y_meter"] = Y_meter

        out["ref_f"] = f

        path = f"{base}_f{f:.3f}.csv"

        out.to_csv(
            path,
            index=False
        )

        print(
            f"保存: {path} "
            f"(f={f:.3f}, "
            f"X_meter中央値={np.median(X_meter):.2f} m, "
            f"Y_meter中央値={np.median(Y_meter):.2f} m)"
        )

    print("\n完了")


if __name__ == "__main__":
    main()