# -*- coding: utf-8 -*-
"""
run_all.py — 4K換算 → Frenet変換 → 車線判定 → 評価 をカメラごとに通しで実行する

使い方（リポジトリ直下で）:
    python compare_bbox2/run_all.py               # common.py の DEFAULT_CAMERAS（5分版の6台）
    python compare_bbox2/run_all.py P33C06_15min  # 15分版
    python compare_bbox2/run_all.py P33C01 P33C05 # 指定したカメラだけ

出力:
    out_<CSV名>2/           カメラごとの結果（各スクリプトの説明を参照）
    out_P33<_5min など>2_summary/  実行したカメラの f* 一覧（fstar_all.csv）
"""

import os
import sys
import pandas as pd

import make_refpoint_csv_I24
import frenet_sweep
import assign_lane_occupancy
import evaluate_refpoint
from common import CAMERAS, DEFAULT_CAMERAS, ROOT, base_name


def main(cams):
    results = []
    for cam in cams:
        print(f"\n################ {cam} ################")
        make_refpoint_csv_I24.main(cam)
        frenet_sweep.main(cam)
        assign_lane_occupancy.main(cam)
        results.append(evaluate_refpoint.main(cam))
    allfs = pd.concat(results, ignore_index=True)
    # 一覧の出力先は CSV 名の共通部分から決める（例: P33C0x_5min → out_P33_5min2_summary）
    suffixes = {base_name(c)[len("P33C0x"):] for c in cams}
    d = os.path.join(ROOT, f"out_P33{'_'.join(sorted(suffixes))}2_summary")
    os.makedirs(d, exist_ok=True)
    allfs.to_csv(os.path.join(d, "fstar_all.csv"), index=False)
    pd.set_option("display.width", 250)
    print("\n================ 全カメラの f*（大型車） ================")
    print(allfs[allfs.group == "large"].drop(columns="group").to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1:] or DEFAULT_CAMERAS)
