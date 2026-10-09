# -*- coding: utf-8 -*-
"""
analyze_fstar_angle.py — 画像上の車線の傾きと f* の関係を調べる（評価パイプラインの後に実行する分析）

使い方（リポジトリ直下で、run_all.py で各カメラを処理した後）:
    python compare_bbox2/analyze_fstar_angle.py                      # P33・P10 の15分版12カメラ
    python compare_bbox2/analyze_fstar_angle.py P33C01_15min P10C01_15min

車線の傾き α:
    その車線の普通車の、画像上の接地点（bbox 下辺中央）の移動方向の向き [deg]。
    画像の横軸から測り、−90〜+90°。＋ = 右下がり（左上↔右下に走る）、− = 右上がり（左下↔右上に走る）、
    0 = 水平（真横）。進む向き（近づく／遠ざかる）は区別しない。
    画像上の軌跡だけから測るので、ホモグラフィから逆算するカメラ位置（外れることがある）は使わない。

出力（out_fstar_angle/）:
    fstar_angle.csv  車線ごとの α・f*・f=0→1 でのクラス間差の変化
    fstar_angle.png  (a) α と f*、(b) α と |f=0→1 での変化|
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import ROOT, out_dir, out_path

# ============== 設定 ==============
DEFAULT = [f"{p}C0{c}_15min" for p in ("P33", "P10") for c in range(1, 7)]
MIN_LARGE = 5        # 大型車がこれ未満の車線は使わない
MIN_MOVE_PX = 50     # これより動かない普通車は向きの計算に使わない
# カメラ位置の推定が外れて車線名がずれたカメラの読み替え（README の既知の限界）
LANE_RENAME = {
    "P10C05_15min": {"L3": "L1", "L4": "L2", "L5": "L3", "L6": "L4", "L7": "L5"},
    "P10C06_15min": {"L4": "L1", "L2": "L2", "L1": "L3", "L3": "L4", "L5": "L5"},
}
# ==================================


def lane_angle(dx, dy):
    """向きの平均（180°違いを同じとみなす）[deg]。右下がり = ＋"""
    a = np.arctan2(dy, dx)
    m = np.angle(np.mean(np.exp(2j * a))) / 2
    return np.degrees(m) if abs(m) <= np.pi / 2 else np.degrees(m - np.sign(m) * np.pi)


def collect(cam):
    v = pd.read_csv(out_path(cam, "vehicles.csv"))
    fs = pd.read_csv(os.path.join(out_dir(cam), "refpoint_fstar.csv"))
    fs = fs[(fs.group == "large") & (fs.regime == "all") & (fs.n >= MIN_LARGE)]
    rows = []
    for r in fs.itertuples():
        car = v[(v.lane == r.lane) & (v.group == "car") & (np.hypot(v.dx_px, v.dy_px) > MIN_MOVE_PX)]
        lo, hi = (r.f_star_range.split("-") + [np.nan])[:2] if isinstance(r.f_star_range, str) else (np.nan, np.nan)
        rows.append(dict(camera=cam, pole=cam[:3], lane=LANE_RENAME.get(cam, {}).get(r.lane, r.lane),
                         motion=r.motion, n_large=r.n, n_car_angle=len(car),
                         angle_deg=round(lane_angle(car.dx_px.values, car.dy_px.values), 1),
                         f_star=pd.to_numeric(r.f_star, errors="coerce"), f_lo=float(lo), f_hi=float(hi),
                         f_star_label=r.f_star, abs_change_f0_to_1=abs(r.change_f0_to_1)))
    return rows


def main(cams):
    d = pd.DataFrame([row for cam in cams for row in collect(cam)])
    od = os.path.join(ROOT, "out_fstar_angle")
    os.makedirs(od, exist_ok=True)
    d.to_csv(os.path.join(od, "fstar_angle.csv"), index=False)
    pd.set_option("display.width", 200)
    print(d.drop(columns=["f_lo", "f_hi", "f_star"]).to_string(index=False))

    colors = {"approach": "tab:red", "recede": "tab:blue", "side": "tab:gray"}
    markers = {"P33": "o", "P10": "s"}
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    for (pole, motion), g in d.groupby(["pole", "motion"]):
        kw = dict(color=colors[motion], marker=markers[pole], linestyle="none", label=f"{pole} {motion}")
        ok = g[g.f_star.notna()]
        if len(ok):
            ax[0].errorbar(ok.angle_deg, ok.f_star, yerr=[ok.f_star - ok.f_lo, ok.f_hi - ok.f_star], capsize=3, **kw)
        ax[1].plot(g.angle_deg, g.abs_change_f0_to_1, **kw)
    ax[0].set_xlabel("lane angle in image [deg]  (+ = down-right, - = up-right, 0 = horizontal)")
    ax[0].set_ylabel("f*  (large - car = 0)")
    ax[0].set_ylim(-0.05, 1.05)
    ax[0].set_title(f"(a) f* vs lane angle  (n_large >= {MIN_LARGE}; side views have no f*)")
    ax[1].set_xlabel("lane angle in image [deg]")
    ax[1].set_ylabel("|class difference change from f=0 to f=1|  [m]")
    ax[1].set_title("(b) how much f matters")
    for a in ax:
        a.axvline(0, color="0.7", lw=0.8)
        a.grid(alpha=0.3)
    ax[0].axhline(0.5, color="0.7", lw=0.8)
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(od, "fstar_angle.png"), dpi=130)
    print(f"保存: {od}/fstar_angle.csv, fstar_angle.png")


if __name__ == "__main__":
    main(sys.argv[1:] or DEFAULT)
