# -*- coding: utf-8 -*-
"""
evaluate_refpoint.py — 基準点 f ごとのクラス間差を、f に依存しない車線割り当てで集計する

使い方:
    python compare_bbox2/evaluate_refpoint.py P33C01      （assign_lane_occupancy.py の後）

評価指標:
    符号     : + = 大型車が普通車より「カメラ側」にずれている（カメラ位置はホモグラフィから逆算）
    代表値   : 車両ごとの d の平均（f ごと）
    主指標   : 大型車の代表値の平均 − 同じ車線の普通車の代表値の平均（Welch 95%区間、Cohen's d）
    f*       : クラス間差が 0 になる f（f=0〜1 を線形補間）。f*区間 = 95%区間が 0 を含む f の範囲
               f による差の変化が小さい（|傾き| < MIN_SLOPE m）車線は「f依存なし」とする
    車両集合 : すべての f で同じ。普通車は車線中心から半車線以内、大型車は占有判定で車線が決まったもの
    参考     : ピックアップ（COCO truck のうち普通車サイズ）の普通車との差
    走行状態 : regime = all / free / congested（車両ごとの平均速度で分ける。assign_lane_occupancy.py の
               CONGEST_KMH）。大型車も普通車も同じ状態の車両どうしで比べる。free と congested で f* が
               同じなら、渋滞の影響は気にしなくてよい

出力（out_<CSV名>2/）:
    refpoint_summary.csv  車線 × f の集計表
    refpoint_fstar.csv    車線ごとの f* と、f=0 / 0.5 / 1 でのクラス間差
    refpoint_sweep.png    横軸 f・縦軸クラス間差（大型車のみ、車線ごと、95%区間）。ピックアップは CSV にのみ出力
"""

import sys
import numpy as np
import pandas as pd

from common import CAMERAS, DEFAULT_CAMERAS, F_VALUES, base_name, out_dir, out_path, col
import os

# ============== 設定 ==============
MIN_LARGE     = 3      # 大型車がこれ未満の車線はクラス間差を出さない
MIN_CAR       = 10
LANE_CHANGE_M = 2.0
LANE_HALF_M   = 1.85
MIN_SLOPE     = 0.25   # f=0→1 でクラス間差がこれ未満しか変わらなければ f依存なし [m]
REGIMES       = ["all", "free", "congested"]
MAKE_PLOT     = True
# ==================================


def welch(a, b):
    """b − a の差、95%区間の半幅、Cohen's d"""
    diff = b.mean() - a.mean()
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return diff, 1.96 * se, diff / sp if sp > 0 else np.nan


def zero_cross(fs, y):
    fg = np.linspace(0, 1, 201)
    yi = np.interp(fg, fs, y)
    z = np.where(np.diff(np.sign(yi)) != 0)[0]
    return fg[z[0]] if len(z) else np.nan


def main(cam):
    v = pd.read_csv(out_path(cam, "vehicles.csv"))
    lanes = pd.read_csv(out_path(cam, "lanes.csv"))
    motion = lanes.motion.iloc[0]
    ok = v.d_shift <= LANE_CHANGE_M
    car = v[ok & (v.group == "car") & (v.off.abs() <= LANE_HALF_M)]
    pickup = v[ok & (v.group == "pickup") & (v.off.abs() <= LANE_HALF_M)]
    large = v[ok & (v.group == "large") & v.lane_method.isin(["unique", "occupancy"])]

    rows, fstar = [], []
    for regime in REGIMES:
        sel = (lambda x: x) if regime == "all" else (lambda x, r=regime: x[x.regime == r])
        for _, ln in lanes.iterrows():
            c = sel(car)[sel(car).lane == ln.lane]
            for gname, grp in [("large", sel(large)), ("pickup", sel(pickup))]:
                b = grp[grp.lane == ln.lane]
                if len(b) < MIN_LARGE or len(c) < MIN_CAR:
                    continue
                res = []
                for f in F_VALUES:
                    diff, ci, cd = welch(c[col("dm", f)] * ln.toward_cam_sign, b[col("dm", f)] * ln.toward_cam_sign)
                    res.append(diff)
                    rows.append(dict(regime=regime, lane=ln.lane, group=gname, f=f, lat_from_cam_m=round(ln.lat_from_cam_m, 1),
                                     n_car=len(c), n=len(b), class_diff_m=round(diff, 3), ci95_m=round(ci, 3),
                                     cohens_d=round(cd, 2),
                                     median_diff_m=round((b[col("dm", f)].median() - c[col("dm", f)].median())
                                                         * ln.toward_cam_sign, 3),
                                     noise_sd_cm=round(pd.concat([c, b])[col("dsd", f)].median() * 100, 1)))
                res = np.array(res)
                ci_arr = np.array([r["ci95_m"] for r in rows[-len(F_VALUES):]])
                slope = res[-1] - res[0]
                row = dict(regime=regime, lane=ln.lane, group=gname, lat_from_cam_m=round(ln.lat_from_cam_m, 1), motion=motion,
                           n_car=len(c), n=len(b), diff_f0=round(res[0], 2), diff_f05=round(res[4], 2),
                           diff_f1=round(res[-1], 2), change_f0_to_1=round(slope, 2))
                if abs(slope) < MIN_SLOPE:
                    row.update(f_star="f依存なし", f_star_range="")
                else:
                    fg = np.linspace(0, 1, 201)
                    lo = np.interp(fg, F_VALUES, res - ci_arr)
                    hi = np.interp(fg, F_VALUES, res + ci_arr)
                    inside = fg[(lo <= 0) & (hi >= 0)]
                    fz = zero_cross(F_VALUES, res)
                    row.update(f_star=round(fz, 2) if np.isfinite(fz) else "0にならない",
                               f_star_range=f"{inside.min():.2f}-{inside.max():.2f}" if len(inside) else "")
                fstar.append(row)

    summary = pd.DataFrame(rows)
    fs = pd.DataFrame(fstar)
    summary.to_csv(os.path.join(out_dir(cam), "refpoint_summary.csv"), index=False)
    fs.to_csv(os.path.join(out_dir(cam), "refpoint_fstar.csv"), index=False)

    pd.set_option("display.width", 250)
    lg = v[v.group == "large"]
    print(f"\n=== [{cam}] 画角 {motion} / 大型車 {len(lg)}台中 車線判定できた {len(large)}台"
          f"（判定不能 {sum(lg.lane_method == 'undecided')}台）===")
    print(lanes[["lane", "center_d", "lat_from_cam_m"]].round(2).to_string(index=False))
    print(f"走行状態の内訳（評価に使う車両）: 普通車 {car.regime.value_counts().to_dict()}, "
          f"大型車 {large.regime.value_counts().to_dict()}")
    print(fs.to_string(index=False) if len(fs) else "  評価できる車線なし")

    # グラフは大型車 − 普通車のクラス間差だけ（ピックアップは CSV にのみ残す）
    plot = summary[(summary.regime == "all") & (summary.group == "large")] if len(summary) else summary
    if MAKE_PLOT and len(plot):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for lane, sl in plot.groupby("lane"):
            ax.errorbar(sl.f, sl.class_diff_m, yerr=sl.ci95_m, marker="o", capsize=3,
                        label=f"{lane} (n={sl.n.iloc[0]}, {sl.lat_from_cam_m.iloc[0]} m from camera)")
        ax.axhline(0, color="0.6", lw=1)
        ax.set_xlabel("f  (0 = bottom-left, 0.5 = bottom-center, 1 = bottom-right)")
        ax.set_ylabel("class difference  large - car  [m]  (+ = toward camera)")
        ax.set_title(f"{base_name(cam)}  ({motion})")
        ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir(cam), "refpoint_sweep.png"), dpi=130)
        plt.close(fig)
    return fs.assign(camera=cam)


if __name__ == "__main__":
    for cam in (sys.argv[1:] or DEFAULT_CAMERAS):
        main(cam)
