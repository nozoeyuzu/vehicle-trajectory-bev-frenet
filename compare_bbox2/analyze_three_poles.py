# -*- coding: utf-8 -*-
"""
analyze_three_poles.py — 3ポール（P33・P10・P19）18カメラの結果をまとめて分析する（評価パイプラインの後に実行）

使い方（リポジトリ直下で、run_all.py と analyze_fstar_angle.py の後）:
    python compare_bbox2/analyze_three_poles.py

内容:
    1. カメラ単位の f*: 車線ごとのクラス間差を、95%区間から求めた重み（1/分散）で平均し、0 になる f を求める
    2. 同じ位置のカメラ（C01〜C06）のポール間の一致
    3. 規則の検証（1本のポールを抜き、残り2本で決めた規則を当てはめる）
         R0: f = 0.5（bbox 下辺中央）
         R1: 画角ごとの f（近づく／遠ざかるの、学習用カメラの f* の平均）
         R2: 車線の傾き α から決める f（端からの距離 = a·|α| + b を学習用カメラで当てはめる）
         参考: そのカメラ自身の f*（上限の目安）
       評価: 車線ごとに、その f でのクラス間差の絶対値（大型車の台数で重み付けした平均）と、
             差の 95%区間が 0 を含む車線の割合
    4. 真横のずれのまとめ

入力: out_<CSV名>2/refpoint_summary.csv、out_fstar_angle/fstar_angle.csv
出力: out_three_poles/（camera_fstar.csv、rule_eval.csv、three_poles.png）
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import ROOT, out_dir, F_VALUES

POLES = ["P33", "P10", "P19"]
CAMS = [f"{p}C0{c}_15min" for p in POLES for c in range(1, 7)]
MIN_LARGE = 5
# カメラ位置の推定が外れたカメラの読み替え（README の既知の限界）。符号は「＋＝カメラ側」に直す
LANE_RENAME = {
    "P10C05_15min": {"L3": "L1", "L4": "L2", "L5": "L3", "L6": "L4", "L7": "L5"},
    "P10C06_15min": {"L4": "L1", "L2": "L2", "L1": "L3", "L3": "L4", "L5": "L5"},
}
SIGN_FLIP = {"P10C06_15min": {"L1", "L2", "L4"}}   # 出力の車線名で、符号が逆になっている車線
FG = np.linspace(0, 1, 201)


def load():
    rows = []
    for cam in CAMS:
        s = pd.read_csv(os.path.join(out_dir(cam), "refpoint_summary.csv"))
        s = s[(s.regime == "all") & (s.group == "large") & (s.n >= MIN_LARGE)].copy()
        flip = s.lane.isin(SIGN_FLIP.get(cam, set()))
        s.loc[flip, "class_diff_m"] *= -1
        s["lane"] = s.lane.map(LANE_RENAME.get(cam, {})).fillna(s.lane)
        s["camera"], s["pole"], s["pos"] = cam, cam[:3], cam[3:6]
        rows.append(s)
    d = pd.concat(rows, ignore_index=True)
    ang = pd.read_csv(os.path.join(ROOT, "out_fstar_angle", "fstar_angle.csv"))[["camera", "lane", "motion", "angle_deg"]]
    return d.merge(ang, on=["camera", "lane"], how="left")


def curve(g):
    """車線 × f の表から、f の格子上の差の曲線（f ごとに並べた配列）"""
    g = g.sort_values("f")
    return np.interp(FG, g.f, g.class_diff_m), np.interp(FG, g.f, g.ci95_m)


def zero(y):
    z = np.where(np.diff(np.sign(y)) != 0)[0]
    return FG[z[0]] if len(z) else np.nan


def camera_level(d):
    out = []
    for cam, g in d.groupby("camera", sort=False):
        ys, ses, ns, angs = [], [], [], []
        for lane, gl in g.groupby("lane"):
            y, ci = curve(gl)
            ys.append(y); ses.append(ci / 1.96); ns.append(gl.n.iloc[0]); angs.append(gl.angle_deg.iloc[0])
        w = 1 / np.array(ses) ** 2
        y = (w * np.array(ys)).sum(0) / w.sum(0)
        se = 1 / np.sqrt(w.sum(0))
        change = y[-1] - y[0]
        inside = FG[np.abs(y) <= 1.96 * se]
        motion = g.motion.iloc[0]
        fs = zero(y) if abs(change) >= 0.25 else np.nan
        out.append(dict(camera=cam, pole=cam[:3], pos=cam[3:6], motion=motion, n_lanes=len(ys), n_large=int(sum(ns)),
                        angle_deg=float(np.average(angs, weights=ns)), f_star=fs,
                        f_lo=inside.min() if len(inside) and np.isfinite(fs) else np.nan,
                        f_hi=inside.max() if len(inside) and np.isfinite(fs) else np.nan,
                        diff_f05=float(np.interp(0.5, FG, y)), change_f0_to_1=float(change)))
    return pd.DataFrame(out)


def edge(f, motion):
    return 1 - f if motion == "approach" else f


def from_edge(e, motion):
    e = float(np.clip(e, 0, 0.5))
    return 1 - e if motion == "approach" else e


def rule_eval(d, cam_df):
    """1本のポールを抜き、残りで決めた規則をそのポールの各車線に当てはめる"""
    rows = []
    oblique = cam_df[cam_df.motion != "side"]
    for test in POLES:
        tr = oblique[(oblique.pole != test) & oblique.f_star.notna()]
        view_f = tr.groupby("motion").f_star.mean().to_dict()
        e = np.array([edge(r.f_star, r.motion) for r in tr.itertuples()])
        a, b = np.polyfit(tr.angle_deg.abs(), e, 1)
        for (cam, lane), gl in d[(d.pole == test) & (d.motion != "side")].groupby(["camera", "lane"]):
            motion = gl.motion.iloc[0]
            c = cam_df[cam_df.camera == cam].iloc[0]
            y, ci = curve(gl)
            rules = {"R0 中央 (0.5)": 0.5,
                     "R1 画角ごと": view_f[motion],
                     "R2 傾きから": from_edge(a * abs(c.angle_deg) + b, motion),
                     "参考 自カメラの f*": c.f_star if np.isfinite(c.f_star) else 0.5}
            for name, f in rules.items():
                k = int(round(f * 200))
                rows.append(dict(test_pole=test, camera=cam, lane=lane, motion=motion, n=gl.n.iloc[0], rule=name, f=f,
                                 abs_diff=abs(y[k]), zero_in_ci=abs(y[k]) <= ci[k]))
    r = pd.DataFrame(rows)
    summ = r.groupby(["rule"]).apply(lambda g: pd.Series(dict(
        lanes=len(g), mean_abs_diff_m=np.average(g.abs_diff, weights=g.n),
        median_abs_diff_m=g.abs_diff.median(), share_zero_in_ci=g.zero_in_ci.mean()))).round(3)
    by_pole = r.groupby(["test_pole", "rule"]).apply(lambda g: round(np.average(g.abs_diff, weights=g.n), 3)).unstack()
    return r, summ, by_pole


def main():
    d = load()
    cam_df = camera_level(d)
    od = os.path.join(ROOT, "out_three_poles")
    os.makedirs(od, exist_ok=True)
    cam_df.to_csv(os.path.join(od, "camera_fstar.csv"), index=False)
    pd.set_option("display.width", 220)

    print("=== 1. カメラ単位の f*（車線を 1/分散 で重み付け） ===")
    print(cam_df.round(3).to_string(index=False))

    print("\n=== 2. 同じ位置のカメラのポール間の比較（f*、真横は f=0.5 の差 [m]） ===")
    t = cam_df.assign(val=np.where(cam_df.motion == "side", cam_df.diff_f05, cam_df.f_star))
    print(t.pivot(index="pos", columns="pole", values="val")[POLES].round(2).to_string())
    for m in ("approach", "recede"):
        x = cam_df[(cam_df.motion == m) & cam_df.f_star.notna()]
        print(f"  {m}: カメラ {len(x)}台, f* 平均 {x.f_star.mean():.3f}, 標準偏差 {x.f_star.std():.3f}, 範囲 {x.f_star.min():.2f}〜{x.f_star.max():.2f}")

    print("\n=== 3. 規則の検証（1本のポールを抜いて当てはめ、近づく・遠ざかる画角の車線） ===")
    r, summ, by_pole = rule_eval(d, cam_df)
    r.to_csv(os.path.join(od, "rule_eval.csv"), index=False)
    print(summ.to_string())
    print("\n  抜いたポールごとの、クラス間差の絶対値の平均 [m]:")
    print(by_pole.to_string())

    print("\n=== 4. 真横 ===")
    s = d[d.motion == "side"]
    lane_side = s.groupby(["camera", "lane"]).apply(lambda g: pd.Series(dict(
        n=g.n.iloc[0], diff=np.interp(0.5, g.f, g.class_diff_m), ci=np.interp(0.5, g.f, g.ci95_m),
        change=g.sort_values("f").class_diff_m.iloc[-1] - g.sort_values("f").class_diff_m.iloc[0]))).reset_index()
    w = 1 / (lane_side.ci / 1.96) ** 2
    m = np.average(lane_side["diff"], weights=w); se = 1 / np.sqrt(w.sum())
    print(lane_side.round(3).to_string(index=False))
    print(f"  {len(lane_side)}車線: f=0.5 の差の重み付き平均 {m:+.3f} m（95%区間 ±{1.96 * se:.3f}）、"
          f"f=0→1 の変化の絶対値の最大 {lane_side.change.abs().max():.2f} m")

    # 図: (a) カメラ単位の f* と傾き、(b) 規則ごとの残る差
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
    mk = {"P33": "o", "P10": "s", "P19": "^"}
    col = {"approach": "tab:red", "recede": "tab:blue"}
    for r_ in cam_df[cam_df.f_star.notna()].itertuples():
        ax[0].errorbar(r_.angle_deg, r_.f_star, yerr=[[r_.f_star - r_.f_lo], [r_.f_hi - r_.f_star]],
                       marker=mk[r_.pole], color=col[r_.motion], capsize=3, linestyle="none")
        ax[0].annotate(r_.camera[:6], (r_.angle_deg, r_.f_star), fontsize=7, xytext=(3, 3), textcoords="offset points")
    for p, m_ in mk.items():
        ax[0].plot([], [], m_, color="0.3", linestyle="none", label=p)
    ax[0].axhline(0.5, color="0.7", lw=0.8)
    ax[0].set_xlabel("camera lane angle in image [deg]  (+ = down-right, - = up-right)")
    ax[0].set_ylabel("camera-level f*")
    ax[0].set_ylim(-0.05, 1.05)
    ax[0].set_title("(a) camera-level f* (12 oblique cameras, 3 poles)")
    ax[0].legend(fontsize=8)
    order = ["R0 中央 (0.5)", "R1 画角ごと", "R2 傾きから", "参考 自カメラの f*"]
    labels = ["f=0.5", "view rule", "angle rule", "own f* (ref)"]
    vals = [summ.loc[o, "mean_abs_diff_m"] for o in order]
    ax[1].bar(labels, vals, color=["0.6", "tab:green", "tab:purple", "0.85"])
    for i, v in enumerate(vals):
        ax[1].text(i, v, f"{v:.2f}", ha="center", va="bottom")
    ax[1].set_ylabel("|large - car| at chosen f  [m]  (weighted by n)")
    ax[1].set_title("(b) leave-one-pole-out: remaining class difference")
    fig.tight_layout()
    fig.savefig(os.path.join(od, "three_poles.png"), dpi=130)
    print(f"\n保存: {od}/camera_fstar.csv, rule_eval.csv, three_poles.png")


if __name__ == "__main__":
    main()
