# -*- coding: utf-8 -*-
"""
make_paper_tables.py — 論文の表と本文の数字を、評価の出力からまとめて作る（評価パイプラインの後に実行）

使い方（リポジトリ直下で、run_all.py・analyze_fstar_angle.py の後）:
    python compare_bbox2/make_paper_tables.py

内容（指標の定義は paper/metrics_f.md）:
    表1: データの統計（カメラごとの台数・速度・渋滞の割合）
    表2: カメラ単位の結果（α、Δ(0.5)、f による変化幅、f* と区間、4つの規則で残るずれ）
    表3: 規則ごとに残るずれ（ポールを1本抜いて計算）
    表4: 車両ごとのばらつき（画角ごと）
    付録表A: 車線単位の結果（全車線）。本文では図4（Δ(f) のグラフ）と図5（α と f* の散布図）で見せる
    本文: 向きの一致、画角ごとの f*、順位相関、真横のずれ、0 と区別できない車線の割合（補助）、占有判定の正解率
    scene_A は中心線の修正後に加える（今は「未確定」と書くだけ）

数字の書き方: 長さ [m] と f は小数2桁（末尾の 0 も残す）、差には符号を付ける、α は整数 [°]、割合は整数 [%]

入力: out_<CSV名>2/（refpoint_summary.csv・vehicles.csv・lanes.csv）、out_fstar_angle/fstar_angle.csv、
      paper/occupancy_check.csv（占有判定の目視確認。なければ省く）
出力: out_paper_tables/（tables.md と、各表の CSV）
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import analyze_three_poles as atp
from common import ROOT, F_VALUES, out_path

OUT = os.path.join(ROOT, "out_paper_tables")
OCC_CSV = os.path.join(ROOT, "paper", "occupancy_check.csv")
MOTION_JA = {"approach": "近づく", "recede": "遠ざかる", "side": "真横"}
LANE_CHANGE_M, LANE_HALF_M = 2.0, 1.85     # evaluate_refpoint.py と同じ（評価に使う車両の条件）
RULES = [("R0 中央 (0.5)", "R0: f = 0.5（下辺中央）"), ("R1 画角ごと", "R1: 画角ごとの f"),
         ("R2 傾きから", "R2: 傾き α から決める f"), ("参考 自カメラの f*", "参考: 自カメラの f*")]
NOTE_TRACTOR = {"P33C05_15min"}            # トラクターの別検出の影響が大きいカメラ（README の既知の限界）


# ---------- 数字の書き方 ----------
def fm(x, sign=False):
    """小数2桁。sign=True なら符号を付ける（マイナスは −）"""
    if not np.isfinite(x):
        return "—"
    s = f"{x:+.2f}" if sign else f"{x:.2f}"
    return s.replace("-", "−")


def deg(x):
    """角度は整数。0 は符号なし、ほかは符号を付ける"""
    k = int(round(x))
    return "0" if k == 0 else f"{k:+d}".replace("-", "−")


def fstar_text(f, lo, hi):
    return f"{fm(f)}（{fm(lo)}–{fm(hi)}）"


def wilson(k, n, z=1.96):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def md(df):
    """DataFrame を Markdown の表にする"""
    head = "| " + " | ".join(df.columns) + " |\n|" + "---|" * len(df.columns) + "\n"
    return head + "\n".join("| " + " | ".join(str(x) for x in r) + " |" for r in df.itertuples(index=False))


# ---------- 集計 ----------
def lane_table(d):
    """車線ごとの Δ(0.5)・変化幅・f*・区間と、f* の左右"""
    rows = []
    for (cam, lane), g in d.groupby(["camera", "lane"], sort=False):
        y, ci = atp.curve(g)
        change = y[-1] - y[0]
        dep = abs(change) >= 0.25
        fz = atp.zero(y) if dep else np.nan
        inside = atp.FG[np.abs(y) <= ci]
        if not dep:
            side = "f依存なし"
        elif np.isfinite(fz):
            side = "右" if fz > 0.5 else "左"
        else:
            side = "右" if abs(y[-1]) < abs(y[0]) else "左"
        rows.append(dict(camera=cam, pole=cam[:3], lane=lane, motion=g.motion.iloc[0], n_large=int(g.n.iloc[0]),
                         n_car=int(g.n_car.iloc[0]), alpha=g.angle_deg.iloc[0], d05=y[100], ci05=ci[100],
                         change=change, f_star=fz, f_lo=inside.min() if dep and len(inside) else np.nan,
                         f_hi=inside.max() if dep and len(inside) else np.nan, side=side))
    return pd.DataFrame(rows)


def dispersion(cam_df, L):
    """車線ごとの、大型車と普通車の d の標準偏差（f=0.5 と、そのカメラの f*。真横は f=0.5）"""
    cols = [f"dm_{x:.3f}" for x in F_VALUES]
    at = lambda df, ff: np.array([np.interp(ff, F_VALUES, r) for r in df[cols].to_numpy()])
    rows = []
    for cam in atp.CAMS:
        c = cam_df[cam_df.camera == cam].iloc[0]
        f = c.f_star if np.isfinite(c.f_star) else 0.5
        v = pd.read_csv(out_path(cam, "vehicles.csv"))
        ok = v.d_shift <= LANE_CHANGE_M
        car = v[ok & (v.group == "car") & (v.off.abs() <= LANE_HALF_M)]
        lg = v[ok & (v.group == "large") & v.lane_method.isin(["unique", "occupancy"])]
        back = {new: old for old, new in atp.LANE_RENAME.get(cam, {}).items()}   # 読み替え後の名前 → 出力の名前
        for lane in L[L.camera == cam].lane:
            b, cc = lg[lg.lane == back.get(lane, lane)], car[car.lane == back.get(lane, lane)]
            rows.append(dict(motion=c.motion, n=len(b), large05=np.std(at(b, 0.5), ddof=1),
                             large_f=np.std(at(b, f), ddof=1), car05=np.std(at(cc, 0.5), ddof=1),
                             car_f=np.std(at(cc, f), ddof=1)))
    return pd.DataFrame(rows)


# ---------- 表 ----------
def table1():
    rows = []
    for cam in atp.CAMS:
        v = pd.read_csv(out_path(cam, "vehicles.csv"))
        lanes = pd.read_csv(out_path(cam, "lanes.csv"))
        ok = v.d_shift <= LANE_CHANGE_M
        lg = v[v.group == "large"]
        n_lanes = len(atp.LANE_RENAME[cam]) if cam in atp.LANE_RENAME else len(lanes)   # 偽の車線を除く
        rows.append({"カメラ": cam[:6] + ("*" if cam in atp.LANE_RENAME else ""),
                     "画角": MOTION_JA[lanes.motion.iloc[0]], "車線数": n_lanes,
                     "普通車 [台]": int((ok & (v.group == "car") & (v.off.abs() <= LANE_HALF_M)).sum()),
                     "大型車 [台]": len(lg),
                     "うち車線を判定できた [台]": int(lg.lane_method.isin(["unique", "occupancy"]).sum()),
                     "速度の中央値 [km/h]": round(v[v.group == "car"].speed_kmh.median()),
                     "渋滞の割合": f"{(v.regime == 'congested').mean():.0%}"})
    return pd.DataFrame(rows)


def table2(cam_df, rule_rows):
    res = rule_rows.groupby(["camera", "rule"]).apply(
        lambda g: np.average(g.abs_diff, weights=g.n), include_groups=False).unstack()
    rows = []
    for c in cam_df.itertuples():
        side = c.motion == "side"
        r = {k: (fm(res.loc[c.camera, k]) if c.camera in res.index else "—") for k, _ in RULES}
        rows.append({"カメラ": c.camera[:6] + ("†" if c.camera in NOTE_TRACTOR else ""),
                     "画角": MOTION_JA[c.motion], "α [°]": deg(c.angle_deg),
                     "大型車 [台]": c.n_large, "Δ(0.5) [m]": fm(c.diff_f05, True),
                     "変化幅 [m]": fm(c.change_f0_to_1, True),
                     "f*（95%区間）": "f依存なし" if side else fstar_text(c.f_star, c.f_lo, c.f_hi),
                     **{f"残るずれ {k.split()[0] if not k.startswith('参考') else '自身のf*'} [m]": r[k] for k, _ in RULES}})
    return pd.DataFrame(rows)


def table3(summ, by_pole):
    return pd.DataFrame([{"規則": name, "平均 [m]": fm(summ.loc[k, "mean_abs_diff_m"]),
                          "中央値 [m]": fm(summ.loc[k, "median_abs_diff_m"]),
                          **{f"{p} を試験用 [m]": fm(by_pole.loc[p, k]) for p in atp.POLES}} for k, name in RULES])


def table4(D):
    rows = []
    for m in ("approach", "recede", "side"):
        g = D[D.motion == m]
        w = g.n
        rows.append({"画角": MOTION_JA[m], "車線数": len(g),
                     "大型車 f=0.5 [m]": fm(np.average(g.large05, weights=w)),
                     "大型車 f* [m]": fm(np.average(g.large_f, weights=w)) if m != "side" else "—",
                     "普通車 f=0.5 [m]": fm(np.average(g.car05, weights=w)),
                     "普通車 f* [m]": fm(np.average(g.car_f, weights=w)) if m != "side" else "—"})
    return pd.DataFrame(rows)


def table_a(L):
    rows = []
    for r in L.itertuples():
        if r.side == "f依存なし":
            fs = "f依存なし"
        elif np.isfinite(r.f_star):
            fs = fstar_text(r.f_star, r.f_lo, r.f_hi)
        else:
            fs = f"0にならない（{r.side}端が最良、区間 {fm(r.f_lo)}–{fm(r.f_hi)}）"
        rows.append({"カメラ": r.camera[:6], "車線": r.lane, "画角": MOTION_JA[r.motion],
                     "α [°]": deg(r.alpha),
                     "大型車 [台]": r.n_large, "普通車 [台]": r.n_car,
                     "Δ(0.5) [m]（95%区間）": f"{fm(r.d05, True)} ± {fm(r.ci05)}",
                     "変化幅 [m]": fm(r.change, True), "f*（95%区間）": fs})
    return pd.DataFrame(rows)


def main():
    os.makedirs(OUT, exist_ok=True)
    d = atp.load()
    cam_df = atp.camera_level(d)
    rule_rows, summ, by_pole = atp.rule_eval(d, cam_df)
    L = lane_table(d)
    D = dispersion(cam_df, L)
    tabs = {"table1": table1(), "table2": table2(cam_df, rule_rows), "table3": table3(summ, by_pole),
            "table4": table4(D), "tableA_lanes": table_a(L)}
    for name, t in tabs.items():
        t.to_csv(os.path.join(OUT, f"{name}.csv"), index=False)
    L.to_csv(os.path.join(OUT, "lanes_raw.csv"), index=False)   # 丸める前の値（図を作るとき用）

    # ---- 本文の数字 ----
    text = []
    obl = L[L.side != "f依存なし"]
    match = np.where(obl.alpha > 0, "右", "左") == obl.side
    mis = obl[~match]
    clear_opp = sum(not (r.f_lo <= 0.5 <= r.f_hi) for r in mis.itertuples())
    text.append(f"- 結論①: α の向きと f* の左右が一致した車線 **{match.sum()}/{len(obl)}**。一致しなかった車線のうち、"
                f"f* の区間が 0.5 をまたがない（はっきり逆）もの **{clear_opp}** 車線"
                + ("（" + "、".join(f"{r.camera[:6]} {r.lane}: f* = {fstar_text(r.f_star, r.f_lo, r.f_hi)}"
                                   for r in mis.itertuples()) + "）" if len(mis) else ""))
    r0, r2 = summ.loc["R0 中央 (0.5)", "mean_abs_diff_m"], summ.loc["R2 傾きから", "mean_abs_diff_m"]
    text.append(f"- 結論②: 残るずれ（ポールを1本抜く、大型車の台数で重み付け、斜めの画角の {int(summ.loc['R0 中央 (0.5)', 'lanes'])} 車線）"
                f"f = 0.5 で **{fm(r0)} m** → 傾きの規則で **{fm(r2)} m**")
    text.append("- 補助: 0 と区別できない車線の割合 " + "、".join(
        f"{name.split('（')[0]} {summ.loc[k, 'share_zero_in_ci']:.0%}" for k, name in RULES))
    sd = L[L.motion == "side"]
    w = 1 / (sd.ci05 / 1.96) ** 2
    text.append(f"- 結論③: 真横のずれ（Δ(0.5)、1/分散で重み付け）**{np.sum(w * sd.d05) / w.sum():+.3f} ± "
                f"{1.96 / np.sqrt(w.sum()):.3f} m**（全車線をまとめた値で区間が狭いので小数3桁）（{sd.camera.nunique()}カメラ {len(sd)}車線）。"
                f"f による変化幅の最大 {fm(sd.change.abs().max())} m、f依存なし {(sd.side == 'f依存なし').sum()}/{len(sd)}")
    for m in ("approach", "recede"):
        x = cam_df[(cam_df.motion == m) & cam_df.f_star.notna()].f_star
        text.append(f"- 結論④: {MOTION_JA[m]}画角のカメラ単位 f* = **{fm(x.mean())} ± {fm(x.std())}**"
                    f"（{len(x)}台、{fm(x.min())}–{fm(x.max())}）")
    cc = cam_df[cam_df.f_star.notna()].copy()
    cc["edge"] = np.where(cc.motion == "approach", 1 - cc.f_star, cc.f_star)
    rho, p = spearmanr(cc.angle_deg.abs(), cc.edge)
    text.append(f"- 結論④: |α| と端からの距離の順位相関（カメラ単位）ρ = **{fm(rho)}**（p = {p:.3f}、{len(cc)}台）")
    if os.path.exists(OCC_CSV):
        o = pd.read_csv(OCC_CSV)
        ok = o.label == o.alg_letter
        lo, hi = wilson(ok.sum(), len(o))
        hc = o.confidence == "high"
        text.append(f"- 結論⑤: 占有判定の正解率（目視、{len(o)}台）**{ok.sum()}/{len(o)} = {ok.mean():.1%}**"
                    f"（95%区間 {lo:.0%}–{hi:.0%}）。目視の確信度が高い車両だけ {ok[hc].sum()}/{hc.sum()}")

    t1 = tabs["table1"]
    n_large, n_ok = t1["大型車 [台]"].sum(), t1["うち車線を判定できた [台]"].sum()
    with open(os.path.join(OUT, "tables.md"), "w", encoding="utf-8") as fp:
        fp.write("# 論文の表（make_paper_tables.py で自動生成。手で編集しない）\n\n")
        fp.write("I-24 の3ポール18カメラ。大型車が5台以上の車線。scene_A は中心線の修正後に加える（**未確定**）。\n\n")
        fp.write("## 表1: データの統計\n\n" + md(t1) + "\n\n")
        fp.write(f"普通車は評価に使える条件の台数（車線変更なし、車線中心から {LANE_HALF_M} m 以内）。速度は普通車の中央値。"
                 f"渋滞は平均速度 60 km/h 未満の車両の割合。大型車のうち車線を判定できたのは全体で {n_ok}/{n_large} 台"
                 f"（{n_ok / n_large:.0%}）。\n"
                 "\\* 路肩の外に、誤検出が集まった偽の車線が検出されたカメラ。偽の車線を除いた数を示す。\n\n")
        fp.write("## 表2: カメラ単位の結果\n\n" + md(tabs["table2"]) + "\n\n")
        fp.write("Δ(0.5)・変化幅は、車線を 1/分散 で重み付けした平均（＋＝カメラ側）。残るずれは、そのカメラのポールを抜いて"
                 "決めた規則を当てはめたときの |Δ|（車線を大型車の台数で重み付け）。真横は規則の対象外。\n"
                 "† トラクターとトレーラーが別々に検出される影響で、規則で決めた f でもずれが大きく残る。\n\n")
        fp.write("## 表3: 規則ごとに残るずれ（ポールを1本抜いて計算、斜めの画角の車線）\n\n" + md(tabs["table3"]) + "\n\n")
        fp.write("## 表4: 車両ごとのばらつき（d の標準偏差、車線を大型車の台数で重み付け）\n\n" + md(tabs["table4"]) + "\n\n")
        fp.write("## 本文の数字\n\n" + "\n".join(text) + "\n\n")
        fp.write("## 付録表A: 車線単位の結果（全車線）\n\n" + md(tabs["tableA_lanes"]) + "\n\n")
        fp.write("本文では図4（代表3カメラの Δ(f)）と図5（α と f* の散布図）で示す。"
                 "P10C05・P10C06 の車線名は、README の読み替え後の名前。\n")
    print("\n".join(text))
    print(f"\n保存: {OUT}/tables.md と各表の CSV")


if __name__ == "__main__":
    main()
