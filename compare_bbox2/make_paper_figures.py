# -*- coding: utf-8 -*-
"""
make_paper_figures.py — 論文の図4・図5を作る（make_paper_tables.py の後に実行）

使い方（リポジトリ直下で）:
    python compare_bbox2/make_paper_figures.py

    図4: 代表3カメラの Δ(f)（近づく・遠ざかる・真横）。線が車線、帯が95%区間、点が f*、点線が f=0.5
         代表は、計算する前に予測を書いたポール P19 の C01・C06・C03（大型車が多く、区間が狭い）
    図5: 画像上の車線の傾き α と f*（全車線、エラーバーが95%区間）。色と形がポール
         Δ が 0 にならない車線は、ずれが小さい方の端に白抜きの点で示す。真横（f依存なし）は f* がないので載せない

印刷（白黒を含む）を想定し、色に加えて点の形でも区別する。図のデータは付録表A と同じ。
入力: out_<CSV名>2/refpoint_summary.csv、out_paper_tables/lanes_raw.csv
出力: out_paper_figures/（fig4_delta_f.pdf/.png、fig5_alpha_fstar.pdf/.png）
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

import analyze_three_poles as atp
from common import ROOT

OUT = os.path.join(ROOT, "out_paper_figures")
FIG4_CAMS = [("P19C01_15min", "(a) 近づく画角（P19C01）"), ("P19C06_15min", "(b) 遠ざかる画角（P19C06）"),
             ("P19C03_15min", "(c) 真横の画角（P19C03）")]
# 色は検証済みの分類用パレットの先頭から順に使う（車線・ポールに固定で割り当てる）
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
MARKERS = ["o", "s", "^", "D"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
FONTS = ["/mnt/c/Windows/Fonts/BIZ-UDGothicR.ttc", "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
         "/mnt/c/Windows/Fonts/meiryo.ttc"]


def setup():
    for fp in FONTS:
        if os.path.exists(fp):
            font_manager.fontManager.addfont(fp)
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=fp).get_name()
            break
    plt.rcParams.update({"axes.unicode_minus": False, "font.size": 9, "axes.edgecolor": INK2,
                         "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                         "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})


def fig4(d):
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7), sharey=True)
    for ax, (cam, title) in zip(axes, FIG4_CAMS):
        g = d[d.camera == cam]
        ax.axhline(0, color=INK2, lw=0.8)
        ax.axvline(0.5, color=INK2, lw=0.8, ls=":")
        for k, (lane, gl) in enumerate(sorted(g.groupby("lane"), key=lambda x: x[0])):
            y, ci = atp.curve(gl)
            gl = gl.sort_values("f")
            c, m = SERIES[k], MARKERS[k]
            ax.fill_between(atp.FG, y - ci, y + ci, color=c, alpha=0.12, lw=0)
            ax.plot(gl.f, gl.class_diff_m, color=c, lw=1.5, marker=m, ms=3.5,
                    label=f"{lane}（大型車 {gl.n.iloc[0]}台）")
            fz = atp.zero(y)
            if abs(y[-1] - y[0]) >= 0.25 and np.isfinite(fz):
                ax.plot(fz, 0, marker=m, ms=7, mfc="white", mec=c, mew=1.5, zorder=5)
        ax.set_title(title, fontsize=9, color=INK)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_xlim(-0.03, 1.03)
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.legend(fontsize=6.5, frameon=False, loc="upper right" if "近づく" in title else "upper left")
    axes[0].set_ylabel("Δ(f)：大型車 − 普通車 [m]\n（＋ = カメラ側）")
    fig.supxlabel("基準点の位置 f（0 = bbox 下辺の左端、1 = 右端）　点線: f = 0.5、白抜き: f*", fontsize=9, color=INK)
    fig.tight_layout()
    return fig


def fig5(L):
    obl = L[L.side != "f依存なし"].copy()
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    ax.axhline(0.5, color=INK2, lw=0.8, ls=":")
    ax.axvline(0, color=INK2, lw=0.8)
    for k, pole in enumerate(atp.POLES):
        g = obl[obl.pole == pole]
        c, m = SERIES[k], MARKERS[k]
        has = g[g.f_star.notna()]
        ax.errorbar(has.alpha, has.f_star, yerr=[has.f_star - has.f_lo, has.f_hi - has.f_star], fmt=m, ms=5,
                    color=c, mec="white", mew=0.6, ecolor=c, elinewidth=0.9, capsize=2, label=f"{pole}（{len(g)}車線）")
        no = g[g.f_star.isna()]     # 0 にならない車線: ずれが小さい方の端に白抜き
        ax.plot(no.alpha, np.where(no.side == "右", 1.0, 0.0), m, ms=5, mfc="white", mec=c, mew=1.2, ls="none")
    ax.text(38, 0.95, "右下がり → 右端寄り", ha="right", va="center", fontsize=7.5, color=INK2)
    ax.text(-38, 0.05, "右上がり → 左端寄り", ha="left", va="center", fontsize=7.5, color=INK2)
    ax.plot([], [], "o", ms=5, mfc="white", mec=INK2, mew=1.2, ls="none", label="白抜き: 0 にならない（端が最良）")
    ax.set_xlabel("画像上の車線の傾き α [°]（＋ = 右下がり、− = 右上がり）")
    ax.set_ylabel("f*（95%区間）")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(-70, 70)
    ax.grid(color=GRID, lw=0.6)
    ax.legend(fontsize=7, frameon=True, framealpha=0.95, edgecolor=GRID, loc="center", bbox_to_anchor=(0.5, 0.40))
    fig.tight_layout()
    return fig


def main():
    setup()
    os.makedirs(OUT, exist_ok=True)
    d = atp.load()
    L = pd.read_csv(os.path.join(ROOT, "out_paper_tables", "lanes_raw.csv"))
    for name, fig in [("fig4_delta_f", fig4(d)), ("fig5_alpha_fstar", fig5(L))]:
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=300)
        plt.close(fig)
    print(f"保存: {OUT}/fig4_delta_f.pdf/.png, fig5_alpha_fstar.pdf/.png")


if __name__ == "__main__":
    main()
