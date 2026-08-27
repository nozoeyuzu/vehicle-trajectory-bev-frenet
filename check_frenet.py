# -*- coding: utf-8 -*-
"""
check_frenet.py — Frenet変換の健全性チェック（3枚組の図を1枚のPNGに出力）

使い方:
    python check_frenet.py
出力:
    check_frenet.png
      (1) BEV上の全軌跡 + クリック点 + フィット中心線 → フィットが道なりか目視確認
      (2) d_meter のヒストグラム → 山の位置が車線中心。LANE_CENTERS_D を決める
      (3) 全車両の d_dev vs s_meter 波形の重ね描き → 膨らみが同じsで揃うか確認
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib import font_manager
import os

# frenet_transform.py / frenet.py のどちらの名前でも読めるように
try:
    import frenet_transform as ft
except ImportError:
    import frenet as ft

# 日本語フォント
for fp in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
           "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf"]:
    if os.path.exists(fp):
        font_manager.fontManager.addfont(fp)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=fp).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False

raw = np.load(ft.CENTERLINE_NPY)
used = raw
dropped = np.empty((0, 2))
print(f"Using all {len(used)} clicked points for fitting", flush=True)
if len(used) < 3:
    raise RuntimeError("クリック点が3点未満です。クリックし直してください")

cxy, s, tan = ft.fit_centerline(used)
print(f"Fitted centerline length: {s[-1]:.1f} m", flush=True)

df = pd.read_csv(ft.CSV_PATH)
xy = df[["X_meter", "Y_meter"]].to_numpy(float)
s_out, d_out = ft.cart_to_frenet(xy, cxy, s, tan)
df["s_meter"], df["d_meter"] = s_out, d_out

fig, ax = plt.subplots(figsize=(16, 7))

# --- (1) BEVオーバーレイ ---
for vid, g in df.groupby("Vehicle_ID"):
    g = g.sort_values("Frame")
    ax.plot(g.X_meter, g.Y_meter, color="0.6", lw=0.5, alpha=0.6, zorder=1)
ax.plot(cxy[:, 0], cxy[:, 1], "b-", lw=2, label="fitted centerline", zorder=3)
ax.scatter(used[:, 0], used[:, 1], c="red", s=40, zorder=4, label="clicked points")
if len(dropped):
    ax.scatter(dropped[:, 0], dropped[:, 1], c="red", s=40, marker="x",
               zorder=4, label="excluded points")
plot_x = np.concatenate([df.X_meter.to_numpy(), cxy[:, 0], used[:, 0]])
plot_y = np.concatenate([df.Y_meter.to_numpy(), cxy[:, 1], used[:, 1]])
x_margin = max((plot_x.max() - plot_x.min()) * 0.05, 1.0)
y_margin = max((plot_y.max() - plot_y.min()) * 0.05, 1.0)
ax.set_xlim(plot_x.min() - x_margin, plot_x.max() + x_margin)
ax.set_ylim(plot_y.min() - y_margin, plot_y.max() + y_margin)

# X軸目盛りを細かくして中心線の重なりを目視しやすくする
x_span = ax.get_xlim()[1] - ax.get_xlim()[0]
if x_span <= 8:
    x_major = 0.5
elif x_span <= 20:
    x_major = 1.0
elif x_span <= 50:
    x_major = 2.0
else:
    x_major = 5.0
ax.xaxis.set_major_locator(MultipleLocator(x_major))
ax.xaxis.set_minor_locator(MultipleLocator(x_major / 2.0))

ax.set_xlabel("X_meter"); ax.set_ylabel("Y_meter")
ax.set_title("Centerline fit on BEV trajectories (fine X ticks)")
ax.legend(fontsize=9)
ax.grid(which="major", color="0.85", lw=0.9)
ax.grid(which="minor", color="0.92", lw=0.6)
ax.set_aspect("auto")

# --- (2) d のヒストグラム（保守用にコメントアウトして残す） ---
# ax = axes[1]
# d_in = df.d_meter[df.d_meter.abs() < 8]
# ax.hist(d_in, bins=120, color="tab:blue", alpha=0.8)
# for lc in getattr(ft, "LANE_CENTERS_D", []):
#     ax.axvline(lc, color="red", ls="--", lw=1.2)
# ax.set_xlabel("d_meter [m] (lateral distance from centerline)")
# ax.set_ylabel("count")
# ax.set_title("(2) Histogram of d_meter")
# ax.grid(color="0.9")

# --- (3) d_dev vs s の全車波形（保守用にコメントアウトして残す） ---
# lanes = np.asarray(getattr(ft, "LANE_CENTERS_D", []) or [np.nan], float)
# if np.isfinite(lanes).all() and len(lanes):
#     nearest = lanes[np.argmin(np.abs(d_out[:, None] - lanes[None, :]), axis=1)]
#     df["d_dev"] = d_out - nearest
# else:
#     df["d_dev"] = df.d_meter - df.groupby("Vehicle_ID").d_meter.transform("median")
#
# ax = axes[2]
# for vid, g in df.groupby("Vehicle_ID"):
#     g = g.sort_values("Frame")
#     if len(g) < 25:
#         continue
#     ax.plot(g.s_meter, g.d_dev, lw=0.7, alpha=0.55)
# ax.axhline(0, color="k", lw=0.8)
# ax.set_xlabel("s_meter [m] (distance along centerline)")
# ax.set_ylabel("d_dev [m] (deviation from nearest lane center)")
# ax.set_title("(3) d_dev aligned by s_meter")
# ax.grid(color="0.9")

plt.tight_layout()
output_path = "check_frenet_bev_only.png"
plt.savefig(output_path, dpi=140)
print(f"Saved: {output_path}", flush=True)