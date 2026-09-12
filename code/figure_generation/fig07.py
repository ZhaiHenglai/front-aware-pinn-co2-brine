import os
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "dejavuserif", "font.size": 8.0,
    "axes.linewidth": 0.7,
})
# family colours (Fig. 4 convention)
FAM = {"M0": "#8a94a3", "M1": "#1f63d6", "M2": "#1f63d6", "M3": "#1f63d6",
       "M4": "#e07b00", "M5": "#e07b00", "M6": "#2a8c5a", "M7": "#c8102e"}
INK = "#5a6472"


from shared import ROOT,OUT,SRC,save,capture
raw=pd.read_csv(capture(ROOT/"06_evidence/H_tables_v2/by_seed.csv"))
raw=raw[raw.study=="M0_M7"]
assert len(raw)==32 and set(raw.seed)=={0,1,2,42}
cols=["S_rel_l2","S_RMSE_front_band","p_phys_RMSE_MPa","local_FV_CO2_RMSE"]
df=raw.groupby("role")[cols].mean().reindex([f"M{i}" for i in range(8)])
err=raw.groupby("role")[cols].std(ddof=1).reindex(df.index)
df.to_csv(SRC/"Figure7_mean.csv");err.to_csv(SRC/"Figure7_SD.csv")

PANELS = [
    ("S_rel_l2", "saturation rel. $L_2$"),
    ("S_RMSE_front_band", "front-band $S$ RMSE"),
    ("p_phys_RMSE_MPa", "$p$ RMSE (MPa)"),
    ("local_FV_CO2_RMSE", "local FV CO$_2$ imbalance"),
]
letters = "abcd"
fig, axes = plt.subplots(1, 4, figsize=(7.48, 2.0))
fig.subplots_adjust(left=0.06, right=0.99, top=0.86, bottom=0.21, wspace=0.34)

x = range(8)
for ax, (col, label), let in zip(axes, PANELS, letters):
    y = df[col].values
    if err is not None:
        ax.errorbar(x, y, yerr=err[col].values, color=INK, lw=0.9,
                    elinewidth=0.7, capsize=1.6, zorder=1)
    else:
        ax.plot(x, y, color=INK, lw=0.9, zorder=1)
    for i, m in enumerate(df.index):
        ax.scatter(i, y[i], s=26, color=FAM[m], zorder=3,
                   edgecolor="#1a1a1a" if m == "M7" else "none", linewidths=0.7)
    ax.set_xticks(list(x))
    ax.set_xticklabels(df.index, fontsize=6.6, rotation=0)
    ax.set_title(f"$\\bf{{{let}}}$  {label}", fontsize=7.6, pad=3, loc="left")
    ax.tick_params(labelsize=6.8, length=2.2)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.08)
    ax.grid(axis="y", lw=0.3, color="#d8dce2", zorder=0)

# family legend (compact, above panels)
import matplotlib.lines as mlines
handles = [mlines.Line2D([], [], marker="o", ls="", ms=4.5, color=c, label=l)
           for l, c in [("vanilla", "#8a94a3"), ("representation (M1–M3)", "#1f63d6"),
                        ("front supervision (M4–M5)", "#e07b00"),
                        ("adaptive sampling (M6)", "#2a8c5a"),
                        ("local FV regularization (M7)", "#c8102e")]]
fig.legend(handles=handles, ncol=5, fontsize=6.3, frameon=False,
           loc="lower center", bbox_to_anchor=(0.5, -0.035), handletextpad=0.15,
           columnspacing=0.9)

save(fig,'Figure7')

