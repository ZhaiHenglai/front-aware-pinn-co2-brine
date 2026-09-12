import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "dejavuserif", "font.size": 8.0,
    "axes.linewidth": 0.7,
})
# family colours (Fig. 4 / Fig. 8 convention)
FAM = {"M0": "#8a94a3", "M1": "#1f63d6", "M2": "#1f63d6", "M3": "#1f63d6",
       "M4": "#e07b00", "M5": "#e07b00", "M6": "#2a8c5a", "M7": "#c8102e"}
INK = "#5a6472"
C_SUP = "#e07b00"   # supervision (M3->M4)
C_FV = "#c8102e"    # conservation (M6->M7)


from shared import ROOT,OUT,SRC,FW,save,capture
MRAY=FW/"multiray_front_M0_M7_BASE"

MODELS = [f"M{i}" for i in range(8)]
by_ray = pd.read_csv(capture(MRAY / "multiray_front_by_ray.csv"))
col = "front_sharpness_ratio_mean"

# --- panel (a): per-model mean +- std over seeds (each seed = mean over its 9 rays) ---
per_seed = (by_ray.groupby(["exp_name", "seed"])[col].mean().reset_index())
agg = (per_seed.groupby("exp_name")[col].agg(["mean", "std"])
       .reindex(MODELS))
sharp_mean = agg["mean"].values
sharp_std = agg["std"].values

# --- panel (b): per-ray mean over seeds, then deltas for the two transitions ---
ray_tab = (by_ray.groupby(["exp_name", "ray_angle_deg"])[col].mean()
           .unstack("exp_name"))            # index = ray angle, cols = models
angles = ray_tab.index.values
d_sup = (ray_tab["M4"] - ray_tab["M3"]).values   # supervision sharpening
d_fv = (ray_tab["M7"] - ray_tab["M6"]).values    # FV blunting

# --- source data ---
src = agg.rename(columns={"mean": "sharpness_ratio_mean", "std": "sharpness_ratio_std"})
src.to_csv(SRC / "Figure10_source.csv")
pd.DataFrame({"ray_angle_deg": angles,
              "delta_sharpness_M3_to_M4": d_sup,
              "delta_sharpness_M6_to_M7": d_fv}).to_csv(
    SRC / "Figure10_source_byray.csv", index=False)

# --- figure ---
fig, (axA, axB) = plt.subplots(1, 2, figsize=(5.6, 2.35))
fig.subplots_adjust(left=0.085, right=0.985, top=0.86, bottom=0.18, wspace=0.30)

# panel (a)
x = range(8)
axA.errorbar(x, sharp_mean, yerr=sharp_std, color=INK, lw=0.9,
             elinewidth=0.7, capsize=1.6, zorder=1)
for i, m in enumerate(MODELS):
    axA.scatter(i, sharp_mean[i], s=28, color=FAM[m], zorder=3,
                edgecolor="#1a1a1a" if m == "M7" else "none", linewidths=0.7)
axA.annotate("front supervision", xy=(4, sharp_mean[4]), xytext=(2.1, 0.455),
             fontsize=6.3, color=C_SUP, ha="left",
             arrowprops=dict(arrowstyle="->", color=C_SUP, lw=0.8))
axA.annotate("local FV\nregularization", xy=(7, sharp_mean[7]), xytext=(6.2, 0.205),
             fontsize=6.3, color=C_FV, ha="center",
             arrowprops=dict(arrowstyle="->", color=C_FV, lw=0.8))
axA.set_xticks(list(x)); axA.set_xticklabels(MODELS, fontsize=6.8)
axA.set_ylabel("front sharpness ratio\n(pred / reference peak gradient)", fontsize=7.2)
axA.set_title(r"$\bf{c}$  front sharpness along rays", fontsize=7.8, pad=3, loc="left")
axA.tick_params(labelsize=6.8, length=2.2)
axA.set_ylim(0, max(.50,float(np.max(sharp_mean+sharp_std))*1.05))
axA.grid(axis="y", lw=0.3, color="#d8dce2", zorder=0)

# panel (b)
axB.axhline(0, color="#9aa3ad", lw=0.7, zorder=0)
axB.plot(angles, d_sup, "-o", color=C_SUP, lw=1.0, ms=3.4,
         label=r"M3$\to$M4 (supervision)", zorder=3)
axB.plot(angles, d_fv, "-s", color=C_FV, lw=1.0, ms=3.2,
         label=r"M6$\to$M7 (local FV)", zorder=3)
axB.set_xticks(angles); axB.set_xticklabels([f"{a:g}" for a in angles], fontsize=6.4)
axB.set_xlabel("ray angle from injector (deg)", fontsize=7.2)
axB.set_ylabel(r"$\Delta$ sharpness ratio", fontsize=7.2)
axB.set_title(r"$\bf{d}$  ray-wise mean changes", fontsize=7.8, pad=3, loc="left")
axB.tick_params(labelsize=6.8, length=2.2)
axB.grid(axis="y", lw=0.3, color="#d8dce2", zorder=0)
axB.legend(fontsize=6.2, frameon=False, loc="upper center", bbox_to_anchor=(.5,-.48), ncol=2, handletextpad=0.4,
           borderaxespad=0.2)

# Paired seed variability at each angle, without treating rays as replicates.
paired=by_ray.pivot(index=["seed","ray_angle_deg"],columns="exp_name",values=col)
for start,end,c in [("M3","M4",C_SUP),("M6","M7",C_FV)]:
    diff=paired[end]-paired[start]
    stat=diff.groupby("ray_angle_deg").agg(["mean","std","count"])
    assert (stat["count"]==4).all()
    axB.errorbar(stat.index,stat["mean"],yerr=stat["std"],fmt="none",ecolor=c,elinewidth=.6,capsize=1.4)
    stat.to_csv(SRC/f"Figure10_{start}_{end}_paired.csv")
axB.set_xticklabels([f"{a:g}" for a in angles],rotation=35,ha="right",fontsize=6)
save(fig,"Front_ray_panels_cd")

