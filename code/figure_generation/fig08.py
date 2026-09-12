import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "dejavuserif", "font.size": 8.0,
    "axes.linewidth": 0.7,
})

from shared import ROOT,OUT,SRC,save,capture

METRICS = ["front_band_RMSE_pooled", "contour_chamfer_S018",
           "front_position_abs_err", "front_excess_width",
           "S_rel_l2", "p_RMSE_MPa",
           "local_FV_CO2_RMSE", "CO2_mass_rel_error"]
COLLAB = {
    "front_band_RMSE_pooled": "front-band\n$S$ RMSE",
    "contour_chamfer_S018": "front\ncontour\nChamfer",
    "front_position_abs_err": "front\nposition\n|error|",
    "front_excess_width": "front\nexcess\nwidth",
    "S_rel_l2": "global $S$\nrel. $L_2$",
    "p_RMSE_MPa": "pressure\nRMSE",
    "local_FV_CO2_RMSE": "local FV CO$_2$\nimbalance",
    "CO2_mass_rel_error": "node inventory\nproxy error",
}
ROWS = ["L1", "L2", "L3", "L4", "L5", "L6"]
ROWLAB = {
    "L1": "L1  representation (→ plain TwoNet)",
    "L2": "L2  front/plume supervision",
    "L3": "L3  pairwise front-gradient",
    "L4": "L4  adaptive resampling (RAR)",
    "L5": "L5  local FV regularization",
    "L6": "L6  coarse/detail branch",
}
# family colours (Fig. 4 convention)
FAM = {"L1": "#1f63d6", "L6": "#1f63d6",      # representation
       "L2": "#e07b00", "L3": "#e07b00",      # front supervision
       "L4": "#2a8c5a",                        # adaptive sampling
       "L5": "#c8102e"}                        # conservation


paired=pd.read_csv(capture(ROOT/"06_evidence/H_tables_v2/paired_effects.csv"))
paired=paired[paired.study=="L0_L6"].set_index(["role","metric"])
rays=pd.read_csv(capture(ROOT/"06_evidence/H_rays_strategies_v1/LOO_paired.csv")).set_index(["removed","metric"])
roles=["PLAIN_TWONET","FRONT_PLUME","PAIRGRAD","RAR","FV","COARSE_DETAIL"]
keys=["S_RMSE_front_band","conditional_chamfer_m","position","excess_width","S_rel_l2",
      "p_phys_RMSE_MPa","local_FV_CO2_RMSE","CO2_mass_rel_error_mean"]
values=[]; spreads=[]
for role in roles:
    values.append([float(rays.loc[(role,k),"mean"]) if k in ["position","excess_width"] else float(paired.loc[(role,k),"mean_percent"]) for k in keys])
    spreads.append([float(rays.loc[(role,k),"SD"]) if k in ["position","excess_width"] else float(paired.loc[(role,k),"SD_percent"]) for k in keys])
M=pd.DataFrame(values,index=ROWS,columns=METRICS)
M.to_csv(SRC/"Figure8_mean_percent.csv")
pd.DataFrame(spreads,index=ROWS,columns=METRICS).to_csv(SRC/"Figure8_SD_percent.csv")

VMAX = 60.0
fig, ax = plt.subplots(figsize=(7.48, 3.12), dpi=300)
fig.subplots_adjust(left=0.305, right=0.985, top=0.76, bottom=0.13)

norm = TwoSlopeNorm(vcenter=0.0, vmin=-VMAX, vmax=VMAX)
im = ax.imshow(np.clip(M.values, -VMAX, VMAX), cmap="RdBu_r", norm=norm,
               aspect="auto")

# cell annotations
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        v = M.values[i, j]
        ax.text(j, i, f"{v:+.0f}%", ha="center", va="center", fontsize=7.0,
                color="white" if abs(np.clip(v, -VMAX, VMAX)) > 0.62 * VMAX else "#1a1a1a")

# grid lines between cells + axis separators
for x in np.arange(0.5, 7.5, 1.0):
    ax.axvline(x, color="white", lw=1.2)
for y in np.arange(0.5, 5.5, 1.0):
    ax.axhline(y, color="white", lw=1.2)
for x in (3.5, 5.5):
    ax.axvline(x, color="#1a1a1a", lw=1.1)

# column labels on top, group headers above them
ax.xaxis.set_ticks_position("top")
ax.set_xticks(range(8))
ax.set_xticklabels([COLLAB[m] for m in METRICS], fontsize=6.0)
for lab in ax.get_xticklabels():
    lab.set_linespacing(0.90)
ax.tick_params(axis="x", length=0, pad=7)
ax.set_ylim(M.shape[0] - 0.5, -1.22)
for spine in ax.spines.values():
    spine.set_visible(False)
for x0, x1, lab in ((-0.5, 3.5, "front axis"), (3.5, 5.5, "global axis"),
                    (5.5, 7.5, "consistency")):
    xc = (x0 + x1) / 2
    ax.annotate(lab, xy=(xc, -0.92), xycoords=("data", "data"),
                ha="center", va="center", fontsize=7.0, style="italic",
                annotation_clip=False)
    ax.plot([x0 + 0.08, x1 - 0.08], [-0.74, -0.74], color="#1a1a1a", lw=0.75,
            clip_on=False)
ax.add_patch(Rectangle((-0.5, -0.5), M.shape[1], M.shape[0], fill=False,
                       edgecolor="#1a1a1a", lw=0.8, clip_on=False))

# row labels with family swatches
ax.set_yticks(range(6))
ax.set_yticklabels([ROWLAB[r] for r in ROWS], fontsize=7.2)
ax.tick_params(axis="y", length=0, pad=14)
for i, r in enumerate(ROWS):
    ax.add_patch(Rectangle((-0.62, i - 0.18), 0.09, 0.36, facecolor=FAM[r],
                           edgecolor="none", clip_on=False))

# colourbar (horizontal, bottom)
cax = fig.add_axes([0.40, 0.075, 0.36, 0.035])
cb = fig.colorbar(im, cax=cax, orientation="horizontal",extend="both")
cb.set_ticks([-60, -30, 0, 30, 60])
cb.ax.set_xticklabels(["−60", "−30", "0", "+30", "+60"], fontsize=6.4)
cb.outline.set_linewidth(0.5)
cb.ax.tick_params(length=2)
fig.text(0.40 - 0.012, 0.0925, "removal improves ←", ha="right", fontsize=6.6,
         color="#1f63d6")
fig.text(0.76 + 0.012, 0.0925, "→ removal worsens (%)", ha="left", fontsize=6.6,
         color="#c8102e")

fig.text(.5,-.025,"Cell values are uncapped paired means; colors saturate at ±60%. Four-seed SDs: Table 7.",ha="center",fontsize=6.5)
save(fig,"Figure8")

