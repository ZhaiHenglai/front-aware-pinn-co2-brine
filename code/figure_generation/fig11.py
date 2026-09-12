import glob
import os
import re
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
C_M4, C_M5, C_M6, C_M7, INK = "#e07b00", "#9c6b00", "#2a8c5a", "#c8102e", "#1a1a1a"


from shared import ROOT,OUT,SRC,save,capture
import numpy as np
ho=pd.read_csv(capture(ROOT/"06_evidence/H_auxiliary_v1/holdout_summary.csv"))
counts=pd.read_csv(capture(ROOT/"06_evidence/H_auxiliary_v1/holdout_by_seed.csv"))
ho["mode"]=np.where(ho.partition.str.contains("_rf"),"random","temporal")
ho["kept_pct"]=0.0
for i,r in ho.iterrows():
    if r["mode"]=="random":
        ho.loc[i,"kept_pct"]=100*(1-float(re.search(r"rf([0-9.]+)",r.partition).group(1)))
    else:
        every=int(re.search(r"he(\d+)",r.partition).group(1))
        ho.loc[i,"kept_pct"]=100*(1-1/every)
ho["heldout"]=ho.heldout_mean
gfull=pd.read_csv(capture(ROOT/"06_evidence/H_tables_v2/by_seed.csv"))
anchor={m:gfull[(gfull.study=="M0_M7")&(gfull.role==m)].S_rel_l2.mean() for m in ["M4","M7"]}
ho.to_csv(SRC/"Figure11a_source.csv",index=False)
bl=pd.read_csv(capture(ROOT/"06_evidence/H_auxiliary_v1/BL_by_seed.csv"))
bl=bl.sort_values("study").drop_duplicates(["exp","seed","n_labels"])
free=bl[bl.exp=="FREE"].front_band_RMSE.mean()
diff=bl[bl.exp=="DIFF"].front_band_RMSE.mean()
sw=(bl[bl.exp.isin(["M4","M5","M6_RAR","M7_RAR_FV"])]
    .groupby(["exp","n_labels"]).front_band_RMSE.agg(["mean","std","count"]).reset_index().sort_values("n_labels"))
assert (sw["count"]==4).all()
bl.to_csv(SRC/"Figure11b_source.csv",index=False)

fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.48, 2.55))
fig.subplots_adjust(left=0.075, right=0.99, top=0.875, bottom=0.185, wspace=0.27)

# (a)
for m, c, mk in (("M4", C_M4, "o"), ("M7", C_M7, "s")):
    r = ho[(ho.model == m) & (ho["mode"] == "random")].sort_values("kept_pct")
    axa.errorbar(r.kept_pct, r.heldout, yerr=r.heldout_SD, capsize=1.6, color=c, lw=1.1, marker=mk, ms=4.3,
             label=f"{m}, random partition")
    t = ho[(ho.model == m) & (ho["mode"] == "temporal")]
    axa.errorbar(t.kept_pct,t.heldout,yerr=t.heldout_SD,fmt="none",ecolor=c,capsize=1.6)
    axa.scatter(t.kept_pct, t.heldout, facecolor="white", edgecolor=c, marker="^",
                s=26, linewidths=1.0, zorder=4,
                label=f"{m}, temporal")
    axa.scatter([100], [anchor[m]], facecolor="white", edgecolor=c, marker=mk,
                s=30, linewidths=1.1, zorder=4)
axa.scatter([], [], facecolor="white", edgecolor="#555555", marker="^", s=26,
            linewidths=1.0)  # legend spacing no-op
axa.set_xlim(105, 5)  # reversed: full data on the left
axa.set_ylim(0.0, 0.34)
axa.set_yticks([0.0, 0.1, 0.2, 0.3])
axa.set_xticks([100, 80, 50, 25, 10])
axa.set_xlabel("nominal supervised fraction (%)", fontsize=7.5)
axa.set_ylabel("held-out $S$ rel. $L_2$", fontsize=7.5)
axa.tick_params(labelsize=6.8, length=2.2)
axa.text(0.03, 0.965, "four-seed mean $\\pm$ SD; separate H holdouts\n"
         "(open symbols at 100%: full-data in-sample value)",
         transform=axa.transAxes, fontsize=6.3, color="#444444", va="top")
axa.legend(fontsize=6.4, frameon=False, loc="lower right", ncol=1, handlelength=1.6)
axa.set_title("2D H: holdout and label fraction", fontsize=7.6, pad=3)
axa.text(0.01, 1.045, "$\\bf{a}$", transform=axa.transAxes, fontsize=8.5, va="bottom")

# (b)
axb.axhline(free, color=INK, lw=0.9, ls="--", label="data-free (no labels)")
axb.axhline(diff, color="#8a94a3", lw=0.9, ls=":", label="data-free + artif. diffusion")
for exp, c, lw, mk, lab in (("M4", C_M4, 1.3, "o", "M4 front supervision"),
                            ("M5", C_M5, 1.1, "D", "M5 + pairwise gradient"),
                            ("M6_RAR", C_M6, 0.8, "^", "+ RAR"),
                            ("M7_RAR_FV", C_M7, 0.8, "s", "+ RAR + FV")):
    s = sw[sw.exp == exp]
    yerr = s["std"].values if (s["count"] > 1).any() else None
    axb.errorbar(s.n_labels, s["mean"], yerr=yerr, color=c, lw=lw, marker=mk,
                 ms=3.6, ls="-" if exp in ("M4", "M5") else "--", label=lab,
                 elinewidth=0.7, capsize=1.6)
axb.set_xscale("log")
axb.set_xticks([8, 15, 30, 60, 120])
axb.set_xticklabels(["8", "15", "30", "60", "120"])
axb.set_xlim(7, 135)
axb.set_ylim(0, max(.0175,float((sw["mean"]+sw["std"]).max())*1.2))
axb.set_xlabel("number of labels $N$", fontsize=7.5)
axb.set_ylabel("front-band RMSE vs Welge", fontsize=7.5)
axb.tick_params(labelsize=6.8, length=2.2)
axb.legend(fontsize=6.0, frameon=False, loc="upper center", ncol=2,
           handlelength=1.6, columnspacing=0.8, borderaxespad=0.15)
axb.set_title("1D Buckley–Leverett: sparse-label regime", fontsize=7.6, pad=3)
axb.text(0.01, 1.045, "$\\bf{b}$", transform=axb.transAxes, fontsize=8.5, va="bottom")

save(fig,'Figure11')
