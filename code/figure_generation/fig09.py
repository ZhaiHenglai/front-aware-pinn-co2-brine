import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "dejavuserif", "font.size": 8.0,
    "axes.linewidth": 0.7,
})
C_M0, C_M4, C_M7, INK = "#8a94a3", "#e07b00", "#c8102e", "#1a1a1a"


from shared import ROOT,OUT,SRC,FW,save,capture
prof=pd.read_csv(capture(FW/"multiray_front_M0_M7_BASE/multiray_front_profiles.csv"))
prof=prof[(prof.ray_angle_deg==45)&(prof.seed==0)&(prof.training_strategy=="BASE")&(prof.leave_out=="NONE")]
times=sorted(prof.time_s.unique());assert len(times)==3
bt=pd.read_csv(capture(FW/"standard_front_timing/breakthrough_curves.csv"))
bt=bt[(bt.seed==0)&(bt.training_strategy=="BASE")&(bt.leave_out=="NONE")]
btsum=pd.read_csv(capture(FW/"standard_front_timing/summary_heldout_front_timing.csv"))
btsum=btsum[(btsum.seed==0)&(btsum.training_strategy=="BASE")&(btsum.leave_out=="NONE")]
def crossing(t,y,threshold=.05):
    hit=np.flatnonzero(y>=threshold)
    if not len(hit):return np.nan
    i=hit[0]
    if i==0:return float(t[0])
    return float(t[i-1]+(threshold-y[i-1])*(t[i]-t[i-1])/(y[i]-y[i-1]))
for exp in ["M0","M4","M7"]:
    sub=bt[bt.exp_name==exp].sort_values("time_s")
    truth=crossing(sub.time_s.to_numpy(),sub.s_true.to_numpy())
    prediction=crossing(sub.time_s.to_numpy(),sub.s_pred.to_numpy())
    stored=float(btsum[btsum.exp_name==exp].iloc[0].bt_abs_error)
    assert abs(abs(prediction-truth)-stored)<1e-6

fig = plt.figure(figsize=(7.48, 2.45))
gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.45], wspace=0.30,
                      left=0.058, right=0.985, top=0.86, bottom=0.19)

# ---------------- (a) ray profiles ----------------
src_a = []
for k, ts in enumerate(times):
    ax = fig.add_subplot(gs[0, k])
    sub7 = prof[(prof.exp_name == "M7") & (prof.time_s == ts)].sort_values("arclength")
    ax.plot(sub7.arclength, sub7.s_true, color=INK, lw=1.5, label="reference")
    for exp, c, lw in (("M0", C_M0, 1.1), ("M4", C_M4, 1.1), ("M7", C_M7, 1.1)):
        s = prof[(prof.exp_name == exp) & (prof.time_s == ts)].sort_values("arclength")
        ax.plot(s.arclength, s.s_pred, color=c, lw=lw, label=exp)
        for _, row in s.iterrows():
            src_a.append(dict(panel=f"a{k+1}", time_s=ts, model=exp,
                              arclength_m=row.arclength, S_pred=row.s_pred,
                              S_ref=row.s_true))
    ax.set_xlim(sub7.arclength.min(), sub7.arclength.max()); ax.set_ylim(-0.02, 0.85)
    ax.set_xlabel("distance along ray (m)", fontsize=7.3)
    if k == 0:
        ax.set_ylabel("$S_{\\mathrm{CO_2}}$", fontsize=7.8)
    else:
        ax.set_yticklabels([])
    ax.tick_params(labelsize=6.8, length=2.2)
    ax.set_title(f"$t={ts/86400.0:.3f}$ d", fontsize=7.6, pad=2.5)
    ax.text(0.02, 1.03, f"$\\bf{{a_{k+1}}}$", transform=ax.transAxes,
            fontsize=8.5, va="bottom")
    if k == 0:
        ax.legend(fontsize=6.4, frameon=False, loc="upper right",
                  borderaxespad=0.1, handlelength=1.5)

# ---------------- (b) breakthrough ----------------
axb = fig.add_subplot(gs[0, 3])
THR = 0.05
sub = bt[bt.exp_name == "M4"].sort_values("time_s")
axb.plot(sub.time_s / 86400.0, sub.s_true, color=INK, lw=1.5, label="reference")
for exp, c in (("M0", C_M0), ("M4", C_M4), ("M7", C_M7)):
    s = bt[bt.exp_name == exp].sort_values("time_s")
    axb.plot(s.time_s / 86400.0, s.s_pred, color=c, lw=1.1, label=exp)
axb.axhline(THR, color="#666666", lw=0.7, ls="--")
axb.text(0.012, THR + 0.004, "breakthrough threshold", fontsize=6.2, color="#666666")
for exp, c, dy in (("M0", C_M0, -0.02), ("M4", C_M4, 0.015), ("M7", C_M7, 0.05)):
    row = btsum[btsum.exp_name == exp].iloc[0]
    axb.text(0.985, 0.70 + dy * 4, f"{exp}: $E_{{\\mathrm{{bt}}}}={row.bt_abs_error:.0f}$ s "
             f"({100*row.bt_rel_error:.1f}%)", transform=axb.transAxes,
             ha="right", fontsize=6.4, color=c)
axb.set_xlim(0, bt.time_s.max() / 86400.0); axb.set_ylim(-0.005, max(.16,bt[bt.exp_name.isin(["M0","M4","M7"])][["s_true","s_pred"]].max().max()*1.1))
axb.set_xlabel("time (days)", fontsize=7.3)
axb.set_ylabel("producer $S_{\\mathrm{CO_2}}$", fontsize=7.3)
axb.tick_params(labelsize=6.8, length=2.2)
axb.set_title("breakthrough at the producer", fontsize=7.6, pad=2.5)
axb.text(-0.13, 1.03, "$\\bf{b}$", transform=axb.transAxes, fontsize=8.5, va="bottom")
axb.legend(fontsize=6.4, frameon=False, loc="upper left", handlelength=1.5)

pd.DataFrame(src_a).to_csv(SRC / "Figure9a_source.csv", index=False)
bt.to_csv(SRC / "Figure9b_source.csv", index=False)
btsum.to_csv(SRC/"Figure9_breakthrough_summary.csv",index=False)
save(fig,"Figure9")
