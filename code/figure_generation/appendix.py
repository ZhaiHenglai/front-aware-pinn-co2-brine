import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"],
                     "mathtext.fontset": "dejavuserif", "font.size": 8.0,
                     "axes.linewidth": 0.7})

from shared import ROOT,OUT,SRC,FW,save,capture
EV=FW/"eval_seed0_all/M0_M7"

MC = {"M0": "#8a94a3", "M1": "#9db8e8", "M2": "#4d82dd", "M3": "#1f63d6",
      "M4": "#f0a04b", "M5": "#e07b00", "M6": "#2a8c5a", "M7": "#c8102e"}
MODELS = [f"M{i}" for i in range(8)]

# ---------------- B.2 per-time curves ----------------
bt = pd.read_csv(capture(EV / "global/by_time_global_accuracy.csv"))
bt["M"] = bt.run_id.str.extract(r"^(M\d)")
bt["t_d"] = bt.time_s / 86400.0
# Saturation accuracy panel uses absolute RMSE (not relative L2): the per-time
# relative L2 diverges at injection onset where the reference CO2 norm -> 0
# (S_rel_l2 ~ 1e30 at t0). S_rmse is well-defined at every level and consistent
# with the front-band/pressure RMSE panels.
bt[["M", "t_d", "S_rmse", "S_RMSE_front_band", "p_phys_RMSE_MPa"]].to_csv(
    SRC / "FigureB2_source.csv", index=False)
PAN = [("S_rmse", "saturation RMSE", (0, 0.16)),
       ("S_RMSE_front_band", "front-band $S$ RMSE", (0, 0.12)),
       ("p_phys_RMSE_MPa", "$p$ RMSE (MPa)", (0, 0.6))]
fig, axes = plt.subplots(1, 3, figsize=(7.48, 2.25))
fig.subplots_adjust(left=0.065, right=0.99, top=0.86, bottom=0.30, wspace=0.30)
for ax, (col, lab, yl) in zip(axes, PAN):
    for m in MODELS:
        s = bt[bt.M == m].sort_values("t_d")
        ax.plot(s.t_d, s[col], color=MC[m], lw=1.5 if m == "M7" else 0.8,
                label=m, zorder=3 if m == "M7" else 2)
    ax.set_xlabel("time (days)", fontsize=7.3)
    ax.set_title(lab, fontsize=7.6, pad=3)
    ax.set_ylim(0,max(yl[1],float(bt[col].max())*1.05)); ax.set_xlim(0, bt.t_d.max())
    ax.tick_params(labelsize=6.8, length=2.2)
    ax.grid(lw=0.3, color="#e3e6ea")
fig.legend(*axes[0].get_legend_handles_labels(), ncol=8, fontsize=6.5,
           frameon=False, loc="lower center", bbox_to_anchor=(0.5, 0.0),
           handlelength=1.4, columnspacing=1.0)
save(fig,"FigureB2")
plt.close(fig)

# ---------------- B.3 snapshot gallery ----------------
ref_done = False
fig = plt.figure(figsize=(7.48, 7.9))
gs = fig.add_gridspec(3, 4, width_ratios=[1, 1, 1, 0.05], wspace=0.07,
                      hspace=0.14, left=0.055, right=0.92, top=0.965, bottom=0.045)
cmap = plt.get_cmap("viridis").copy(); cmap.set_bad("white")
panels = ["reference"] + MODELS
for k, name in enumerate(panels):
    r, c = divmod(k, 3)
    ax = fig.add_subplot(gs[r, c])
    d = np.load(capture(EV / "snapshots" / (name if name != "reference" else "M7") /
                "final" / "snapshot_grid.npz"))
    assert abs(float(d["time_seconds"])-31544.99609375)<1e-9
    if name=="reference":
        reference=d["S_true"].copy();gridx=d["X"].copy();gridy=d["Y"].copy()
    else:
        np.testing.assert_allclose(d["S_true"],reference,equal_nan=True)
        assert np.array_equal(d["X"],gridx) and np.array_equal(d["Y"],gridy)
    assert np.nanmin(d["S_pred"])>=0 and np.nanmax(d["S_pred"])<=.8
    F = d["S_true"] if name == "reference" else d["S_pred"]
    im = ax.pcolormesh(d["X"], d["Y"], np.ma.masked_invalid(F), cmap=cmap,
                       vmin=0, vmax=0.8, rasterized=True, shading="auto")
    ax.set_aspect("equal"); ax.set_xlim(0, 5); ax.set_ylim(0, 5)
    ax.set_xticks([0, 2.5, 5]); ax.set_yticks([0, 2.5, 5])
    if r < 2: ax.set_xticklabels([])
    else:
        ax.set_xlabel("$x$ (m)", fontsize=7.3)
        # suppress edge labels shared with the neighbouring panel so that
        # "5.0" and "0.0" do not collide at the tight interior boundaries
        ax.set_xticklabels(["0.0" if c == 0 else "", "2.5",
                            "5.0" if c == 2 else ""])
    if c == 0: ax.set_ylabel("$y$ (m)", fontsize=7.3)
    else: ax.set_yticklabels([])
    ax.tick_params(labelsize=6.6, length=2)
    ax.set_title(name, fontsize=7.8, pad=2.5,
                 color="#1a1a1a" if name == "reference" else MC.get(name, "k"))
cax = fig.add_subplot(gs[:, 3])
cb = fig.colorbar(im, cax=cax); cb.set_label("$S_{\\mathrm{CO_2}}$", fontsize=7.5)
cb.ax.tick_params(labelsize=6.6, length=2)
save(fig,"FigureB3")
plt.close(fig)

# ---------------- B.1 per-term loss trajectories (M7) ----------------
lp = pd.read_csv(capture(ROOT / "06_evidence/H_M7_log_rematch_v1/loss_points.csv"))
lp=lp[(lp.seed==0)&(lp.kind=="adam")]
assert len(lp)>50 and lp.x.max()==20000
m7 = lp[lp.exp_name == "M7"].sort_values("x")
m7.to_csv(SRC / "FigureB1_source.csv", index=False)
TERMS = [("loss", "total", "#1a1a1a", 1.5, "-"),
         ("dataS", "saturation data", "#4d82dd", 0.9, "-"),
         ("dataP", "pressure data", "#9db8e8", 0.9, "-"),
         ("Sf", "front-band term", "#e07b00", 0.9, "-"),
         ("Sfg", "pairwise gradient", "#9c6b00", 0.9, "--"),
         ("pde", "PDE residual", "#2a8c5a", 0.9, "-"),
         ("fv", "local FV", "#c8102e", 1.1, "-")]
fig, ax = plt.subplots(figsize=(7.48, 2.5))
fig.subplots_adjust(left=0.07, right=0.99, top=0.92, bottom=0.25)
for col, lab, c, lw, ls in TERMS:
    s = m7[m7[col] > 0]
    ax.plot(s.x, s[col], color=c, lw=lw, ls=ls, label=lab)
for it, lab in ((2000, "PDE ramp"), (4000, "pairgrad"), (8000, "FV")):
    ax.axvline(it, color="#888888", lw=0.7, ls=":")
    ax.text(it, .97, f" {lab}", transform=ax.get_xaxis_transform(), fontsize=6.2, color="#666666",
            va="top", rotation=90)
ax.set_yscale("log"); ax.set_xlim(0, 20000)
ax.set_xlabel("iteration", fontsize=7.5)
ax.set_ylabel("logged value (log scale)", fontsize=7.5)
ax.tick_params(labelsize=6.8, length=2.2)
ax.legend(fontsize=6.3, frameon=False, ncol=4, loc="upper right",
          handlelength=1.6, columnspacing=0.9)
fig.text(.5,.015,"Seed 0, matched main run. Components shown before outer weighting; total is weighted. Zero values omitted.",ha="center",fontsize=6.2)
save(fig,"FigureB1")
print("[done] FigureB1-B3 ->", OUT)

