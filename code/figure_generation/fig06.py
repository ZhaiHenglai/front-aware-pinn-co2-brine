import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "dejavuserif", "font.size": 8.0,
    "axes.linewidth": 0.7,
})


from shared import ROOT,OUT,SRC,save,capture
SNAP=ROOT/"06_evidence/H_fields_v1"

TIMES = ["early", "middle", "final"]
data = {t: np.load(capture(SNAP / f"{t}.npz")) for t in TIMES}

S_LIM = (0.0, 0.8)
DS_LIM = 0.10          # symmetric ΔS limits, identical across rows
DP_LIM = 0.2           # MPa

fig = plt.figure(figsize=(7.48, 8.0))
gs = fig.add_gridspec(4, 6, width_ratios=[1, 1, 0.032, 0.135, 1, 0.032],
                      wspace=0.075, hspace=0.13, left=0.092, right=0.955,
                      top=0.97, bottom=0.065)

def style_ax(ax, row, col):
    ax.set_aspect("equal")
    ax.set_xlim(0, 5); ax.set_ylim(0, 5)
    ax.set_xticks([0, 2.5, 5]); ax.set_yticks([0, 2.5, 5])
    if row < 3: ax.set_xticklabels([])
    else: ax.set_xlabel("$x$ (m)", fontsize=7.5)
    if col == 0: ax.set_ylabel("$y$ (m)", fontsize=7.5)
    else: ax.set_yticklabels([])
    ax.tick_params(labelsize=6.8, length=2.2)

cmap_f = plt.get_cmap("viridis").copy(); cmap_f.set_bad("white")
cmap_e = plt.get_cmap("RdBu_r").copy(); cmap_e.set_bad("white")

rows = []
for i, t in enumerate(TIMES):
    d = data[t]
    rows.append(dict(kind="S", true=d["S_true"], pred=d["S_pred"],
                     err=d["S_pred"] - d["S_true"], X=d["X"], Y=d["Y"],
                     tday=float(d["time_seconds"]) / 86400.0))
d = data["final"]
rows.append(dict(kind="p", true=d["p_true"] / 1e6, pred=d["p_pred"] / 1e6,
                 err=(d["p_pred"] - d["p_true"]) / 1e6, X=d["X"], Y=d["Y"],
                 tday=float(d["time_seconds"]) / 86400.0))

row_letters = "abcd"
manifest = []
for i, r in enumerate(rows):
    is_S = r["kind"] == "S"
    vmin, vmax = (S_LIM if is_S else
                  (10.0,11.4))
    elim = DS_LIM if is_S else DP_LIM
    axs = [fig.add_subplot(gs[i, j]) for j in (0, 1, 4)]
    ims = []
    for j, (ax, F) in enumerate(zip(axs, [r["true"], r["pred"], r["err"]])):
        cm, vl = (cmap_e, (-elim, elim)) if j == 2 else (cmap_f, (vmin, vmax))
        im = ax.pcolormesh(r["X"], r["Y"], np.ma.masked_invalid(F),
                           cmap=cm, vmin=vl[0], vmax=vl[1], rasterized=True,
                           shading="auto")
        ims.append(im)
        style_ax(ax, i, j)
    if i == 0:
        axs[0].set_title("reference", fontsize=7.8, pad=3)
        axs[1].set_title("M7 prediction", fontsize=7.8, pad=3)
        axs[2].set_title("signed error", fontsize=7.8, pad=3)
    qty = "$S_{\\mathrm{CO_2}}$" if is_S else "$p$ (MPa)"
    axs[0].text(-0.40, 0.5, f"$\\bf{{{row_letters[i]}}}$  {qty},  $t={r['tday']:.3f}$ d",
                transform=axs[0].transAxes, rotation=90, va="center", ha="center",
                fontsize=7.6)
    cax1 = fig.add_subplot(gs[i, 2]); cax2 = fig.add_subplot(gs[i, 5])
    cb1 = fig.colorbar(ims[1], cax=cax1); cb2 = fig.colorbar(ims[2], cax=cax2,extend="both")
    for cb in (cb1, cb2):
        cb.ax.tick_params(labelsize=6.2, length=2)
        cb.outline.set_linewidth(0.5)
    manifest.append(dict(row=i + 1, quantity=r["kind"], time_days=round(r["tday"], 4),
                         above_color_limit_percent=float(np.count_nonzero(np.abs(r["err"])>elim)/np.isfinite(r["err"]).sum()*100), max_absolute_error=float(np.nanmax(np.abs(r["err"])))))

import csv
with open(SRC / "Figure6_manifest.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(manifest[0]))
    w.writeheader(); w.writerows(manifest)

fig.text(.5,.006,"Seed 0; signed error = prediction − reference. Extended colorbars indicate values outside the displayed limits.",ha="center",fontsize=6.4)
save(fig,"Figure6")

