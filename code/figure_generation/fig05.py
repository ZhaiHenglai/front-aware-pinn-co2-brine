import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "dejavuserif", "font.size": 8.4,
    "axes.linewidth": 0.9, "savefig.dpi": 300,
})

C_SUP = "#2e6fdb"   # supervised / training label
C_OUT = "#c8102e"   # held-out / test

nx, nt = 8, 6
xs = np.arange(nx); ts = np.arange(nt)
XX, TT = np.meshgrid(xs, ts)

rng = np.random.default_rng(3)
held = {}
# (a) random fraction ~25%
m = np.zeros((nt, nx), dtype=bool)
idx = rng.choice(nx * nt, size=int(round(0.25 * nx * nt)), replace=False)
m.flat[idx] = True
held["a"] = m
# (b) temporal hold-out: two whole time rows
m = np.zeros((nt, nx), dtype=bool); m[[2, 4], :] = True
held["b"] = m
# (c) spatial sub-region: contiguous band of columns
m = np.zeros((nt, nx), dtype=bool); m[:, [4, 5]] = True
held["c"] = m

titles = {
    "a": "(a)  Random-fraction (sparse)",
    "b": "(b)  Temporal interpolation",
    "c": "(c)  Spatial sub-region",
}
order = ["a", "b", "c"]

fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.55))
fig.subplots_adjust(left=0.055, right=0.99, bottom=0.20, top=0.83, wspace=0.18)

for ax, key in zip(axes, order):
    m = held[key]
    ax.add_patch(Rectangle((-0.5, -0.5), nx, nt,
                 facecolor="#f4f6f9", edgecolor="#c7ccd3", lw=0.8, zorder=0))
    sup_x = XX[~m]; sup_t = TT[~m]
    out_x = XX[m];  out_t = TT[m]
    ax.scatter(sup_x, sup_t, s=42, facecolor=C_SUP, edgecolor="white", lw=0.6,
               zorder=3)
    ax.scatter(out_x, out_t, s=46, facecolor="white", edgecolor=C_OUT, lw=1.5,
               zorder=4)
    ax.set_title(titles[key], loc="left", fontsize=9.0, pad=4)
    ax.set_xlim(-0.95, nx - 0.05); ax.set_ylim(-1.0, nt - 0.2)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("space  $x$", fontsize=8.0, labelpad=2)
    for sp in ax.spines.values():
        sp.set_visible(False)
axes[0].set_ylabel("time  $t$", fontsize=8.0, labelpad=2)
# small axis arrows for the first panel to orient space/time
axes[0].annotate("", xy=(nx - 1.0, -0.72), xytext=(-0.72, -0.72),
                 arrowprops=dict(arrowstyle="->", lw=0.8, color="#777777"))
axes[0].annotate("", xy=(-0.72, nt - 1.0), xytext=(-0.72, -0.72),
                 arrowprops=dict(arrowstyle="->", lw=0.8, color="#777777"))

handles = [
    Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=C_SUP,
           markeredgecolor="white", markersize=8, label="supervised label (training)"),
    Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="white",
           markeredgecolor=C_OUT, markeredgewidth=1.5, markersize=8,
           label="held-out point (test)"),
]
fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
           fontsize=7.8, bbox_to_anchor=(0.5, -0.02), handletextpad=0.4,
           columnspacing=1.6)

axes[2].text(.5,.95,'Not executed',transform=axes[2].transAxes,ha='center',color='#c8102e',fontsize=8,bbox=dict(facecolor='white',edgecolor='none'))
from shared import save
save(fig,'Figure5')

