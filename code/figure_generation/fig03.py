import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Rectangle, FancyArrowPatch, FancyBboxPatch

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "dejavuserif", "font.size": 8.6,
    "axes.linewidth": 0.9, "savefig.dpi": 300,
})

C_INJ  = "#0b3d91"
C_PROD = "#7a2048"
C_DOM  = "#f4f6f9"
C_NOF  = "#5f6b7a"
C_BAND = "#f7b500"
L, R = 5.0, 0.5

fig, ax = plt.subplots(figsize=(4.7, 4.5))
ax.set_xlim(-1.25, 6.75); ax.set_ylim(-1.15, 6.05)
ax.set_aspect("equal"); ax.axis("off")

# ---- domain + a faint CO2 plume to orient the reader ------------------------
ax.add_patch(Rectangle((0, 0), L, L, facecolor=C_DOM, edgecolor="none", zorder=0))
clip = Rectangle((0, 0), L, L, transform=ax.transData)
for r, c in [(3.0, "#dcebf6"), (2.3, "#cfe0f7"), (1.5, "#bcd4f2")]:
    w = Wedge((0, 0), r, 0, 90, facecolor=c, edgecolor="none", zorder=1)
    w.set_clip_path(clip); ax.add_patch(w)
wf = Wedge((0, 0), 2.55, 0, 90, width=0.4, facecolor=C_BAND, alpha=0.45,
           edgecolor="none", zorder=2); wf.set_clip_path(clip); ax.add_patch(wf)

# ---- no-flow outer edges: hatch ticks on all four sides ---------------------
ax.add_patch(Rectangle((0, 0), L, L, fill=False, edgecolor="#2b2b2b",
                        lw=1.6, zorder=5))
n_t = 13
for s in np.linspace(0.12, L - 0.12, n_t):
    d = 0.12
    ax.plot([s, s], [L, L + d], color=C_NOF, lw=0.8, zorder=4)      # top
    ax.plot([s, s], [0, -d], color=C_NOF, lw=0.8, zorder=4)          # bottom
    ax.plot([L, L + d], [s, s], color=C_NOF, lw=0.8, zorder=4)       # right
    ax.plot([0, -d], [s, s], color=C_NOF, lw=0.8, zorder=4)          # left
ax.text(L / 2, L + 0.30, "No-flow boundary  $\\mathbf{v}\\cdot\\mathbf{n}=0$",
        color=C_NOF, fontsize=8.2, ha="center", va="bottom")

# ---- injection well (rate-controlled), quarter disk at origin ---------------
ax.add_patch(Wedge((0, 0), R, 0, 90, facecolor=C_INJ, edgecolor="white",
                   lw=0.8, zorder=6))
for th in (22, 45, 68):
    a = np.deg2rad(th)
    ax.add_patch(FancyArrowPatch((R * np.cos(a), R * np.sin(a)),
                                 (1.18 * np.cos(a), 1.18 * np.sin(a)),
                                 arrowstyle="-|>", mutation_scale=10,
                                 lw=1.3, color=C_INJ, zorder=6))
ax.annotate(
    "Injection well (rate-controlled), $r_{\mathrm{well}}=0.5$ m\n"
    "$\\mathbf{v}\\cdot\\mathbf{n}_{\\mathrm{in}}=U_{\\mathrm{in}}=5.4\\times10^{-5}$ m s$^{-1}$\n"
    "$S_{\\mathrm{CO_2}}=1-S_{w,\\mathrm{irr}}=0.8$",
    xy=(0.66, 0.40), xytext=(3.05, -0.78), fontsize=7.4, color=C_INJ, ha="center",
    va="center", arrowprops=dict(arrowstyle="->", lw=0.9, color=C_INJ))

# ---- production / monitoring well (pressure-controlled) at (L,L) ------------
ax.add_patch(Wedge((L, L), R, 180, 270, facecolor=C_PROD, edgecolor="white",
                   lw=0.8, zorder=6))
for th in (200, 225, 250):
    a = np.deg2rad(th)
    ax.add_patch(FancyArrowPatch((L + 1.18 * np.cos(a), L + 1.18 * np.sin(a)),
                                 (L + R * np.cos(a), L + R * np.sin(a)),
                                 arrowstyle="-|>", mutation_scale=10,
                                 lw=1.3, color=C_PROD, zorder=6))
ax.annotate(
    "Production / monitoring well\n"
    "$p=p_0=10$ MPa  (no backflow)",
    xy=(L - 0.55, L - 0.55), xytext=(1.95, 4.15), fontsize=7.4, color=C_PROD,
    ha="center", va="center", arrowprops=dict(arrowstyle="->", lw=0.9, color=C_PROD))

# ---- initial condition box (open lower-right) -------------------------------
ax.add_patch(FancyBboxPatch((3.42, 0.95), 1.95, 0.92,
                            boxstyle="round,pad=0.04,rounding_size=0.10",
                            facecolor="white", edgecolor="#9aa3ad", lw=0.8,
                            zorder=7))
ax.text(4.40, 1.41, "Initial state ($t=0$)\n$S_{\\mathrm{CO_2}}=0,\\;\\; p=p_0$",
        fontsize=7.4, color="#333333", ha="center", va="center", zorder=8)

# ---- size annotation: L_ref as a vertical dimension on the left edge --------
ax.annotate("", xy=(-0.55, 0), xytext=(-0.55, L),
            arrowprops=dict(arrowstyle="<->", lw=0.9, color="#555555"))
ax.text(-0.72, L / 2, "$L_{\\mathrm{ref}} = 5$ m", fontsize=7.6, color="#555555",
        ha="center", va="center", rotation=90)
ax.text(2.42, 1.98, "CO$_2$ plume", fontsize=7.4, color="#3a5a86",
        ha="center", rotation=-45, zorder=3)

from shared import save
save(fig,'Figure3')

