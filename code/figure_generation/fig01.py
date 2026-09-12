import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.patches import FancyArrowPatch, Rectangle, Wedge, Circle, FancyBboxPatch, Polygon
from matplotlib.lines import Line2D

# ---- publication styling -----------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 8.5,
    "axes.linewidth": 0.9,
    "axes.titlesize": 9.5,
    "savefig.dpi": 300,
})

C_TRUTH  = "#1a1a1a"
C_VAN    = "#c8102e"   # vanilla PINN (failure)
C_OURS   = "#1f63d6"   # our model
C_PLUME  = "#9ec3ef"
C_PLUME2 = "#cfe0f7"
C_BAND   = "#f7b500"   # front band highlight
C_CV     = "#2a8c5a"   # control volumes
from shared import OUT, SRC, save
OUT.mkdir(parents=True, exist_ok=True)

fig = plt.figure(figsize=(7.4, 2.7))
gs = fig.add_gridspec(1, 3, width_ratios=[1.02, 1.0, 1.0], wspace=0.36,
                      left=0.012, right=0.988, bottom=0.13, top=0.83)
TITLE_FS = 8.0


def early_finger_radius(theta, base=2.72):
    """Illustrative fingered front; not a reference snapshot."""
    theta = np.asarray(theta)
    centers = np.deg2rad([9, 15, 21, 27, 33, 39, 45, 51, 57, 63, 69, 75, 81])
    amps = np.array([0.11, 0.17, 0.23, 0.31, 0.36, 0.33, 0.28,
                     0.30, 0.35, 0.31, 0.25, 0.18, 0.12])
    sigma = np.deg2rad(1.55)
    radius = base + 0.08 * np.sin(1.7 * theta + 0.25) - 0.06 * np.cos(3.2 * theta)
    for center, amp in zip(centers, amps):
        radius += amp * np.exp(-0.5 * ((theta - center) / sigma) ** 2)
    return radius


def add_radial_polygon(ax, theta, radius, color, zorder, clip, alpha=1.0):
    verts = [(0.0, 0.0)]
    verts.extend(zip(radius * np.cos(theta), radius * np.sin(theta)))
    verts.append((0.0, 0.0))
    patch = Polygon(verts, closed=True, facecolor=color, edgecolor="none",
                    alpha=alpha, zorder=zorder)
    patch.set_clip_path(clip)
    ax.add_patch(patch)
    return patch


def add_annular_polygon(ax, theta, inner_radius, outer_radius, color, zorder, clip, alpha=1.0):
    outer = list(zip(outer_radius * np.cos(theta), outer_radius * np.sin(theta)))
    inner = list(zip(inner_radius[::-1] * np.cos(theta[::-1]),
                     inner_radius[::-1] * np.sin(theta[::-1])))
    patch = Polygon(outer + inner, closed=True, facecolor=color, edgecolor="none",
                    alpha=alpha, zorder=zorder)
    patch.set_clip_path(clip)
    ax.add_patch(patch)
    return patch

# =============================================================================
# Panel (a): two-well domain + plume + sharp front + front band
# =============================================================================
axA = fig.add_subplot(gs[0, 0])
L = 5.0
axA.set_xlim(-0.35, L + 0.35); axA.set_ylim(-0.35, L + 0.35)
axA.set_aspect("equal"); axA.axis("off")

# domain
axA.add_patch(Rectangle((0, 0), L, L, facecolor="#f4f6f9",
                        edgecolor="#3b3b3b", lw=1.1, zorder=0))

# Illustrative plume geometry, not quantitative field data.
inj = (0.0, 0.0)
clip = Rectangle((0, 0), L, L, transform=axA.transData)
th = np.linspace(0, np.pi / 2, 700)
r_front = early_finger_radius(th)
r_outer = r_front + 0.40
r_band_inner = r_front - 0.25
r_band_outer = r_front + 0.21

add_radial_polygon(axA, th, r_outer, C_PLUME2, zorder=1, clip=clip)
add_radial_polygon(axA, th, r_front + 0.03, C_PLUME, zorder=1.5, clip=clip)
for r, col in [(2.25, "#7fb0e8"), (1.45, "#5f99e0")]:
    wedge = Wedge(inj, r, 0, 90, facecolor=col, edgecolor="none", zorder=2)
    wedge.set_clip_path(clip)
    axA.add_patch(wedge)

# front band (annulus) + sharp fingered front
add_annular_polygon(axA, th, r_band_inner, r_band_outer, C_BAND, zorder=3,
                    clip=clip, alpha=0.55)
axA.plot(r_front * np.cos(th), r_front * np.sin(th), color="#0b3d91", lw=2.2, zorder=4,
         solid_capstyle="round")

# propagation arrow
axA.add_patch(FancyArrowPatch((1.35, 1.35), (2.55, 2.55), arrowstyle="-|>",
                              mutation_scale=11, lw=1.4, color="#0b3d91", zorder=5))

# wells
axA.add_patch(Wedge(inj,.5,0,90,fc="white",ec="#0b3d91",lw=1.1,zorder=6))
axA.add_patch(Wedge((L,L),.5,180,270,fc="white",ec="#7a2048",lw=1.1,zorder=6))

axA.text(0.18, -0.05, "Injector", color="#0b3d91", fontsize=7.6,
         ha="left", va="top")
axA.text(4.78, 4.62, "Producer /\nmonitoring", color="#7a2048", fontsize=7.4,
         ha="right", va="top")
front_label_theta = np.deg2rad(58)
front_label_r = early_finger_radius(front_label_theta)
band_label_theta = np.deg2rad(16)
band_label_r = early_finger_radius(band_label_theta) + 0.12
axA.annotate("fingered\nsharp front",
             xy=(front_label_r * np.cos(front_label_theta),
                 front_label_r * np.sin(front_label_theta)),
             xytext=(1.78, 4.42), fontsize=7.2, color="#0b3d91", ha="center",
             arrowprops=dict(arrowstyle="->", lw=0.8, color="#0b3d91"))
axA.annotate("front band\n$S_{\\mathrm{CO_2}}\\in[0.05,0.30]$",
             xy=(band_label_r * np.cos(band_label_theta),
                 band_label_r * np.sin(band_label_theta)),
             xytext=(4.05, 1.15), fontsize=6.8, color="#7a5b00", ha="center",
             arrowprops=dict(arrowstyle="-", lw=0.8, color="#7a5b00"))
axA.text(0.92, 0.6, "CO$_2$\nplume", color="#08306b", fontsize=7.4, ha="center",
         va="center")
axA.set_title("(a)  Two-well CO$_2$ displacement", loc="left", pad=6,
              fontsize=TITLE_FS)

# =============================================================================
# helper: saturation profile across the front
# =============================================================================
def s_profile(x, x_f, width, s_hi=0.8):
    """Smooth front: high behind (small x), zero ahead (large x)."""
    return 0.5 * s_hi * (1.0 - np.tanh((x - x_f) / width))

x = np.linspace(0, 1, 400)

# =============================================================================
# Panel (b): the failure -- smearing + breakthrough delay
# =============================================================================
axB = fig.add_subplot(gs[0, 1])
xf_true, xf_van = 0.52, 0.40
s_true = s_profile(x, xf_true, 0.018)          # sharp, correct position
s_van  = s_profile(x, xf_van, 0.085, 0.78)     # smeared AND lagging

# vertical markers of the two front positions
axB.plot([xf_true, xf_true], [0, 0.86], color=C_TRUTH, lw=0.8, ls=":", zorder=1)
axB.plot([xf_van, xf_van], [0, 0.86], color=C_VAN, lw=0.8, ls=":", zorder=1)

axB.plot(x, s_true, color=C_TRUTH, lw=2.0, zorder=4, label="Illustrative sharp front")
axB.plot(x, s_van, color=C_VAN, lw=2.0, ls="--", zorder=3, label="Broadened / shifted")

# front-width error: 10-90% span of the smeared curve, near top
# (labels sit immediately left of their arrows, in clear space, so nothing
#  separates a label from the arrow it describes)
axB.annotate("", xy=(0.305, 0.62), xytext=(0.495, 0.62),
             arrowprops=dict(arrowstyle="<->", color=C_VAN, lw=1.1))
axB.text(0.29, 0.62, "front-width\nerror", color=C_VAN, fontsize=6.7,
         ha="right", va="center")
# front-position lag -> delayed breakthrough, between the two markers near bottom
axB.annotate("", xy=(xf_van, 0.12), xytext=(xf_true, 0.12),
             arrowprops=dict(arrowstyle="<->", color=C_VAN, lw=1.1))
axB.text(0.372, 0.12, "position\noffset", color=C_VAN,
         fontsize=6.7, ha="right", va="center")

axB.set_title("(b)  Front-reconstruction difficulties", loc="left", pad=6,
              fontsize=TITLE_FS)


axB.set(xlim=(0,1),ylim=(0,1.02),xticks=[],yticks=[0,.4,.8])
axB.set_xlabel("position across front (schematic)",fontsize=7)
axB.set_ylabel("CO$_2$ saturation",fontsize=7)
axB.spines["top"].set_visible(False);axB.spines["right"].set_visible(False)
axB.legend(loc="upper right",fontsize=5.8,frameon=False,handlelength=1.4)
axC=fig.add_subplot(gs[0,2]);axC.set(xlim=(0,1),ylim=(0,1));axC.axis("off")
axC.set_title("(c)  Front-aware approach",loc="left",fontsize=TITLE_FS,pad=6)
axC.text(.5,.91,"Saturation-specific representation",ha="center",color=C_OURS,fontsize=7)
axC.add_patch(Rectangle((.05,.13),.9,.61,fc="#f4f6f9",ec="#666666",lw=.8))
yy=np.linspace(.13,.74,200); xx=.50+.08*np.sin(10*yy)
axC.fill_betweenx(yy,.05,xx,color=C_PLUME2)
axC.plot(xx,yy,color="#0b3d91",lw=1.4)
for cy in [.28,.48,.66]:
    cx=.50+.08*np.sin(10*cy)
    axC.add_patch(Rectangle((cx-.075,cy-.075),.15,.15,fc="none",ec=C_CV,lw=1))
axC.scatter([.46,.59],[.42,.42],color=C_OURS,s=18,zorder=4)
axC.plot([.46,.59],[.42,.42],color=C_OURS,lw=1.3)
axC.text(.14,.56,"front / plume\nlabels",color=C_OURS,fontsize=6.5)
axC.text(.72,.22,"local FV\npenalty",color=C_CV,fontsize=6.5,ha="center")
axC.text(.5,.045,"Spatial control volumes (schematic)",fontsize=6.5,ha="center")
axA.text(2.5,-.66,"5 m square; well radius 0.5 m",ha="center",fontsize=6)
np.savetxt(SRC/"Figure1_schematic_profiles.csv",np.column_stack([x,s_true,s_van]),delimiter=",",
           header="schematic_position,illustrative_sharp,broadened_shifted",comments="")
save(fig,"Figure1")
