import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "dejavuserif", "font.size": 8.2,
    "savefig.dpi": 300,
})

fig = plt.figure(figsize=(7.4, 7.4 * 531 / 1328))
ax = fig.add_axes([0.0, 0.0, 1.0, 1.0]); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.axis("off")

stages = [
 ("1  Problem & data", "#5f6b7a","•  CO$_2$–brine\n    displacement\n•  H: homogeneous\n•  K: one prescribed field\n•  zero capillary pressure"),
 ("2  Reconstruction", "#0b7d63","•  simulator-data\n    anchored\n•  fixed case and controls\n•  separate H holdouts:\n    random / temporal"),
 ("3  Front-aware PINN", "#1f63d6","•  spectral saturation\n•  front / plume labels\n•  local FV soft penalty\n•  four training seeds"),
 ("4  Component effects", "#7a2048","•  H construction\n    + deletion\n•  limited K extension\n•  front / field accuracy\n•  local-balance trade-offs"),
]

n = len(stages)
margin, gap = 0.018, 0.028
bw = (1.0 - 2 * margin - (n - 1) * gap) / n
bh = 0.62
y0 = 0.19
hdr = 0.17
for i, (title, col, body) in enumerate(stages):
    x0 = margin + i * (bw + gap)
    cx = x0 + bw / 2
    # box
    ax.add_patch(FancyBboxPatch((x0, y0), bw, bh,
                 boxstyle="round,pad=0.004,rounding_size=0.02",
                 facecolor="#fbfbfc", edgecolor=col, lw=1.6, zorder=2))
    # header band
    ax.add_patch(FancyBboxPatch((x0, y0 + bh - hdr), bw, hdr,
                 boxstyle="round,pad=0.004,rounding_size=0.02",
                 facecolor=col, edgecolor=col, lw=1.0, zorder=3))
    ax.text(cx, y0 + bh - hdr / 2, title, ha="center", va="center",
            fontsize=9, color="white", fontweight="bold", zorder=4)
    # body, vertically centred in the area below the header
    ax.text(x0 + 0.014, y0 + (bh - hdr) / 2, body, ha="left", va="center",
            fontsize=7.5, color="#1a1a1a", zorder=4, linespacing=1.35)
    # arrow to next stage
    if i < n - 1:
        xa = x0 + bw
        ax.add_patch(FancyArrowPatch((xa + 0.002, y0 + bh / 2),
                     (xa + gap - 0.002, y0 + bh / 2),
                     arrowstyle="-|>", mutation_scale=11, lw=1.6,
                     color="#555555", zorder=5))

from shared import ROOT
import fitz
out = ROOT / '02_figures/numbered'
for ext in ['pdf', 'svg', 'png', 'tiff']:
    fig.savefig(out / f'graphical_abstract.{ext}', dpi=600, facecolor='white')
with fitz.open(out / 'graphical_abstract.pdf') as doc:
    doc[0].get_pixmap(matrix=fitz.Matrix(2,2)).save(out / 'graphical_abstract_preview.png')
