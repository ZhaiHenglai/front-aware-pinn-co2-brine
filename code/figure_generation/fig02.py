import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "dejavuserif", "font.size": 8.6,
    "axes.linewidth": 0.9, "savefig.dpi": 300,
})

C_W = "#2e6fdb"    # brine / water
C_C = "#c8102e"    # CO2
C_F = "#1a1a1a"    # fractional flow
C_T = "#0b7d63"    # Welge tangent

# --- physics from Table 1 ----------------------------------------------------
Sw_irr, Snr = 0.20, 0.0
nw = nc = 2.0
krw0 = krc0 = 1.0
mu_w, mu_c = 2.5e-4, 2.25e-5
Sc_max = 1.0 - Sw_irr                      # 0.8, maximum mobile CO2 saturation
M = mu_w / mu_c                            # end-point mobility ratio ~ 11.1

def relperm(Sc):
    Se_c = np.clip((Sc - Snr) / (1 - Sw_irr - Snr), 0, 1)
    Se_w = np.clip(((1 - Sc) - Sw_irr) / (1 - Sw_irr - Snr), 0, 1)
    return krw0 * Se_w**nw, krc0 * Se_c**nc

def frac_flow(Sc):
    krw, krc = relperm(Sc)
    lam_w = krw / mu_w
    lam_c = krc / mu_c
    return np.divide(lam_c, lam_c + lam_w, out=np.zeros_like(Sc),
                     where=(lam_c + lam_w) > 0)

Sc = np.linspace(0, Sc_max, 801)
krw, krc = relperm(Sc)
fc = frac_flow(Sc)

# Welge tangent from the initial state S_c = 0: tangent point maximises f_c/S_c.
inner = (Sc > 1e-6) & (Sc < Sc_max - 1e-9)
slope = np.where(inner, fc / np.where(Sc == 0, np.nan, Sc), -np.inf)
Sstar = Sc_max / np.sqrt(1 + M)
fstar = float(frac_flow(np.array([Sstar]))[0])
assert abs(Sstar-0.229878)<1e-5

fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.2, 3.1))
fig.subplots_adjust(left=0.085, right=0.985, bottom=0.155, top=0.88, wspace=0.30)

# ---- (a) relative permeabilities -------------------------------------------
axA.plot(Sc, krw, color=C_W, lw=2.0, label="$k_{rw}$ (brine)")
axA.plot(Sc, krc, color=C_C, lw=2.0, label="$k_{rc}$ (CO$_2$)")
axA.set_xlim(0, Sc_max); axA.set_ylim(0, 1.02)
axA.set_xlabel("CO$_2$ saturation  $S_{\\mathrm{CO_2}}$", labelpad=2)
axA.set_ylabel("relative permeability  $k_{r\\alpha}$", labelpad=3)
axA.set_title("(a)  Brooks–Corey relative permeability", loc="left",
              fontsize=9.2, pad=5)
axA.legend(loc="center", fontsize=8.0, frameon=False)
axA.text(0.5, 0.06, "$n_w=n_c=2,\\;\\; S_{w,\\mathrm{irr}}=0.2$",
         transform=axA.transAxes, ha="center", fontsize=7.6, color="#555555")
axA.spines["top"].set_visible(False); axA.spines["right"].set_visible(False)

# ---- (b) fractional flow + Welge tangent -----------------------------------
axB.plot(Sc, fc, color=C_F, lw=2.0, zorder=4, label="$f_c(S_{\\mathrm{CO_2}})$")
# tangent line from (0,0) through the tangent point, extended across the axis
xt = np.array([0.0, Sc_max])
axB.plot(xt, (fstar / Sstar) * xt, color=C_T, lw=1.5, ls="--", zorder=3,
         label="Welge tangent")
axB.scatter([Sstar], [fstar], s=34, color=C_T, zorder=6, ec="white", lw=0.7)
axB.plot([Sstar, Sstar], [0, fstar], color=C_T, lw=0.8, ls=":", zorder=2)
axB.annotate(f"shock\n$S^*\\!\\approx{Sstar:.2f}$", xy=(Sstar, fstar),
             xytext=(Sstar + 0.14, fstar - 0.28), fontsize=7.6, color=C_T,
             ha="center", arrowprops=dict(arrowstyle="->", lw=0.8, color=C_T))
axB.set_xlim(0, Sc_max); axB.set_ylim(0, 1.02)
axB.set_xlabel("CO$_2$ saturation  $S_{\\mathrm{CO_2}}$", labelpad=2)
axB.set_ylabel("CO$_2$ fractional flow  $f_c$", labelpad=3)
axB.set_title("(b)  Fractional flow and Welge shock", loc="left",
              fontsize=9.2, pad=5)
axB.legend(loc="lower right", fontsize=8.0, frameon=False)
axB.text(0.04, 0.92, f"$M=\\mu_w/\\mu_c\\approx{M:.0f}$\n(non-convex flux)",
         transform=axB.transAxes, ha="left", va="top", fontsize=7.6,
         color="#555555")
axB.spines["top"].set_visible(False); axB.spines["right"].set_visible(False)

from shared import save, SRC
np.savetxt(SRC/'Figure2_curves.csv',np.column_stack([Sc,krw,krc,fc]),delimiter=',',header='S,krw,krc,fc',comments='')
save(fig,'Figure2')

