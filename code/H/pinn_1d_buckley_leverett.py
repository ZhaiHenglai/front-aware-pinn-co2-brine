"""
1D Buckley-Leverett validation case (analytical Welge ground truth + PINN).

Purpose (paper §4.1 / Table 3): a clean, analytically verifiable testbed that
(i) confronts the Fuks-Tchelepi shock-smearing failure against an EXACT solution,
(ii) discriminates point-wise supervision (M4) vs the label-efficient pairwise
front-gradient (M5) under SPARSE labels, and (iii) tests the 1-D analogues of
RAR (M6_RAR) and finite-volume conservation (M7_RAR_FV).

Shares the 2D physics: Brooks-Corey n=2, Sw_irr=0.2, Snr=0, mu_w/mu_c ~ 11.
Self-contained; needs only numpy + torch.

Env controls (all optional):
  BL_EXP        FREE | DIFF | M4 | M5 | M6_RAR | M7_RAR_FV  (default M5)
  BL_SEED       int                              (default 0)
  BL_N_LABELS   # sparse interior labels         (default 60; 0 => data-free)
  BL_ITERS      Adam iterations                  (default 20000)
  BL_FOURIER    0|1  multi-scale Fourier on (x,t) (default 1)
  BL_DIFF_EPS   artificial-diffusion coeff for DIFF (default 1e-3)
  BL_RAR_POINTS residual-adaptive points for M6/M7 (default 512)
  BL_FV_CELLS    finite-volume cells per step for M7_RAR_FV (default 512)
  BL_OUTDIR     output dir                        (default ./BL_results)
  BL_DEVICE     cpu|cuda                          (default cuda if available)

Outputs (per run, in BL_OUTDIR/<run_id>/):
  metrics.csv         shock-position err, breakthrough err, W_front, front-band RMSE, global relL2
  profile_final.csv   x, S_pred, S_analytic at t=T
  profile_final.png|pdf|tiff|svg   predicted vs analytical front
"""
import os, math, json
import numpy as np
import torch
import torch.nn as nn

torch.set_default_dtype(torch.float64)

# ----------------------------- config ---------------------------------------
EXP_RAW    = os.environ.get("BL_EXP", "M5").upper()
EXP        = {"M6": "M6_RAR", "M7": "M7_RAR_FV"}.get(EXP_RAW, EXP_RAW)
VALID_EXPS = {"FREE", "DIFF", "M4", "M5", "M6_RAR", "M7_RAR_FV"}
if EXP not in VALID_EXPS:
    raise SystemExit(f"ERROR: unsupported BL_EXP={EXP_RAW}; expected one of {sorted(VALID_EXPS)}")
SEED       = int(os.environ.get("BL_SEED", "0"))
N_LABELS   = int(os.environ.get("BL_N_LABELS", "60"))
ITERS      = int(os.environ.get("BL_ITERS", "20000"))
USE_FOURIER= os.environ.get("BL_FOURIER", "1") == "1"
DIFF_EPS   = float(os.environ.get("BL_DIFF_EPS", "1e-3"))
RAR_POINTS = int(os.environ.get("BL_RAR_POINTS", "512"))
RAR_CANDIDATES = int(os.environ.get("BL_RAR_CANDIDATES", "4096"))
RAR_START  = int(os.environ.get("BL_RAR_START", "6000"))
RAR_EVERY  = int(os.environ.get("BL_RAR_EVERY", "1000"))
RAR_WEIGHT = float(os.environ.get("BL_RAR_WEIGHT", "1.0"))
FV_CELLS   = int(os.environ.get("BL_FV_CELLS", "512"))
FV_START   = int(os.environ.get("BL_FV_START", "6000"))
FV_RAMP    = int(os.environ.get("BL_FV_RAMP", "4000"))
FV_WEIGHT  = float(os.environ.get("BL_FV_WEIGHT", "2.0"))
FV_DX_MIN  = float(os.environ.get("BL_FV_DX_MIN", "0.005"))
FV_DX_MAX  = float(os.environ.get("BL_FV_DX_MAX", "0.050"))
FV_DT_MIN_FRAC = float(os.environ.get("BL_FV_DT_MIN_FRAC", "0.002"))
FV_DT_MAX_FRAC = float(os.environ.get("BL_FV_DT_MAX_FRAC", "0.040"))
OUTDIR     = os.environ.get("BL_OUTDIR", "./BL_results")
DEVICE     = os.environ.get("BL_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
RUN_ID     = f"BL_{EXP}_seed{SEED}_N{N_LABELS}_F{int(USE_FOURIER)}"

np.random.seed(SEED); torch.manual_seed(SEED)

# ----------------------------- physics (shared with 2D) ----------------------
Sw_irr, Snr = 0.20, 0.0
nw = nc = 2.0
krw0 = krc0 = 1.0
mu_w, mu_c = 2.5e-4, 2.25e-5
phi   = 0.2
u_t   = 1.0                      # nondim total Darcy velocity
A_SPEED = u_t / phi              # characteristic factor in S_t + A f(S)_x = 0
S_INJ = 1.0 - Sw_irr             # 0.8 injected CO2 saturation
S_EPS = 1.0e-6
L = 1.0                          # nondim domain length

def _krw_krc_np(Sc):
    Se_c = np.clip((Sc - Snr)/(1 - Sw_irr - Snr), 0, 1)
    Se_w = np.clip(((1 - Sc) - Sw_irr)/(1 - Sw_irr - Snr), 0, 1)
    return krw0*Se_w**nw, krc0*Se_c**nc

def frac_flow_np(Sc):
    krw, krc = _krw_krc_np(Sc)
    lam_w = krw/mu_w; lam_c = krc/mu_c
    return np.divide(lam_c, lam_c+lam_w, out=np.zeros_like(Sc), where=(lam_c+lam_w)>0)

# ----------------------------- analytical Welge solution ---------------------
_Sg = np.linspace(0, S_INJ, 200001)
_fg = frac_flow_np(_Sg)
_fpg = np.gradient(_fg, _Sg)                      # f'(S)
# Welge tangent from S=0: S* maximises f(S)/S
_with = _Sg > 1e-9
_ratio = np.where(_with, _fg/np.where(_Sg==0, np.nan, _Sg), -np.inf)
_istar = int(np.nanargmax(_ratio))
S_STAR = _Sg[_istar]; F_STAR = _fg[_istar]
V_SHOCK = A_SPEED * F_STAR / S_STAR               # shock speed in x
T_BT = L / V_SHOCK                                # breakthrough time at x=L
T_END = 1.15 * T_BT                               # train/eval slightly past breakthrough (so E_bt is observable)
T_MID = 0.50 * T_BT                               # front mid-domain: used for the sharpness profile / W_front

# rarefaction branch S in [S*, S_inj]: characteristic speed c(S)=A f'(S), monotone there
_rar_S = np.linspace(S_STAR, S_INJ, 4000)
_rar_c = A_SPEED * np.interp(_rar_S, _Sg, _fpg)
_order = np.argsort(_rar_c)
_rar_c_sorted = _rar_c[_order]; _rar_S_sorted = _rar_S[_order]
_c_inj = A_SPEED * float(np.interp(S_INJ, _Sg, _fpg))

def analytic_S(x, t):
    """Exact BL saturation at (x,t)>0 (arrays)."""
    x = np.atleast_1d(x).astype(float); t = np.atleast_1d(t).astype(float)
    xi = np.where(t > 0, x/np.maximum(t, 1e-30), np.inf)
    S = np.zeros_like(xi)
    S[xi >= V_SHOCK] = 0.0                                   # ahead of shock
    rar = (xi < V_SHOCK) & (xi >= _c_inj)                    # rarefaction fan
    S[rar] = np.interp(xi[rar], _rar_c_sorted, _rar_S_sorted)
    S[xi < _c_inj] = S_INJ                                   # injected plateau
    return S

def shock_position(t):     # exact front location
    return V_SHOCK * t

print(f"[Analytic] S*={S_STAR:.4f}  f(S*)={F_STAR:.4f}  v_shock={V_SHOCK:.4f}  "
      f"t_bt={T_BT:.4f}  T_end={T_END:.4f}  M=mu_w/mu_c={mu_w/mu_c:.2f}")

# ----------------------------- PINN ------------------------------------------
class MSFourier(nn.Module):
    def __init__(self, in_dim=2, m=16, scales=(1,2,4,8,16), base=3.0):
        super().__init__()
        B = torch.cat([torch.randn(in_dim, m)*(base*s) for s in scales], dim=1)
        self.register_buffer("B", B)
        self.out_dim = 2*B.shape[1]
    def forward(self, z):
        p = z @ self.B
        return torch.cat([torch.sin(p), torch.cos(p)], dim=-1)

class BLNet(nn.Module):
    def __init__(self, width=64, depth=6, fourier=True, beta=8.0):
        super().__init__()
        self.ff = MSFourier(2) if fourier else None
        din = self.ff.out_dim if fourier else 2
        layers = [nn.Linear(din, width), nn.Tanh()]
        for _ in range(depth-1): layers += [nn.Linear(width, width), nn.Tanh()]
        layers += [nn.Linear(width, 1)]
        self.net = nn.Sequential(*layers); self.beta = beta
    def forward(self, x, t):
        z = torch.cat([2*x/L-1, 2*t/T_END-1], dim=1)
        z = self.ff(z) if self.ff is not None else z
        logit = self.net(z)
        return S_EPS + (S_INJ - S_EPS)*torch.sigmoid(self.beta*logit)

def grad(u, v):
    return torch.autograd.grad(u, v, torch.ones_like(u), create_graph=True)[0]

def frac_flow_t(Sc):
    Se_c = torch.clamp((Sc-Snr)/(1-Sw_irr-Snr), 0, 1)
    Se_w = torch.clamp(((1-Sc)-Sw_irr)/(1-Sw_irr-Snr), 0, 1)
    lam_c = (krc0*Se_c**nc)/mu_c; lam_w = (krw0*Se_w**nw)/mu_w
    return lam_c/(lam_c+lam_w+1e-30)

def pde_residual(model, x, t, eps=0.0):
    S = model(x, t)
    f = frac_flow_t(S)
    r = grad(S, t) + A_SPEED*grad(f, x)
    if eps > 0: r = r - eps*grad(grad(S, x), x)     # artificial diffusion (DIFF)
    return r

def sample_collocation(n):
    x = torch.rand(n, 1, device=DEVICE) * L
    t = torch.rand(n, 1, device=DEVICE) * T_END
    return x.requires_grad_(True), t.requires_grad_(True)

def sample_rar_points(model, n_points, n_candidates, eps=0.0):
    n_candidates = max(int(n_candidates), int(n_points))
    x, t = sample_collocation(n_candidates)
    r = pde_residual(model, x, t, eps).detach().abs().flatten()
    k = min(int(n_points), int(r.numel()))
    idx = torch.topk(r, k=k, largest=True).indices
    return x.detach()[idx], t.detach()[idx]

def fv_conservation_residual(model, n_cells):
    dx = torch.empty(n_cells, 1, device=DEVICE).uniform_(FV_DX_MIN, FV_DX_MAX)
    dt = torch.empty(n_cells, 1, device=DEVICE).uniform_(
        FV_DT_MIN_FRAC * T_END, FV_DT_MAX_FRAC * T_END
    )
    x0 = torch.rand(n_cells, 1, device=DEVICE) * (L - dx)
    t0 = torch.rand(n_cells, 1, device=DEVICE) * (T_END - dt)
    x1, t1 = x0 + dx, t0 + dt
    xc, tc = 0.5 * (x0 + x1), 0.5 * (t0 + t1)

    storage = dx * (model(xc, t1) - model(xc, t0))
    flux = A_SPEED * dt * (frac_flow_t(model(x1, tc)) - frac_flow_t(model(x0, tc)))
    scale = dx * S_INJ + A_SPEED * dt + 1.0e-12
    return (storage + flux) / scale

# sparse labels from the analytical solution
FRONT_LO, FRONT_HI, FRONT_C, FRONT_SIG = 0.05, 0.30, 0.175, 0.075
def make_labels(n, seed):
    rng = np.random.default_rng(seed)
    xs = rng.uniform(0.02, L, n); ts = rng.uniform(0.05*T_END, T_END, n)
    Ss = analytic_S(xs, ts)
    return (torch.tensor(xs).view(-1,1).to(DEVICE),
            torch.tensor(ts).view(-1,1).to(DEVICE),
            torch.tensor(Ss).view(-1,1).to(DEVICE))

def main():
    use_data = EXP in ("M4", "M5", "M6_RAR", "M7_RAR_FV") and N_LABELS > 0
    use_pair = EXP in ("M5", "M6_RAR", "M7_RAR_FV")
    use_rar = EXP in ("M6_RAR", "M7_RAR_FV") and RAR_POINTS > 0
    use_fv = EXP == "M7_RAR_FV" and FV_CELLS > 0
    eps_diff = DIFF_EPS if EXP == "DIFF" else 0.0
    model = BLNet(fourier=USE_FOURIER).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=2e-4)
    rar_x, rar_t = None, None
    print(f"[Config] exp={EXP} seed={SEED} labels={N_LABELS} fourier={int(USE_FOURIER)} "
          f"data={int(use_data)} pair={int(use_pair)} rar={int(use_rar)} fv={int(use_fv)}")
    if use_data:
        xd, td, Sd = make_labels(N_LABELS, SEED)
        wf = torch.exp(-((Sd-FRONT_C)**2)/(2*FRONT_SIG**2))   # front-band weight
        plume = (Sd > FRONT_LO).double()

    for it in range(1, ITERS+1):
        opt.zero_grad(set_to_none=True)
        # collocation
        xc, tc = sample_collocation(4000)
        w_pde = 0.0 if it < 2000 else min(1.0, (it-2000)/4000.0)
        loss = w_pde * (pde_residual(model, xc, tc, eps_diff)**2).mean()
        if use_rar and it >= RAR_START and (rar_x is None or (it - RAR_START) % max(RAR_EVERY, 1) == 0):
            rar_x, rar_t = sample_rar_points(model, RAR_POINTS, RAR_CANDIDATES, eps_diff)
            print(f"[{RUN_ID}] RAR refresh it={it} points={rar_x.shape[0]}")
        if use_rar and rar_x is not None:
            xr = rar_x.detach().clone().requires_grad_(True)
            tr = rar_t.detach().clone().requires_grad_(True)
            loss = loss + w_pde * RAR_WEIGHT * (pde_residual(model, xr, tr, eps_diff)**2).mean()
        # IC  S(x,0)=0
        xi = torch.rand(800,1, device=DEVICE)*L
        loss = loss + 5.0*(model(xi, torch.zeros_like(xi))**2).mean()
        # BC  S(0,t)=S_inj
        tb = torch.rand(800,1, device=DEVICE)*T_END
        loss = loss + 5.0*((model(torch.zeros_like(tb), tb)-S_INJ)**2).mean()
        # Sparse point labels, pairwise front-gradient, RAR, and FV are enabled
        # progressively to preserve the causal chain FREE -> DIFF -> M4 -> M7.
        if use_data:
            Sp = model(xd, td); e2 = (Sp-Sd)**2
            loss = loss + 10.0*(e2.mean()
                                + (wf*e2).sum()/(wf.sum()+1e-12)
                                + (plume*e2).sum()/(plume.sum()+1e-12))
            if use_pair and it > 4000:
                # pairs of labels close in x at same-ish t, constrain local slope
                idx = torch.randperm(xd.shape[0], device=DEVICE)
                xi2, ti2, Si2 = xd[idx], td[idx], Sd[idx]
                d = (xd - xi2); m = (d.abs() > 1e-3).squeeze()
                if m.any():
                    gp = (model(xd[m], td[m]) - model(xi2[m], ti2[m]))/d[m]
                    gt = (Sd[m] - Si2[m])/d[m]
                    wpg = min(1.0, (it-4000)/4000.0)*0.03
                    loss = loss + 10.0*wpg*(((gp-gt)/30.0)**2).mean()
        if use_fv and it >= FV_START:
            wfv = min(1.0, max(0, it - FV_START) / max(FV_RAMP, 1)) * FV_WEIGHT
            if wfv > 0:
                loss = loss + wfv * (fv_conservation_residual(model, FV_CELLS)**2).mean()
        loss.backward(); opt.step()
        if it % 2000 == 0:
            print(f"[{RUN_ID}] it={it:6d} loss={loss.item():.3e}")

    # --------------------------- evaluation vs analytical --------------------
    model.eval()
    xg = np.linspace(0, L, 400); tg = np.linspace(0.05*T_END, T_END, 60)
    XX, TT = np.meshgrid(xg, tg)
    with torch.no_grad():
        Sp = model(torch.tensor(XX.reshape(-1,1)).to(DEVICE),
                   torch.tensor(TT.reshape(-1,1)).to(DEVICE)).cpu().numpy().reshape(XX.shape)
    Sa = analytic_S(XX.ravel(), TT.ravel()).reshape(XX.shape)
    relL2 = np.linalg.norm(Sp-Sa)/ (np.linalg.norm(Sa)+1e-30)
    fb = (Sa>=FRONT_LO)&(Sa<=FRONT_HI)
    front_rmse = float(np.sqrt(np.mean((Sp[fb]-Sa[fb])**2))) if fb.any() else float("nan")
    # shock-position error (S=S*/2 contour) at each t, vs exact V_SHOCK*t
    def contour_x(Srow, level):
        i = np.argmin(np.abs(Srow-level)); return xg[i]
    inside = tg < 0.95*T_BT     # only where the shock is still inside the domain
    xf_pred = np.array([contour_x(Sp[j], S_STAR*0.5) for j in range(len(tg))])
    xf_true = shock_position(tg)
    e_front = float(np.sqrt(np.mean((xf_pred[inside]-xf_true[inside])**2)))
    # breakthrough at x=L*0.95: time S exceeds 0.05
    xbt = int(np.argmin(np.abs(xg-0.95*L)))
    sp_col = Sp[:, xbt]
    above = np.where(sp_col > 0.05)[0]
    t_bt_pred = tg[above[0]] if len(above) else float("nan")
    t_bt_true = (0.95*L)/V_SHOCK
    e_bt = abs(t_bt_pred - t_bt_true) if t_bt_pred==t_bt_pred else float("nan")
    # front thickness with the shock mid-domain (10-90% of local jump); true shock => 0
    jmid = int(np.argmin(np.abs(tg - T_MID)))
    Srow = Sp[jmid]
    s10, s90 = 0.1*S_STAR, 0.9*S_STAR
    try:
        x10 = xg[np.argmin(np.abs(Srow-s10))]; x90 = xg[np.argmin(np.abs(Srow-s90))]
        W_front = abs(x10-x90)
    except Exception:
        W_front = float("nan")

    outdir = os.path.join(OUTDIR, RUN_ID); os.makedirs(outdir, exist_ok=True)
    import csv
    with open(os.path.join(outdir, "metrics.csv"), "w", newline="") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(["run_id","exp","seed","n_labels","fourier","use_rar","use_fv","rar_points","fv_cells",
                    "relL2_S","front_band_RMSE",
                    "E_front_shockpos","E_bt","W_front_pred","S_star","v_shock","t_bt_true"])
        w.writerow([RUN_ID,EXP,SEED,N_LABELS,int(USE_FOURIER),int(use_rar),int(use_fv),RAR_POINTS,FV_CELLS,
                    f"{relL2:.6f}",f"{front_rmse:.6f}",
                    f"{e_front:.6f}",f"{e_bt:.6f}",f"{W_front:.6f}",f"{S_STAR:.6f}",
                    f"{V_SHOCK:.6f}",f"{t_bt_true:.6f}"])
    # final-time profile
    np.savetxt(os.path.join(outdir, "profile_final.csv"),
               np.column_stack([xg, Sp[jmid], Sa[jmid]]), delimiter=",",
               header="x,S_pred,S_analytic", comments="")
    try:
        import sys
        from pathlib import Path
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

        style_dir = Path(__file__).resolve().parent / "Results"
        if style_dir.exists():
            sys.path.insert(0, str(style_dir))
        try:
            from _nature_plot_style_M0_M7 import (
                setup_nature_rcparams,
                save_pub_figure,
                apply_nature_axis,
                color_for_label,
            )
            setup_nature_rcparams(font_size=7.0, line_width=0.9)
            use_pub_style = True
        except Exception:
            use_pub_style = False

        fig, ax = plt.subplots(figsize=(3.55, 2.35), constrained_layout=False)
        ax.plot(xg, Sa[jmid], color="black", lw=1.2, label="analytic (Welge)")
        pinn_color = color_for_label(EXP, "#B45A4A") if use_pub_style else "tab:red"
        ax.plot(xg, Sp[jmid], color=pinn_color, ls="--", lw=1.2, label=f"PINN ({EXP})")
        ax.axvline(shock_position(tg[jmid]), color="0.55", ls=":", lw=0.8)
        ax.set_xlabel("x")
        ax.set_ylabel("$S_{CO2}$")
        ax.set_ylim(0, 0.85)
        ax.set_title(f"1D BL  t={tg[jmid]:.4f}  E_front={e_front:.3f}  W_front={W_front:.3f}")
        ax.legend(frameon=False)
        if use_pub_style:
            apply_nature_axis(ax, grid=True, grid_axis="both")
            save_pub_figure(
                fig,
                os.path.join(outdir, "profile_final"),
                dpi=600,
                export_pdf=True,
                export_tiff=True,
                export_svg=True,
                export_png=True,
                pad_inches=0.03,
            )
        else:
            fig.tight_layout()
            fig.savefig(os.path.join(outdir, "profile_final.png"), dpi=200, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        print("plot skipped:", e)
    print(f"[DONE {RUN_ID}] relL2={relL2:.4f} front_RMSE={front_rmse:.4f} "
          f"E_front={e_front:.4f} E_bt={e_bt:.4f} W_front={W_front:.4f} -> {outdir}")

if __name__ == "__main__":
    main()
