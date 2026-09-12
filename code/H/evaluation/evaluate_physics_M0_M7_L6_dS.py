#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Physics-consistency evaluator for M0-M7 PINN checkpoints and M7 leave-one-out runs.

Outputs
-------
  eval_M0_M7/physics/summary_physics_consistency.csv
  eval_M0_M7/physics/pde_metrics.csv
  eval_M0_M7/physics/bc_ic_metrics.csv
  eval_M0_M7/physics/mass_metrics.csv
  eval_M0_M7/physics/local_fv_metrics.csv
  eval_M0_M7/physics/physics_evaluation_config.json
  eval_M0_M7/physics/figures/*.png|*.pdf|*.tiff

Scope
-----
This script evaluates physical consistency only:
  - strong-form PDE residuals, including high-percentile/tail residuals;
  - initial-condition, injection, outlet-pressure, outlet-backflow, and no-flow BC metrics;
  - global CO2/water/total mass diagnostics over all data time layers;
  - local finite-volume CO2/water/total residual diagnostics.

It intentionally does not compute global fitting metrics, front/plume support metrics, or
snapshot fields. Those belong to the global/front/plume/snapshot evaluators.

Design notes
------------
1. Time-layer selection is aligned with evaluate_snapshot_gridviz_M0_M7_publication_snapshots.py:
   t0, early, middle, late, final.
2. Sco2 initial saturation defaults to S_IC_CO2=0.0. IC metrics are evaluated against
   this value, not zero.
3. CO2 residual saturation Snr defaults to 0.0 and is used only in relative-permeability
   physics; raw Sco2 is not remapped.
4. Figure labels are short: M0-M7 for forward ablation, and L0-L6 for leave-one-out.
   Long checkpoint names are not used in legends.

Dependency
----------
This script imports the model/checkpoint/dataset utilities from
`evaluate_global_accuracy_M0_M7_L6_dS_shortlabels.py`. Keep both scripts in the
same directory, or add that location to PYTHONPATH.
"""
from __future__ import annotations

import os
import csv
import json
import math
import argparse
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from _nature_plot_style_M0_M7 import (
        setup_nature_rcparams,
        save_pub_figure,
        color_for_label,
        apply_nature_axis,
    )
except Exception:  # pragma: no cover
    def setup_nature_rcparams(font_size: float = 7.0, line_width: float = 1.15):
        plt.rcParams.update({"font.size": font_size, "lines.linewidth": line_width})
    def save_pub_figure(fig, base_without_ext, dpi=600, export_pdf=True, export_tiff=True, export_svg=True, export_png=True, pad_inches=0.035):
        if export_png:
            fig.savefig(str(base_without_ext) + ".png", dpi=dpi, bbox_inches="tight", pad_inches=pad_inches)
        if export_pdf:
            fig.savefig(str(base_without_ext) + ".pdf", bbox_inches="tight", pad_inches=pad_inches)
        if export_tiff:
            fig.savefig(str(base_without_ext) + ".tiff", dpi=dpi, bbox_inches="tight", pad_inches=pad_inches)
    def color_for_label(label, fallback=None):
        return fallback
    def apply_nature_axis(ax, grid: bool = True, grid_axis: str = "y"):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if grid:
            ax.grid(True, axis=grid_axis, linewidth=0.25, alpha=0.35)

from evaluate_global_accuracy_M0_M7_L6_dS_shortlabels import (  # type: ignore
    ALL_EXPS,
    DEFAULT_DATA_PATH,
    DEFAULT_CKPT_DIR,
    DEFAULT_DEVICE,
    DEFAULT_GPU_ID,
    DEFAULT_DTYPE,
    DEFAULT_INFER_BATCH_SIZE,
    DEFAULT_RANDOM_SEED,
    DEFAULT_MAX_TIME_STEPS,
    DEFAULT_MAX_SORT_N,
    DEFAULT_TIME_INDICES,
    DEFAULT_TIME_UNIT,
    DEFAULT_S_IC_CO2,
    DEFAULT_SNR,
    DEFAULT_L_REF,
    DEFAULT_T_REF,
    DEFAULT_MU_REF,
    DEFAULT_U_REF,
    DEFAULT_K_REF,
    DEFAULT_P0,
    ensure_dir,
    to_np_1d,
    dtype_from_cfg,
    save_json,
    load_checkpoint,
    default_cfg,
    apply_exp_preset,
    cfg_update_from_ckpt,
    build_model,
    load_dataset_pack,
    make_run_specs,
    build_eval_plan,
    predict_model,
    write_csv,
    time_to_display,
    time_axis_label,
)

try:
    from _time_series_scaling import apply_robust_time_axis, transient_cut_for_unit
except Exception:  # pragma: no cover
    def apply_robust_time_axis(ax, series, **kwargs):
        return {"applied": False}
    def transient_cut_for_unit(time_unit: str, cut_days: float = 0.01):
        return cut_days


# =============================================================================
# Defaults
# =============================================================================
DEFAULT_OUT_ROOT = "./eval_M0_M7"
DEFAULT_FIGURE_DPI = 600
DEFAULT_EXPORT_PDF = True
DEFAULT_EXPORT_TIFF = True
DEFAULT_EXPORT_SVG = True
EXPORT_SVG = DEFAULT_EXPORT_SVG

# Physical defaults. Checkpoint physical_parameters override these values.
DEFAULT_K = 1.0e-14
DEFAULT_PHI = 0.2
DEFAULT_RHO_W_CONST = 1027.61
DEFAULT_MU_W = 2.5e-4
DEFAULT_MU_C = 2.25e-5
DEFAULT_SW_IRR = 0.2
DEFAULT_KRW0 = 1.0
DEFAULT_KRC0 = 1.0
DEFAULT_NW = 2.0
DEFAULT_NC = 2.0
DEFAULT_P_OUT = 10.0e6
DEFAULT_R_WELL = 0.5
DEFAULT_INJ_CENTER = (0.0, 0.0)
DEFAULT_OUT_CENTER = (5.0, 5.0)
DEFAULT_T_IC_GATE = 1.0e-2
DEFAULT_INJ_BC_T_START_TILDE = 10.0 * DEFAULT_T_IC_GATE
DEFAULT_GEOM_EPS = 1.0e-6

# Evaluation sampling defaults.
DEFAULT_PDE_POINTS = 4096
DEFAULT_PDE_BATCH_SIZE = 512
DEFAULT_BC_POINTS = 2048
DEFAULT_OUTER_POINTS_EACH = 512
DEFAULT_IC_POINTS = 4096
DEFAULT_MASS_SAMPLE_PER_TIME = 30000
DEFAULT_FV_N_CELLS_EVAL = 2048
DEFAULT_FV_BATCH_SIZE = 256
DEFAULT_FV_H_TILDE = 1.0 / 64.0
DEFAULT_FV_DT_TILDE = None
DEFAULT_EOS_FIT_SAMPLE = 200000
DEFAULT_EOS_DEGREE = 16

# Region definitions for physics diagnostics.
DEFAULT_FRONT_LOW = 0.05
DEFAULT_FRONT_HIGH = 0.30
DEFAULT_BACKGROUND_THRESHOLD = 0.02
DEFAULT_NEAR_INJECTION_RADIUS_M = 1.0


# =============================================================================
# Small utilities
# =============================================================================
def parse_float_list(s: str) -> list[float]:
    return [float(x.strip()) for x in str(s).split(",") if x.strip()]


def short_label(spec, loo_labels: bool = False) -> str:
    exp = str(getattr(spec, "exp_name", "")).upper()
    leave = str(getattr(spec, "leave_out", "NONE")).upper()
    loo_map = {
        "NONE": "L0",
        "PLAIN_TWONET": "L1",
        "FRONT_PLUME": "L2",
        "PAIRGRAD": "L3",
        "RAR": "L4",
        "FV": "L5",
        "COARSE_DETAIL": "L6",
        "WATER_FV": "LFVw",
        "MSFF": "Lmsff",
    }
    if leave != "NONE":
        return loo_map.get(leave, leave)
    if loo_labels and exp == "M7":
        return "L0"
    return exp


def metric_values(values: np.ndarray, prefix: str) -> dict[str, float]:
    a = np.asarray(values, dtype=np.float64).reshape(-1)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {
            f"{prefix}_RMSE": np.nan, f"{prefix}_MAE": np.nan, f"{prefix}_mean": np.nan,
            f"{prefix}_p50": np.nan, f"{prefix}_p90": np.nan, f"{prefix}_p95": np.nan,
            f"{prefix}_p99": np.nan, f"{prefix}_max": np.nan, f"{prefix}_n": 0,
        }
    aa = np.abs(a)
    return {
        f"{prefix}_RMSE": float(math.sqrt(np.mean(a * a))),
        f"{prefix}_MAE": float(np.mean(aa)),
        f"{prefix}_mean": float(np.mean(a)),
        f"{prefix}_p50": float(np.percentile(aa, 50)),
        f"{prefix}_p90": float(np.percentile(aa, 90)),
        f"{prefix}_p95": float(np.percentile(aa, 95)),
        f"{prefix}_p99": float(np.percentile(aa, 99)),
        f"{prefix}_max": float(np.max(aa)),
        f"{prefix}_n": int(a.size),
    }


def safe_rel_error(pred: np.ndarray, true: np.ndarray, eps: float = 1.0e-30) -> np.ndarray:
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    return np.abs(pred - true) / (np.abs(true) + eps)


def set_journal_style():
    setup_nature_rcparams(font_size=7.0, line_width=1.2)


def _savefig(fig, base: str, dpi: int, export_pdf: bool, export_tiff: bool):
    save_pub_figure(fig, base, dpi=dpi, export_pdf=export_pdf, export_tiff=export_tiff, export_svg=EXPORT_SVG, export_png=True, pad_inches=0.035)


def physics_cfg_update_from_ckpt(cfg: dict[str, Any], ckpt_cfg: dict[str, Any] | None):
    """Add physics keys not required by the global evaluator."""
    cfg.setdefault("K", DEFAULT_K)
    cfg.setdefault("phi", DEFAULT_PHI)
    cfg.setdefault("rho_w_const", DEFAULT_RHO_W_CONST)
    cfg.setdefault("mu_w", DEFAULT_MU_W)
    cfg.setdefault("mu_c", DEFAULT_MU_C)
    cfg.setdefault("p_out", DEFAULT_P_OUT)
    cfg.setdefault("Sw_irr", DEFAULT_SW_IRR)
    cfg.setdefault("Snr", DEFAULT_SNR)
    cfg.setdefault("krw0", DEFAULT_KRW0)
    cfg.setdefault("krc0", DEFAULT_KRC0)
    cfg.setdefault("nw", DEFAULT_NW)
    cfg.setdefault("nc", DEFAULT_NC)
    cfg.setdefault("r_well", DEFAULT_R_WELL)
    cfg.setdefault("inj_center", DEFAULT_INJ_CENTER)
    cfg.setdefault("out_center", DEFAULT_OUT_CENTER)
    cfg.setdefault("inj_bc_t_start_tilde", DEFAULT_INJ_BC_T_START_TILDE)
    if not isinstance(ckpt_cfg, dict):
        return
    phys = ckpt_cfg.get("physical_parameters", {})
    if isinstance(phys, dict):
        mp = {
            "K": "K", "phi": "phi", "rho_w_const": "rho_w_const", "mu_w": "mu_w", "mu_c": "mu_c",
            "p_out": "p_out", "P_ref": "P_ref_override", "A_time": "A_time",
            "Sw_irr": "Sw_irr", "Snr": "Snr", "krw0": "krw0", "krc0": "krc0", "nw": "nw", "nc": "nc",
            "r_well": "r_well", "inj_center": "inj_center", "out_center": "out_center",
            "inj_bc_t_start_tilde": "inj_bc_t_start_tilde",
        }
        for src, dst in mp.items():
            if src in phys:
                cfg[dst] = phys[src]
    # Some training scripts save this at top level.
    for src, dst in {
        "K": "K", "phi": "phi", "rho_w_const": "rho_w_const", "mu_w": "mu_w", "mu_c": "mu_c",
        "p_out": "p_out", "A_time": "A_time", "Sw_irr": "Sw_irr", "Snr": "Snr",
        "krw0": "krw0", "krc0": "krc0", "nw": "nw", "nc": "nc",
    }.items():
        if src in ckpt_cfg:
            cfg[dst] = ckpt_cfg[src]
    sat_ic = ckpt_cfg.get("saturation_ic", {})
    if isinstance(sat_ic, dict):
        for src, dst in {
            "s_ic_co2": "s_ic_co2",
            "s_inj_co2": "s_inj_co2",
            "s_co2_output_min": "S_eps",
            "s_co2_output_max": "S_CO2_MAX",
            "inj_bc_t_start_tilde": "inj_bc_t_start_tilde",
            "inj_bc_start_dt_multiplier": "inj_bc_start_dt_multiplier",
        }.items():
            if src in sat_ic:
                cfg[dst] = sat_ic[src]
    fv = ckpt_cfg.get("m7_fv_mass", {})
    if isinstance(fv, dict):
        for src, dst in {
            "fv_h_tilde": "fv_h_tilde",
            "fv_dt_tilde": "fv_dt_tilde",
            "fv_w_water": "fv_w_water",
            "front_center": "fv_front_center",
            "front_sigma": "fv_front_sigma",
            "front_weight_floor": "fv_front_weight_floor",
        }.items():
            if src in fv:
                cfg[dst] = fv[src]
    if "A_time" not in cfg:
        cfg["A_time"] = float(cfg["L_ref"]) / (float(cfg["U_ref"]) * float(cfg["T_ref"]))
    cfg["p_out_tilde"] = (float(cfg.get("p_out", DEFAULT_P_OUT)) - float(cfg["p0"])) / float(cfg["P_ref"])


# =============================================================================
# EOS: differentiable CO2 density model from dataset pack
# =============================================================================
class ChebRhoCO2(nn.Module):
    def __init__(self, coeffs: np.ndarray, rho_ref: float, pmin: float, pmax: float, dtype: torch.dtype):
        super().__init__()
        self.register_buffer("c", torch.tensor(coeffs, dtype=dtype))
        self.rho_ref = float(rho_ref)
        self.pmin = float(pmin)
        self.pmax = float(pmax)
        self.pmid = 0.5 * (self.pmin + self.pmax)
        self.prng = max(0.5 * (self.pmax - self.pmin), 1.0)

    def _chebval(self, z: torch.Tensor) -> torch.Tensor:
        c = self.c
        if c.numel() == 1:
            return c[0] + 0.0 * z
        b1 = torch.zeros_like(z)
        b2 = torch.zeros_like(z)
        for a in torch.flip(c[1:], dims=[0]):
            b0 = 2.0 * z * b1 - b2 + a
            b2 = b1
            b1 = b0
        return z * b1 - b2 + c[0]

    def forward(self, p_pa: torch.Tensor) -> torch.Tensor:
        z = (p_pa - self.pmid) / self.prng
        z = torch.clamp(z, -1.0 + 1.0e-12, 1.0 - 1.0e-12)
        return self.rho_ref * self._chebval(z)


def fit_rho_model_from_arrays(arrays: dict[str, Any], cfg: dict[str, Any], degree: int, sample: int, seed: int, dtype: torch.dtype, device: torch.device):
    if "rho_c" not in arrays:
        rho0 = 600.0
        coeffs = np.array([1.0], dtype=np.float64)
        return ChebRhoCO2(coeffs, rho0, float(cfg["p0"]) - 1.0e6, float(cfg["p0"]) + 1.0e6, dtype).to(device), rho0
    p = to_np_1d(arrays["p"]).astype(np.float64, copy=False)
    rho = to_np_1d(arrays["rho_c"]).astype(np.float64, copy=False)
    m = np.isfinite(p) & np.isfinite(rho) & (rho > 0)
    p = p[m]; rho = rho[m]
    if p.size == 0:
        rho0 = 600.0
        coeffs = np.array([1.0], dtype=np.float64)
        return ChebRhoCO2(coeffs, rho0, float(cfg["p0"]) - 1.0e6, float(cfg["p0"]) + 1.0e6, dtype).to(device), rho0
    rng = np.random.default_rng(seed)
    if sample > 0 and p.size > sample:
        idx = rng.choice(p.size, size=int(sample), replace=False)
        p = p[idx]; rho = rho[idx]
    order = np.argsort(p)
    p = p[order]; rho = rho[order]
    pmin = float(np.min(p)); pmax = float(np.max(p))
    pmid = 0.5 * (pmin + pmax)
    prng = max(0.5 * (pmax - pmin), 1.0)
    rho_ref = float(np.interp(pmid, p, rho))
    z = (p - pmid) / prng
    deg_eff = min(int(degree), max(0, int(np.unique(p).size) - 1))
    if deg_eff < 1:
        coeffs = np.array([1.0], dtype=np.float64)
    else:
        coeffs = np.polynomial.chebyshev.chebfit(z, rho / rho_ref, deg=deg_eff)
    return ChebRhoCO2(coeffs, rho_ref, pmin, pmax, dtype).to(device), rho_ref


# =============================================================================
# Autograd physics helpers
# =============================================================================
def grad(u, x, create_graph=True, retain_graph=True):
    return torch.autograd.grad(
        u, x,
        grad_outputs=torch.ones_like(u),
        create_graph=create_graph,
        retain_graph=retain_graph,
        allow_unused=False,
    )[0]


def relperm_from_Sco2(S: torch.Tensor, cfg: dict[str, Any]):
    Sw = 1.0 - S
    sw_irr = float(cfg.get("Sw_irr", DEFAULT_SW_IRR))
    snr = float(cfg.get("Snr", DEFAULT_SNR))
    denom = max(1.0 - sw_irr - snr, 1.0e-12)
    Se_w = torch.clamp((Sw - sw_irr) / denom, 0.0, 1.0)
    Se_c = torch.clamp((S - snr) / denom, 0.0, 1.0)
    krw = float(cfg.get("krw0", DEFAULT_KRW0)) * Se_w ** float(cfg.get("nw", DEFAULT_NW))
    krc = float(cfg.get("krc0", DEFAULT_KRC0)) * Se_c ** float(cfg.get("nc", DEFAULT_NC))
    return krw, krc


def phase_fluxes(model, x_t, y_t, t_t, cfg: dict[str, Any], rho_co2_model: nn.Module, rho_c_ref: float):
    p_tilde, S = model(x_t, y_t, t_t)
    p_phys = float(cfg["p0"]) + float(cfg["P_ref"]) * p_tilde
    rho_c = rho_co2_model(p_phys)
    rho_c_tilde = rho_c / float(rho_c_ref)
    rho_w_tilde = torch.ones_like(rho_c_tilde)
    krw, krc = relperm_from_Sco2(S, cfg)
    dpdx = grad(p_tilde, x_t)
    dpdy = grad(p_tilde, y_t)
    K_tilde = float(cfg.get("K", DEFAULT_K)) / float(cfg.get("k_ref", DEFAULT_K_REF))
    mu_ref = float(cfg.get("mu_ref", DEFAULT_MU_REF))
    vwx = -K_tilde * (mu_ref / float(cfg.get("mu_w", DEFAULT_MU_W))) * krw * dpdx
    vwy = -K_tilde * (mu_ref / float(cfg.get("mu_w", DEFAULT_MU_W))) * krw * dpdy
    vcx = -K_tilde * (mu_ref / float(cfg.get("mu_c", DEFAULT_MU_C))) * krc * dpdx
    vcy = -K_tilde * (mu_ref / float(cfg.get("mu_c", DEFAULT_MU_C))) * krc * dpdy
    return FPack(p_tilde, S, p_phys, rho_c_tilde, rho_w_tilde, vwx, vwy, vcx, vcy)


class FPack:
    __slots__ = ("p_tilde", "S", "p_phys", "rho_c_tilde", "rho_w_tilde", "vwx", "vwy", "vcx", "vcy")
    def __init__(self, p_tilde, S, p_phys, rho_c_tilde, rho_w_tilde, vwx, vwy, vcx, vcy):
        self.p_tilde = p_tilde; self.S = S; self.p_phys = p_phys
        self.rho_c_tilde = rho_c_tilde; self.rho_w_tilde = rho_w_tilde
        self.vwx = vwx; self.vwy = vwy; self.vcx = vcx; self.vcy = vcy


def pde_residual_batch(model, x_t, y_t, t_t, cfg: dict[str, Any], rho_co2_model: nn.Module, rho_c_ref: float):
    pack = phase_fluxes(model, x_t, y_t, t_t, cfg, rho_co2_model, rho_c_ref)
    Sw = 1.0 - pack.S
    d_w_dt = grad(pack.rho_w_tilde * Sw, t_t)
    d_c_dt = grad(pack.rho_c_tilde * pack.S, t_t)
    div_w = grad(pack.rho_w_tilde * pack.vwx, x_t) + grad(pack.rho_w_tilde * pack.vwy, y_t)
    div_c = grad(pack.rho_c_tilde * pack.vcx, x_t) + grad(pack.rho_c_tilde * pack.vcy, y_t)
    r_w = float(cfg.get("phi", DEFAULT_PHI)) * float(cfg["A_time"]) * d_w_dt + div_w
    r_c = float(cfg.get("phi", DEFAULT_PHI)) * float(cfg["A_time"]) * d_c_dt + div_c
    vtx = pack.vwx + pack.vcx
    vty = pack.vwy + pack.vcy
    return r_w, r_c, pack.p_tilde, pack.S, vtx, vty


def total_velocity_batch(model, x_t, y_t, t_t, cfg: dict[str, Any], rho_co2_model: nn.Module, rho_c_ref: float):
    pack = phase_fluxes(model, x_t, y_t, t_t, cfg, rho_co2_model, rho_c_ref)
    return pack.p_tilde, pack.S, pack.vwx + pack.vcx, pack.vwy + pack.vcy


# =============================================================================
# Sampling geometry
# =============================================================================
def _outside_wells_np(x: np.ndarray, y: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    L = float(cfg["L_ref"])
    R = float(cfg.get("r_well", DEFAULT_R_WELL))
    inj = cfg.get("inj_center", DEFAULT_INJ_CENTER)
    out = cfg.get("out_center", DEFAULT_OUT_CENTER)
    inj_cx, inj_cy = float(inj[0]), float(inj[1])
    out_cx, out_cy = float(out[0]), float(out[1])
    return (((x - inj_cx) ** 2 + (y - inj_cy) ** 2) >= R * R) & (((x - out_cx) ** 2 + (y - out_cy) ** 2) >= R * R) & (x >= 0) & (x <= L) & (y >= 0) & (y <= L)


def sample_xy_phys(N: int, cfg: dict[str, Any], rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    L = float(cfg["L_ref"])
    xs = [] ; ys = [] ; got = 0
    while got < N:
        M = int((N - got) * 1.25) + 128
        x = rng.random(M) * L
        y = rng.random(M) * L
        mask = _outside_wells_np(x, y, cfg)
        if np.any(mask):
            xs.append(x[mask]); ys.append(y[mask]); got += int(np.sum(mask))
    return np.concatenate(xs)[:N], np.concatenate(ys)[:N]


def tensors_from_phys(x: np.ndarray, y: np.ndarray, t: np.ndarray, cfg: dict[str, Any], device: torch.device, dtype: torch.dtype, requires_grad: bool = True):
    xt = torch.as_tensor(x.reshape(-1, 1) / float(cfg["L_ref"]), device=device, dtype=dtype)
    yt = torch.as_tensor(y.reshape(-1, 1) / float(cfg["L_ref"]), device=device, dtype=dtype)
    tt = torch.as_tensor(t.reshape(-1, 1) / float(cfg["T_ref"]), device=device, dtype=dtype)
    if requires_grad:
        xt.requires_grad_(True); yt.requires_grad_(True); tt.requires_grad_(True)
    return xt, yt, tt


def sample_injection_arc_np(N: int, cfg: dict[str, Any], rng: np.random.Generator):
    eps = DEFAULT_GEOM_EPS
    R = float(cfg.get("r_well", DEFAULT_R_WELL)); L = float(cfg["L_ref"]); T = float(cfg["T_ref"])
    inj = cfg.get("inj_center", DEFAULT_INJ_CENTER)
    cx, cy = float(inj[0]), float(inj[1])
    theta = eps + rng.random(N) * (0.5 * math.pi - 2.0 * eps)
    x = cx + R * np.cos(theta)
    y = cy + R * np.sin(theta)
    t0 = max(0.0, min(float(cfg.get("inj_bc_t_start_tilde", DEFAULT_INJ_BC_T_START_TILDE)), 0.99)) * T
    t = t0 + rng.random(N) * (T - t0)
    nx = np.cos(theta); ny = np.sin(theta)
    return x, y, t, nx, ny


def sample_outlet_arc_np(N: int, cfg: dict[str, Any], rng: np.random.Generator):
    eps = DEFAULT_GEOM_EPS
    R = float(cfg.get("r_well", DEFAULT_R_WELL)); T = float(cfg["T_ref"])
    out = cfg.get("out_center", DEFAULT_OUT_CENTER)
    cx, cy = float(out[0]), float(out[1])
    theta = math.pi + eps + rng.random(N) * (0.5 * math.pi - 2.0 * eps)
    x = cx + R * np.cos(theta)
    y = cy + R * np.sin(theta)
    t = rng.random(N) * T
    nx = -np.cos(theta); ny = -np.sin(theta)
    return x, y, t, nx, ny


def sample_outer_boundary_np(N_each: int, cfg: dict[str, Any], rng: np.random.Generator):
    eps = DEFAULT_GEOM_EPS
    L = float(cfg["L_ref"]); R = float(cfg.get("r_well", DEFAULT_R_WELL)); T = float(cfg["T_ref"])
    y1 = (R + eps) + rng.random(N_each) * (L - R - 2.0 * eps); x1 = np.zeros_like(y1); n1 = np.tile(np.array([-1.0, 0.0]), (N_each, 1))
    x2 = (R + eps) + rng.random(N_each) * (L - R - 2.0 * eps); y2 = np.zeros_like(x2); n2 = np.tile(np.array([0.0, -1.0]), (N_each, 1))
    y3 = rng.random(N_each) * (L - R - eps); x3 = np.full_like(y3, L); n3 = np.tile(np.array([1.0, 0.0]), (N_each, 1))
    x4 = rng.random(N_each) * (L - R - eps); y4 = np.full_like(x4, L); n4 = np.tile(np.array([0.0, 1.0]), (N_each, 1))
    x = np.concatenate([x1, x2, x3, x4]); y = np.concatenate([y1, y2, y3, y4]); n = np.vstack([n1, n2, n3, n4])
    t = rng.random(x.size) * T
    return x, y, t, n[:, 0], n[:, 1]


# =============================================================================
# Evaluation kernels
# =============================================================================
def load_run_model(spec, device: torch.device, physical_fallbacks: dict[str, float] | None = None):
    state, ckpt_cfg = load_checkpoint(spec.ckpt_path)
    cfg = default_cfg()
    # CLI values are fallbacks for legacy checkpoints; checkpoint config wins.
    if physical_fallbacks:
        for key, value in physical_fallbacks.items():
            cfg[key] = float(value)
    apply_exp_preset(cfg, spec.exp_name)
    cfg_update_from_ckpt(cfg, ckpt_cfg)
    physics_cfg_update_from_ckpt(cfg, ckpt_cfg)
    if "A_time" not in cfg:
        cfg["A_time"] = float(cfg["L_ref"]) / (float(cfg["U_ref"]) * float(cfg["T_ref"]))
    cfg["P_ref"] = float(cfg.get("P_ref_override")) if cfg.get("P_ref_override", None) is not None else float(cfg["P_ref"])
    model = build_model(cfg).to(device=device, dtype=dtype_from_cfg(cfg))
    model.load_state_dict(state, strict=False)
    model.set_beta(float(cfg.get("beta_eval", 1.0)))
    model.eval()
    return model, cfg, ckpt_cfg


def evaluate_pde(model, cfg, arrays, eval_plan, device, rho_model, rho_ref, n_points: int, batch_size: int, seed: int):
    rng = np.random.default_rng(seed)
    idx_all = eval_plan["pooled_idx"]
    if idx_all.size == 0:
        return {"summary": {}, "points": {}}
    n_eff = min(int(n_points), int(idx_all.size))
    idx = rng.choice(idx_all, size=n_eff, replace=False)
    x = eval_plan["x"][idx]; y = eval_plan["y"][idx]; t = eval_plan["t"][idx]
    s_true = eval_plan["s"][idx]
    dtype = dtype_from_cfg(cfg)
    rw_list=[]; rc_list=[]; st_list=[]; near_list=[]
    for i0 in range(0, n_eff, batch_size):
        sl = slice(i0, min(i0+batch_size, n_eff))
        xt, yt, tt = tensors_from_phys(x[sl], y[sl], t[sl], cfg, device, dtype, requires_grad=True)
        r_w, r_c, _, S, _, _ = pde_residual_batch(model, xt, yt, tt, cfg, rho_model, rho_ref)
        rw_list.append(r_w.detach().cpu().numpy().reshape(-1))
        rc_list.append(r_c.detach().cpu().numpy().reshape(-1))
        st_list.append(S.detach().cpu().numpy().reshape(-1))
        del xt, yt, tt, r_w, r_c, S
    rw = np.concatenate(rw_list); rc = np.concatenate(rc_list); sp = np.concatenate(st_list)
    total = np.sqrt(rw*rw + rc*rc)
    front_low = float(cfg.get("front_low", DEFAULT_FRONT_LOW))
    front_high = float(cfg.get("front_high", DEFAULT_FRONT_HIGH))
    front_mask = (s_true >= front_low) & (s_true <= front_high)
    inj = cfg.get("inj_center", DEFAULT_INJ_CENTER); r_near = float(DEFAULT_NEAR_INJECTION_RADIUS_M)
    near_mask = ((x - float(inj[0]))**2 + (y - float(inj[1]))**2) <= r_near*r_near
    row = {}
    row.update(metric_values(rw, "PDE_water"))
    row.update(metric_values(rc, "PDE_CO2"))
    row.update(metric_values(total, "PDE_total"))
    row.update(metric_values(total[front_mask], "PDE_total_front"))
    row.update(metric_values(rc[front_mask], "PDE_CO2_front"))
    row.update(metric_values(total[near_mask], "PDE_total_near_injection"))
    row.update(metric_values(rc[near_mask], "PDE_CO2_near_injection"))
    row["pde_eval_points"] = int(n_eff)
    row["front_fraction"] = float(np.mean(front_mask))
    row["near_injection_fraction"] = float(np.mean(near_mask))
    return row


def evaluate_bc_ic(model, cfg, device, rho_model, rho_ref, n_bc: int, n_outer_each: int, n_ic: int, seed: int):
    rng = np.random.default_rng(seed)
    dtype = dtype_from_cfg(cfg)
    row = {}
    # IC
    x0, y0 = sample_xy_phys(n_ic, cfg, rng)
    t0 = np.zeros_like(x0)
    xt, yt, tt = tensors_from_phys(x0, y0, t0, cfg, device, dtype, requires_grad=False)
    with torch.no_grad():
        p0t, S0 = model(xt, yt, tt)
    p0_np = p0t.detach().cpu().numpy().reshape(-1)
    S0_np = S0.detach().cpu().numpy().reshape(-1)
    row.update(metric_values(p0_np - 0.0, "IC_p_tilde"))
    row.update(metric_values(S0_np - float(cfg.get("s_ic_co2", DEFAULT_S_IC_CO2)), "IC_S"))

    # Injection BC
    x, y, t, nx, ny = sample_injection_arc_np(n_bc, cfg, rng)
    xt, yt, tt = tensors_from_phys(x, y, t, cfg, device, dtype, requires_grad=True)
    p, S, vx, vy = total_velocity_batch(model, xt, yt, tt, cfg, rho_model, rho_ref)
    nx_t = torch.as_tensor(nx.reshape(-1,1), device=device, dtype=dtype); ny_t = torch.as_tensor(ny.reshape(-1,1), device=device, dtype=dtype)
    vn = (vx * nx_t + vy * ny_t).detach().cpu().numpy().reshape(-1)
    S_inj = S.detach().cpu().numpy().reshape(-1)
    S_inj_target = float(cfg.get(
        "s_inj_co2",
        cfg.get("S_CO2_MAX", 1.0 - float(cfg.get("Sw_irr", DEFAULT_SW_IRR))),
    ))
    row.update(metric_values(vn - 1.0, "inj_flux"))
    row.update(metric_values(S_inj - S_inj_target, "inj_S"))
    row["inj_S_target"] = S_inj_target
    row["inj_t_start_tilde"] = float(cfg.get("inj_bc_t_start_tilde", DEFAULT_INJ_BC_T_START_TILDE))

    # Outlet BC and backflow
    x, y, t, nx, ny = sample_outlet_arc_np(n_bc, cfg, rng)
    xt, yt, tt = tensors_from_phys(x, y, t, cfg, device, dtype, requires_grad=True)
    p, S, vx, vy = total_velocity_batch(model, xt, yt, tt, cfg, rho_model, rho_ref)
    nx_t = torch.as_tensor(nx.reshape(-1,1), device=device, dtype=dtype); ny_t = torch.as_tensor(ny.reshape(-1,1), device=device, dtype=dtype)
    vn = (vx * nx_t + vy * ny_t).detach().cpu().numpy().reshape(-1)
    p_out_err = (p - float(cfg.get("p_out_tilde", 0.0))).detach().cpu().numpy().reshape(-1)
    backflow = np.maximum(0.0, -vn)
    row.update(metric_values(p_out_err, "outlet_p_tilde"))
    row.update(metric_values(backflow, "outlet_backflow"))
    row["outlet_backflow_fraction"] = float(np.mean(backflow > 0.0))

    # Outer no-flow
    x, y, t, nx, ny = sample_outer_boundary_np(n_outer_each, cfg, rng)
    vals = []
    for i0 in range(0, x.size, n_bc):
        sl = slice(i0, min(i0+n_bc, x.size))
        xt, yt, tt = tensors_from_phys(x[sl], y[sl], t[sl], cfg, device, dtype, requires_grad=True)
        _, _, vx, vy = total_velocity_batch(model, xt, yt, tt, cfg, rho_model, rho_ref)
        nx_t = torch.as_tensor(nx[sl].reshape(-1,1), device=device, dtype=dtype); ny_t = torch.as_tensor(ny[sl].reshape(-1,1), device=device, dtype=dtype)
        vals.append((vx * nx_t + vy * ny_t).detach().cpu().numpy().reshape(-1))
    noflow_vn = np.concatenate(vals)
    row.update(metric_values(noflow_vn, "noflow_vn"))
    return row


def evaluate_mass(model, cfg, arrays, eval_plan, device, rho_model, rho_ref, sample_per_time: int, batch_size: int, seed: int):
    rng = np.random.default_rng(seed)
    x_all = eval_plan["x"]; y_all = eval_plan["y"]; t_all = eval_plan["t"]; p_all = eval_plan["p"]; s_all = eval_plan["s"]
    rho_c_true_arr = to_np_1d(arrays["rho_c"]).astype(np.float64, copy=False) if "rho_c" in arrays else None
    time_rows = []
    area_domain = float(cfg["L_ref"])**2 - 0.5 * math.pi * float(cfg.get("r_well", DEFAULT_R_WELL))**2
    dtype = dtype_from_cfg(cfg)
    for k, idx0 in enumerate(eval_plan["indices_per_time"]):
        if idx0.size == 0:
            continue
        idx = idx0
        if sample_per_time > 0 and idx.size > sample_per_time:
            idx = rng.choice(idx, size=int(sample_per_time), replace=False)
        x = x_all[idx]; y = y_all[idx]; t = t_all[idx]; s_true = s_all[idx]; p_true = p_all[idx]
        p_pred_tilde, s_pred = predict_model(model, x, y, t, cfg, device, batch_size)
        p_pred_pa = float(cfg["p0"]) + float(cfg["P_ref"]) * p_pred_tilde
        # rho pred via torch model in no_grad batches
        rho_pred = []
        with torch.no_grad():
            for i0 in range(0, p_pred_pa.size, batch_size):
                pp = torch.as_tensor(p_pred_pa[i0:i0+batch_size].reshape(-1,1), device=device, dtype=dtype)
                rho_pred.append(rho_model(pp).detach().cpu().numpy().reshape(-1))
        rho_pred = np.concatenate(rho_pred)
        if rho_c_true_arr is not None:
            rho_true = rho_c_true_arr[idx]
        else:
            # Fall back to EOS evaluated at true pressure.
            rho_true = []
            with torch.no_grad():
                for i0 in range(0, p_true.size, batch_size):
                    pp = torch.as_tensor(p_true[i0:i0+batch_size].reshape(-1,1), device=device, dtype=dtype)
                    rho_true.append(rho_model(pp).detach().cpu().numpy().reshape(-1))
            rho_true = np.concatenate(rho_true)
        M_c_pred = area_domain * float(cfg.get("phi", DEFAULT_PHI)) * float(np.mean(rho_pred * s_pred))
        M_c_true = area_domain * float(cfg.get("phi", DEFAULT_PHI)) * float(np.mean(rho_true * s_true))
        M_w_pred = area_domain * float(cfg.get("phi", DEFAULT_PHI)) * float(cfg.get("rho_w_const", DEFAULT_RHO_W_CONST)) * float(np.mean(1.0 - s_pred))
        M_w_true = area_domain * float(cfg.get("phi", DEFAULT_PHI)) * float(cfg.get("rho_w_const", DEFAULT_RHO_W_CONST)) * float(np.mean(1.0 - s_true))
        row = {
            "time_index": int(k), "time_s": float(eval_plan["time_seconds"][k]),
            "CO2_mass_true": M_c_true, "CO2_mass_pred": M_c_pred,
            "CO2_mass_abs_error": abs(M_c_pred - M_c_true),
            "CO2_mass_rel_error": np.nan,  # set after the loop: normalized by peak CO2 mass (see below)
            "water_mass_true": M_w_true, "water_mass_pred": M_w_pred,
            "water_mass_rel_error": abs(M_w_pred - M_w_true) / (abs(M_w_true) + 1.0e-30),
            "total_mass_true": M_c_true + M_w_true, "total_mass_pred": M_c_pred + M_w_pred,
            "total_mass_rel_error": abs((M_c_pred + M_w_pred) - (M_c_true + M_w_true)) / (abs(M_c_true + M_w_true) + 1.0e-30),
            "mass_eval_points": int(idx.size),
        }
        time_rows.append(row)

    # Robust CO2-mass relative error: normalize by the PEAK true CO2 mass in place
    # (~ cumulative injected CO2), not the near-zero early-time mass. The raw per-time
    # ratio abs(dM)/(|M_c_true|+1e-30) blew up to ~1e27 because M_c_true ~ 0 before CO2
    # arrives; this rescales every row by a single stable reference instead.
    _co2_ref = max((r["CO2_mass_true"] for r in time_rows), default=0.0)
    _co2_ref = abs(_co2_ref) if abs(_co2_ref) > 1.0e-30 else 1.0
    for r in time_rows:
        r["CO2_mass_rel_error"] = r["CO2_mass_abs_error"] / _co2_ref
    return time_rows


def sample_fv_cells_np(N: int, cfg: dict[str, Any], h_tilde: float, dt_tilde: float, rng: np.random.Generator):
    L = float(cfg["L_ref"]); T = float(cfg["T_ref"])
    h_phys = float(h_tilde) * L
    margin = 0.55 * h_phys
    xs=[]; ys=[]; got=0
    while got < N:
        M = int((N-got)*1.3)+128
        x = margin + rng.random(M) * (L - 2.0*margin)
        y = margin + rng.random(M) * (L - 2.0*margin)
        mask = _outside_wells_np(x, y, cfg)
        # also keep faces outside wells
        mask &= _outside_wells_np(x+0.5*h_phys, y, cfg) & _outside_wells_np(x-0.5*h_phys, y, cfg)
        mask &= _outside_wells_np(x, y+0.5*h_phys, cfg) & _outside_wells_np(x, y-0.5*h_phys, cfg)
        if np.any(mask):
            xs.append(x[mask]); ys.append(y[mask]); got += int(np.sum(mask))
    x = np.concatenate(xs)[:N]; y = np.concatenate(ys)[:N]
    t0_tilde = rng.random(N) * max(1.0e-12, 1.0 - float(dt_tilde))
    t1_tilde = t0_tilde + float(dt_tilde)
    return x/L, y/L, t0_tilde, t1_tilde


def evaluate_local_fv(model, cfg, device, rho_model, rho_ref, N: int, batch_size: int, h_tilde: float, dt_tilde: float | None, seed: int):
    rng = np.random.default_rng(seed)
    if dt_tilde is None or float(dt_tilde) <= 0.0:
        # If dataset times are not used, default to a small, stable nondimensional interval.
        dt_tilde = 1.0 / 128.0
    dtype = dtype_from_cfg(cfg)
    xc_all, yc_all, t0_all, t1_all = sample_fv_cells_np(N, cfg, h_tilde, float(dt_tilde), rng)
    rc_all=[]; rw_all=[]; s1_all=[]
    for i0 in range(0, N, batch_size):
        sl = slice(i0, min(i0+batch_size, N))
        xc = torch.as_tensor(xc_all[sl].reshape(-1,1), device=device, dtype=dtype)
        yc = torch.as_tensor(yc_all[sl].reshape(-1,1), device=device, dtype=dtype)
        t0 = torch.as_tensor(t0_all[sl].reshape(-1,1), device=device, dtype=dtype)
        t1 = torch.as_tensor(t1_all[sl].reshape(-1,1), device=device, dtype=dtype)
        h = torch.tensor(float(h_tilde), device=device, dtype=dtype)
        area = h*h
        with torch.enable_grad():
            p0t, S0 = model(xc, yc, t0)
            p1t, S1 = model(xc, yc, t1)
            p0phys = float(cfg["p0"]) + float(cfg["P_ref"]) * p0t
            p1phys = float(cfg["p0"]) + float(cfg["P_ref"]) * p1t
            rho_c0 = rho_model(p0phys) / float(rho_ref)
            rho_c1 = rho_model(p1phys) / float(rho_ref)
            acc_c = float(cfg.get("phi", DEFAULT_PHI)) * float(cfg["A_time"]) * area * ((rho_c1*S1 - rho_c0*S0) / float(dt_tilde))
            Sw0 = 1.0 - S0; Sw1 = 1.0 - S1
            acc_w = float(cfg.get("phi", DEFAULT_PHI)) * float(cfg["A_time"]) * area * ((Sw1 - Sw0) / float(dt_tilde))
            xe = (xc + 0.5*h).detach().clone().requires_grad_(True); xw = (xc - 0.5*h).detach().clone().requires_grad_(True)
            xn = xc.detach().clone().requires_grad_(True); xs = xc.detach().clone().requires_grad_(True)
            ye = yc.detach().clone().requires_grad_(True); yw = yc.detach().clone().requires_grad_(True)
            yn = (yc + 0.5*h).detach().clone().requires_grad_(True); ys = (yc - 0.5*h).detach().clone().requires_grad_(True)
            te = t1.detach().clone(); tw = t1.detach().clone(); tn = t1.detach().clone(); ts = t1.detach().clone()
            Fe = phase_fluxes(model, xe, ye, te, cfg, rho_model, rho_ref)
            Fw = phase_fluxes(model, xw, yw, tw, cfg, rho_model, rho_ref)
            Fn = phase_fluxes(model, xn, yn, tn, cfg, rho_model, rho_ref)
            Fs = phase_fluxes(model, xs, ys, ts, cfg, rho_model, rho_ref)
            flux_c = h * ((Fe.rho_c_tilde*Fe.vcx) - (Fw.rho_c_tilde*Fw.vcx) + (Fn.rho_c_tilde*Fn.vcy) - (Fs.rho_c_tilde*Fs.vcy))
            flux_w = h * ((Fe.rho_w_tilde*Fe.vwx) - (Fw.rho_w_tilde*Fw.vwx) + (Fn.rho_w_tilde*Fn.vwy) - (Fs.rho_w_tilde*Fs.vwy))
            r_c = (acc_c + flux_c) / (area + 1.0e-12)
            r_w = (acc_w + flux_w) / (area + 1.0e-12)
        rc_all.append(r_c.detach().cpu().numpy().reshape(-1))
        rw_all.append(r_w.detach().cpu().numpy().reshape(-1))
        s1_all.append(S1.detach().cpu().numpy().reshape(-1))
    rc = np.concatenate(rc_all); rw = np.concatenate(rw_all); s1 = np.concatenate(s1_all)
    total = rc + rw
    front_low = float(cfg.get("front_low", DEFAULT_FRONT_LOW))
    front_high = float(cfg.get("front_high", DEFAULT_FRONT_HIGH))
    front = (s1 >= front_low) & (s1 <= front_high)
    back = s1 < DEFAULT_BACKGROUND_THRESHOLD
    row = {}
    row.update(metric_values(rc, "local_FV_CO2"))
    row.update(metric_values(rw, "local_FV_water"))
    row.update(metric_values(total, "local_FV_total"))
    row.update(metric_values(rc[front], "local_FV_CO2_front"))
    row.update(metric_values(total[front], "local_FV_total_front"))
    row.update(metric_values(rc[back], "local_FV_CO2_background"))
    row.update(metric_values(total[back], "local_FV_total_background"))
    row["fv_eval_cells"] = int(N)
    row["fv_h_tilde"] = float(h_tilde)
    row["fv_dt_tilde"] = float(dt_tilde)
    row["fv_front_fraction"] = float(np.mean(front))
    row["fv_background_fraction"] = float(np.mean(back))
    return row


# =============================================================================
# Run evaluation
# =============================================================================
def evaluate_run(spec, arrays, eval_plan, device, args):
    model, cfg, ckpt_cfg = load_run_model(
        spec,
        device,
        {"s_ic_co2": args.s_ic_co2, "Snr": args.Snr, "Sw_irr": args.Sw_irr},
    )
    dtype = dtype_from_cfg(cfg)
    rho_model, rho_ref = fit_rho_model_from_arrays(arrays, cfg, args.eos_degree, args.eos_fit_sample, args.random_seed, dtype, device)
    common = dict(
        run_id=short_label(spec, bool(args.loo_labels)),
        raw_run_id=spec.run_id,
        exp_name=spec.exp_name,
        training_strategy=spec.training_strategy,
        leave_out=spec.leave_out,
        seed=spec.seed,
        ckpt_path=spec.ckpt_path,
    )
    print(f"[physics] {common['run_id']} raw={spec.run_id}")
    pde = evaluate_pde(model, cfg, arrays, eval_plan, device, rho_model, rho_ref, args.pde_points, args.pde_batch_size, args.random_seed)
    bcic = evaluate_bc_ic(model, cfg, device, rho_model, rho_ref, args.bc_points, args.outer_points_each, args.ic_points, args.random_seed + 17)
    mass_rows = evaluate_mass(model, cfg, arrays, eval_plan, device, rho_model, rho_ref, args.mass_sample_per_time, args.infer_batch_size, args.random_seed + 31)
    fv_h_tilde = float(cfg.get("fv_h_tilde", args.fv_h_tilde))
    fv_dt_cfg = cfg.get("fv_dt_tilde", args.fv_dt_tilde)
    fv_dt_tilde = None if fv_dt_cfg is None else float(fv_dt_cfg)
    fv = evaluate_local_fv(model, cfg, device, rho_model, rho_ref, args.fv_n_cells_eval, args.fv_batch_size, fv_h_tilde, fv_dt_tilde, args.random_seed + 47)

    pde_row = {**common, **pde}
    bcic_row = {**common, **bcic}
    fv_row = {**common, **fv}
    mass_out = []
    for r in mass_rows:
        rr = {**common, **r}
        rr["time_display"] = time_to_display(rr["time_s"], args.time_unit)
        mass_out.append(rr)
    # Summary row combines selected headline metrics.
    co2_rel = np.array([r["CO2_mass_rel_error"] for r in mass_rows], dtype=np.float64)
    total_rel = np.array([r["total_mass_rel_error"] for r in mass_rows], dtype=np.float64)
    summary = {**common}
    for key in ["PDE_total_RMSE", "PDE_total_p95", "PDE_CO2_RMSE", "PDE_CO2_p95", "PDE_total_front_RMSE", "PDE_total_front_p95"]:
        summary[key] = pde.get(key, np.nan)
    for key in ["IC_S_RMSE", "IC_p_tilde_RMSE", "inj_flux_RMSE", "inj_S_RMSE", "outlet_p_tilde_RMSE", "outlet_backflow_fraction", "noflow_vn_RMSE"]:
        summary[key] = bcic.get(key, np.nan)
    summary["CO2_mass_rel_error_mean"] = float(np.nanmean(co2_rel)) if co2_rel.size else np.nan
    summary["CO2_mass_rel_error_max"] = float(np.nanmax(co2_rel)) if co2_rel.size else np.nan
    summary["total_mass_rel_error_mean"] = float(np.nanmean(total_rel)) if total_rel.size else np.nan
    summary["local_FV_CO2_RMSE"] = fv.get("local_FV_CO2_RMSE", np.nan)
    summary["local_FV_total_RMSE"] = fv.get("local_FV_total_RMSE", np.nan)
    return summary, pde_row, bcic_row, mass_out, fv_row, {k: v for k, v in cfg.items() if k != "checkpoint_config"}


# =============================================================================
# Plotting
# =============================================================================
def plot_physics(summary_rows: list[dict[str, Any]], mass_rows: list[dict[str, Any]], out_dir: str, time_unit: str, dpi: int, export_pdf: bool, export_tiff: bool):
    import pandas as pd
    if not summary_rows:
        return
    set_journal_style()
    fig_dir = os.path.join(out_dir, "figures")
    ensure_dir(fig_dir)
    df = pd.DataFrame(summary_rows)

    # Overview bar panels. Keep legends outside; use short run_id only.
    metrics = [
        ("PDE_total_p95", "PDE total p95"),
        ("IC_S_RMSE", "IC S RMSE"),
        ("local_FV_CO2_RMSE", "Local FV CO₂ RMSE"),
        ("CO2_mass_rel_error_mean", "CO₂ mass rel. err."),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.6), constrained_layout=False)
    axes = axes.ravel()
    labels = df["run_id"].astype(str).tolist()
    x = np.arange(len(labels))
    for ax, (col, title) in zip(axes, metrics):
        vals = df[col].to_numpy(dtype=float) if col in df.columns else np.full(len(df), np.nan)
        ax.bar(x, vals, width=0.72, color=[color_for_label(lab, "#7F7F7F") for lab in labels])
        ax.set_title(title, pad=3)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        apply_nature_axis(ax, grid=True, grid_axis="y")
    fig.subplots_adjust(left=0.08, right=0.99, top=0.93, bottom=0.20, wspace=0.33, hspace=0.45)
    _savefig(fig, os.path.join(fig_dir, "physics_summary_overview"), dpi, export_pdf, export_tiff)
    plt.close(fig)

    # Mass error over time.
    if mass_rows:
        mdf = pd.DataFrame(mass_rows)
        fig, ax = plt.subplots(figsize=(3.6, 2.45), constrained_layout=False)
        plotted = {}
        for run_id, g in mdf.groupby("run_id", sort=False):
            x = g["time_display"].to_numpy(dtype=float)
            y = g["CO2_mass_rel_error"].to_numpy(dtype=float)
            plotted[str(run_id)] = (x, y)
            ax.plot(x, y, label=run_id, color=color_for_label(run_id, None))
        ax.set_xlabel(time_axis_label(time_unit))
        ax.set_ylabel("CO₂ mass rel. error")
        apply_nature_axis(ax, grid=True, grid_axis="both")
        apply_robust_time_axis(
            ax,
            plotted,
            x_cut=transient_cut_for_unit(time_unit),
            color_for_label=color_for_label,
        )
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=min(4, max(1, mdf["run_id"].nunique())), frameon=False, columnspacing=1.0, handlelength=1.6)
        fig.subplots_adjust(left=0.16, right=0.98, top=0.96, bottom=0.36)
        _savefig(fig, os.path.join(fig_dir, "CO2_mass_rel_error_vs_time"), dpi, export_pdf, export_tiff)
        plt.close(fig)


def write_csv_rows(path: str, rows: list[dict[str, Any]]):
    write_csv(path, rows)


# =============================================================================
# Main
# =============================================================================
def main():
    ap = argparse.ArgumentParser(description="Physics-consistency evaluator for M0-M7 and M7 leave-one-out checkpoints.")
    ap.add_argument("--ckpt-dir", type=str, default=DEFAULT_CKPT_DIR)
    ap.add_argument("--data", type=str, default=DEFAULT_DATA_PATH)
    ap.add_argument("--out-root", type=str, default=DEFAULT_OUT_ROOT)
    ap.add_argument("--experiments", type=str, default=",".join(ALL_EXPS))
    ap.add_argument("--training-strategy", type=str, default="BASE")
    ap.add_argument("--include-loo", type=int, default=0)
    ap.add_argument("--loo-exp", type=str, default="M7")
    ap.add_argument("--leave-outs", type=str, default="NONE,PLAIN_TWONET,FRONT_PLUME,PAIRGRAD,RAR,FV,COARSE_DETAIL")
    ap.add_argument("--loo-labels", type=int, default=0, help="Label M7 LOO-NONE as L0 in plots/tables.")
    ap.add_argument("--seed", type=str, default="ANY")
    ap.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    ap.add_argument("--gpu-id", type=int, default=DEFAULT_GPU_ID)
    ap.add_argument("--dtype", type=str, default=DEFAULT_DTYPE)
    ap.add_argument("--infer-batch-size", type=int, default=DEFAULT_INFER_BATCH_SIZE)
    ap.add_argument("--max-time-steps", type=int, default=DEFAULT_MAX_TIME_STEPS)
    ap.add_argument("--max-sort-n", type=int, default=DEFAULT_MAX_SORT_N)
    ap.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    ap.add_argument("--time-indices", type=str, default=DEFAULT_TIME_INDICES)
    ap.add_argument("--time-unit", type=str, default=DEFAULT_TIME_UNIT)
    ap.add_argument("--pde-points", type=int, default=DEFAULT_PDE_POINTS)
    ap.add_argument("--pde-batch-size", type=int, default=DEFAULT_PDE_BATCH_SIZE)
    ap.add_argument("--bc-points", type=int, default=DEFAULT_BC_POINTS)
    ap.add_argument("--outer-points-each", type=int, default=DEFAULT_OUTER_POINTS_EACH)
    ap.add_argument("--ic-points", type=int, default=DEFAULT_IC_POINTS)
    ap.add_argument("--mass-sample-per-time", type=int, default=DEFAULT_MASS_SAMPLE_PER_TIME)
    ap.add_argument("--fv-n-cells-eval", type=int, default=DEFAULT_FV_N_CELLS_EVAL)
    ap.add_argument("--fv-batch-size", type=int, default=DEFAULT_FV_BATCH_SIZE)
    ap.add_argument("--fv-h-tilde", type=float, default=DEFAULT_FV_H_TILDE)
    ap.add_argument("--fv-dt-tilde", type=float, default=DEFAULT_FV_DT_TILDE)
    ap.add_argument("--eos-fit-sample", type=int, default=DEFAULT_EOS_FIT_SAMPLE)
    ap.add_argument("--eos-degree", type=int, default=DEFAULT_EOS_DEGREE)
    ap.add_argument("--figure-dpi", type=int, default=DEFAULT_FIGURE_DPI)
    ap.add_argument("--export-pdf", type=int, default=int(DEFAULT_EXPORT_PDF))
    ap.add_argument("--export-tiff", type=int, default=int(DEFAULT_EXPORT_TIFF))
    ap.add_argument("--export-svg", type=int, default=int(DEFAULT_EXPORT_SVG),
                    help="Export editable SVG files in addition to PNG/PDF/TIFF.")
    ap.add_argument("--s-ic-co2", type=float, default=DEFAULT_S_IC_CO2,
                    help="Training-script initial CO2 saturation used for IC metrics.")
    ap.add_argument("--Snr", type=float, default=DEFAULT_SNR,
                    help="Training-script CO2 residual saturation used in relative permeability.")
    ap.add_argument("--Sw-irr", type=float, default=DEFAULT_SW_IRR,
                    help="Training-script irreducible water saturation used in relative permeability.")
    args = ap.parse_args()
    global EXPORT_SVG
    EXPORT_SVG = bool(args.export_svg)

    out_dir = os.path.join(args.out_root, "physics")
    ensure_dir(out_dir); ensure_dir(os.path.join(out_dir, "figures"))
    device = torch.device(f"cuda:{args.gpu_id}" if args.device.lower().startswith("cuda") and torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")
    print(f"[data] {args.data}")
    print(f"[ckpt-dir] {args.ckpt_dir}")
    arrays, time_unique = load_dataset_pack(args.data)
    eval_plan = build_eval_plan(arrays, time_unique, args.mass_sample_per_time, args.max_time_steps, args.max_sort_n, args.random_seed, args.time_indices)
    print(f"[time] n={len(eval_plan['time_seconds'])}, selected={eval_plan['selected_items']}")

    experiments = [x.strip().upper() for x in args.experiments.split(",") if x.strip()]
    leave_outs = [x.strip().upper() for x in args.leave_outs.split(",") if x.strip()]
    seed = None if args.seed.upper() == "ANY" else args.seed
    specs = make_run_specs(args.ckpt_dir, experiments, args.training_strategy.upper(), bool(args.include_loo), args.loo_exp.upper(), leave_outs, seed)
    if not specs:
        raise RuntimeError("No checkpoints found. Check --ckpt-dir, --experiments, --training-strategy, and --leave-outs.")
    print(f"[runs] {len(specs)}")
    for s in specs:
        print(f"  - {short_label(s, bool(args.loo_labels))}: {s.ckpt_path}")

    summary_rows=[]; pde_rows=[]; bcic_rows=[]; mass_rows=[]; fv_rows=[]; resolved={}
    for spec in specs:
        summary, pde, bcic, mass, fv, cfg = evaluate_run(spec, arrays, eval_plan, device, args)
        summary_rows.append(summary); pde_rows.append(pde); bcic_rows.append(bcic); mass_rows.extend(mass); fv_rows.append(fv); resolved[summary["run_id"]] = cfg

    write_csv_rows(os.path.join(out_dir, "summary_physics_consistency.csv"), summary_rows)
    write_csv_rows(os.path.join(out_dir, "pde_metrics.csv"), pde_rows)
    write_csv_rows(os.path.join(out_dir, "bc_ic_metrics.csv"), bcic_rows)
    write_csv_rows(os.path.join(out_dir, "mass_metrics.csv"), mass_rows)
    write_csv_rows(os.path.join(out_dir, "local_fv_metrics.csv"), fv_rows)
    save_json({
        "args": vars(args),
        "selected_time_items": eval_plan["selected_items"],
        "S_IC_CO2_default": DEFAULT_S_IC_CO2,
        "Snr_default": DEFAULT_SNR,
        "notes": {
            "S_IC": "IC saturation metrics use the training-script default S_IC_CO2=0.0 unless overridden by checkpoint metadata.",
            "Snr": "CO2 residual saturation Snr=0.0 by default; raw Sco2 is not remapped.",
            "FV": "LEAVE_OUT=FV means the training FV loss was removed; this evaluator still computes diagnostic FV residuals for all runs.",
        },
        "resolved_configs": resolved,
    }, os.path.join(out_dir, "physics_evaluation_config.json"))
    plot_physics(summary_rows, mass_rows, out_dir, args.time_unit, args.figure_dpi, bool(args.export_pdf), bool(args.export_tiff))
    print(f"[done] outputs written to {out_dir}")


if __name__ == "__main__":
    main()
