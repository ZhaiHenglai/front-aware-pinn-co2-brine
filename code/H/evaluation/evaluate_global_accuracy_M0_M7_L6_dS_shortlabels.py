#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Global-accuracy evaluation for M0-M7 PINN checkpoints and M7 leave-one-out runs.

Outputs
-------
  global/summary_global_accuracy.csv
  global/by_time_global_accuracy.csv
  global/selected_time_global_accuracy.csv
  global/figures/*.png|.pdf|.tiff

Scope
-----
This script evaluates only global and region-wise pointwise accuracy:
  - pressure p_tilde and physical pressure p_phys
  - CO2 saturation Sco2
  - front-band / plume / background / core saturation accuracy
  - dS-based excess-plume / excess-background diagnostics aligned to S_IC_CO2
  - initial-condition saturation error against S_IC_CO2, default 0.0

It intentionally does not compute contour, pair-gradient, FV, PDE, BC, or snapshot
visualization metrics; those belong to the front/physics/snapshot evaluators.

The time-layer selector is intentionally aligned with
`evaluate_snapshot_gridviz_M0_M7_publication_snapshots.py`:
  t0, early, middle, late, final.
"""
from __future__ import annotations

import os
import re
import csv
import json
import math
import argparse
import glob
from dataclasses import dataclass
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
except Exception:  # pragma: no cover - keep script portable if copied alone
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
ALL_EXPS = tuple(f"M{i}" for i in range(8))
DEFAULT_DATA_PATH = "./tables_cache_0_723_step1.pt"
DEFAULT_CKPT_DIR = "./Results1"
DEFAULT_OUT_DIR = "./eval_M0_M7/global"
DEFAULT_DEVICE = "cpu"
DEFAULT_GPU_ID = 0
DEFAULT_DTYPE = "float64"
DEFAULT_SAMPLE_PER_TIME = 20000   # <= 0 means use all points per time layer
DEFAULT_INFER_BATCH_SIZE = 65536
DEFAULT_RANDOM_SEED = 0
DEFAULT_MAX_TIME_STEPS = None
DEFAULT_MAX_SORT_N = 500_000_000
DEFAULT_TIME_INDICES = "t0,early,middle,late,final"
DEFAULT_TIME_UNIT = "days"

# Physical/model defaults, overwritten by checkpoint config whenever present.
DEFAULT_WIDTH = 128
DEFAULT_MS_M_PER_SCALE = 24
DEFAULT_MS_SCALES = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
DEFAULT_MS_BASE_SCALE = 3.0
DEFAULT_M3_COARSE_M_PER_SCALE = 24
DEFAULT_M3_COARSE_SCALES = (1.0, 2.0, 4.0, 8.0)
DEFAULT_M3_COARSE_BASE_SCALE = 3.0
DEFAULT_M3_DETAIL_M_PER_SCALE = 24
DEFAULT_M3_DETAIL_SCALES = (4.0, 8.0, 16.0, 32.0)
DEFAULT_M3_DETAIL_BASE_SCALE = 3.0
DEFAULT_M3_COARSE_DEPTH = 6
DEFAULT_M3_DETAIL_DEPTH = 6
DEFAULT_M3_DETAIL_GAIN = 1.0
DEFAULT_M3_DETAIL_TANH = True
DEFAULT_S_EPS = 1.0e-6
DEFAULT_BETA_EVAL = 1.0
DEFAULT_S_IC_CO2 = 0.0
DEFAULT_T_IC_GATE = 1.0e-2
DEFAULT_SW_IRR = 0.2
DEFAULT_SNR = 0.0
DEFAULT_S_CO2_MAX = 1.0 - DEFAULT_SW_IRR

DEFAULT_L_REF = 5.0
DEFAULT_T_REF = 1.0e5
DEFAULT_MU_REF = 2.5e-4
DEFAULT_U_REF = 5.4e-5
DEFAULT_K_REF = 1.0e-14
DEFAULT_P0 = 10.0e6
DEFAULT_P_REF_OVERRIDE = None

# Region definitions for global-region accuracy.
DEFAULT_FRONT_LOW = 0.05
DEFAULT_FRONT_HIGH = 0.30
DEFAULT_FRONT_CENTER = 0.175
DEFAULT_FRONT_SIGMA = 0.075
DEFAULT_PLUME_THRESHOLD = 0.05
# dS = Sco2 - S_IC_CO2 diagnostics. These are the preferred plume/background
# definitions when the initial CO2 saturation is nonzero.  The current M0-M7
# training script uses S_IC_CO2=0.0, so raw-S and dS regions coincide by default.
DEFAULT_EXCESS_PLUME_THRESHOLD = 0.05
DEFAULT_EXCESS_MOBILE_THRESHOLD = 0.02
DEFAULT_EXCESS_BACKGROUND_THRESHOLD = 0.01
DEFAULT_EXCESS_LEADING_BAND_CENTER = 0.175
DEFAULT_EXCESS_LEADING_BAND_HALF_WIDTH = 0.02
DEFAULT_BACKGROUND_THRESHOLD = 0.02
DEFAULT_CORE_THRESHOLD = 0.80

# Publication figure defaults.
DEFAULT_FIGURE_DPI = 600
DEFAULT_EXPORT_PDF = True
DEFAULT_EXPORT_TIFF = True
DEFAULT_EXPORT_SVG = True
EXPORT_SVG = DEFAULT_EXPORT_SVG
DEFAULT_PRESSURE_DISPLAY_UNIT = "MPa"  # Pa / kPa / MPa



# Compact labels used in all figures.  Long run IDs are retained in CSV files
# for traceability, but figures should only show M0-M7 or L0-L6.
LOO_LABELS = {
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

# =============================================================================
# Small utilities
# =============================================================================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def to_np_1d(a) -> np.ndarray:
    if isinstance(a, torch.Tensor):
        return a.detach().cpu().numpy().reshape(-1)
    return np.asarray(a).reshape(-1)


def dtype_from_cfg(cfg: dict[str, Any]) -> torch.dtype:
    return torch.float32 if "32" in str(cfg.get("dtype", DEFAULT_DTYPE)).lower() else torch.float64


def pressure_unit_factor(unit: str) -> float:
    u = str(unit).lower()
    if u == "pa":
        return 1.0
    if u == "kpa":
        return 1.0e-3
    if u == "mpa":
        return 1.0e-6
    raise ValueError(f"Unsupported pressure unit: {unit!r}; use Pa/kPa/MPa")


def safe_float(x, default=np.nan) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def save_json(obj: Any, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


# =============================================================================
# Time selectors aligned with snapshot script
# =============================================================================
def parse_time_indices(s: str):
    if isinstance(s, (list, tuple)):
        return list(s)
    out = []
    for part in str(s).split(","):
        p = part.strip()
        if not p:
            continue
        pl = p.lower()
        if pl in ("t0", "initial", "early", "middle", "mid", "late", "final", "last"):
            out.append(pl)
        else:
            out.append(int(p))
    return out


def _named_snapshot_index(name: str, n_time: int) -> int:
    if n_time <= 0:
        raise ValueError("time_unique is empty")
    name = str(name).lower().strip()
    if name in ("t0", "initial"):
        return 0
    if name == "early":
        return min(max(1, int(round(0.25 * (n_time - 1)))), n_time - 1) if n_time > 1 else 0
    if name in ("middle", "mid"):
        return int(round(0.50 * (n_time - 1)))
    if name == "late":
        return int(round(0.75 * (n_time - 1)))
    if name in ("final", "last"):
        return n_time - 1
    raise ValueError(f"Unknown named time index: {name!r}")


def resolve_snapshot_items(time_unique: np.ndarray, spec) -> list[tuple[str, int]]:
    Tn = int(len(time_unique))
    seen = set()
    out: list[tuple[str, int]] = []
    for item in spec:
        if isinstance(item, str):
            idx = _named_snapshot_index(item, Tn)
            label = item.lower().strip()
        else:
            idx = int(item)
            if idx < 0:
                idx = Tn + idx
            idx = min(max(idx, 0), Tn - 1)
            label = f"idx{idx}"
        if idx in seen:
            continue
        seen.add(idx)
        out.append((label, idx))
    return out


def time_to_display(t_seconds: float, unit: str) -> float:
    u = str(unit).lower()
    if u in ("s", "sec", "second", "seconds"):
        return float(t_seconds)
    if u in ("h", "hr", "hour", "hours"):
        return float(t_seconds) / 3600.0
    if u in ("d", "day", "days"):
        return float(t_seconds) / 86400.0
    return float(t_seconds)


def time_axis_label(unit: str) -> str:
    u = str(unit).lower()
    if u in ("s", "sec", "second", "seconds"):
        return "time (s)"
    if u in ("h", "hr", "hour", "hours"):
        return "time (h)"
    if u in ("d", "day", "days"):
        return "time (days)"
    return "time"


# =============================================================================
# Model definitions matching M0-M7 training architecture
# =============================================================================
class MultiScaleFourierFeatures(nn.Module):
    def __init__(self, dtype: torch.dtype, in_dim=3, m_per_scale=24, scales=DEFAULT_MS_SCALES, base_scale=3.0):
        super().__init__()
        self.scales = tuple(float(s) for s in scales)
        B_list = []
        for s in self.scales:
            B_list.append(torch.randn(in_dim, int(m_per_scale), dtype=dtype) * (float(base_scale) * s))
        self.register_buffer("B", torch.cat(B_list, dim=1))

    @property
    def out_dim(self):
        return 2 * self.B.shape[1]

    def forward(self, x):
        if x.dtype != self.B.dtype or x.device != self.B.device:
            x = x.to(device=self.B.device, dtype=self.B.dtype)
        proj = x @ self.B
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, width=128, depth=6):
        super().__init__()
        layers = [nn.Linear(in_dim, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers += [nn.Linear(width, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        param = next(self.parameters())
        if x.dtype != param.dtype or x.device != param.device:
            x = x.to(device=param.device, dtype=param.dtype)
        return self.net(x)


class _HardICMixin:
    def apply_hard_ic(self, S_raw: torch.Tensor, t_t: torch.Tensor) -> torch.Tensor:
        if not bool(getattr(self, "use_hard_ic_s", False)):
            return S_raw
        gate = t_t / (t_t + float(getattr(self, "t_ic_gate", DEFAULT_T_IC_GATE)))
        s_ic = torch.as_tensor(float(getattr(self, "s_ic_co2", DEFAULT_S_IC_CO2)), device=S_raw.device, dtype=S_raw.dtype)
        return torch.clamp(
            s_ic + gate * (S_raw - s_ic),
            float(getattr(self, "s_eps", DEFAULT_S_EPS)),
            float(getattr(self, "s_co2_max", DEFAULT_S_CO2_MAX)),
        )


class SingleNetPINN(nn.Module, _HardICMixin):
    def __init__(self, cfg: dict[str, Any]):
        super().__init__()
        dtype = dtype_from_cfg(cfg)
        self.use_msfourier = bool(cfg.get("use_msfourier", False))
        self.s_eps = float(cfg.get("S_eps", DEFAULT_S_EPS))
        self.s_co2_max = float(cfg.get("S_CO2_MAX", DEFAULT_S_CO2_MAX))
        self.use_hard_ic_s = bool(cfg.get("use_hard_ic_s", False))
        self.s_ic_co2 = float(cfg.get("s_ic_co2", DEFAULT_S_IC_CO2))
        self.t_ic_gate = float(cfg.get("t_ic_gate", DEFAULT_T_IC_GATE))
        if self.use_msfourier:
            self.ff = MultiScaleFourierFeatures(dtype, 3, cfg["ms_m_per_scale"], cfg["ms_scales"], cfg["ms_base_scale"])
            in_dim = self.ff.out_dim
        else:
            self.ff = None
            in_dim = 3
        self.backbone = MLP(in_dim, 2, width=int(cfg["width"]), depth=8)
        self.beta = 1.0

    def set_beta(self, beta: float):
        self.beta = float(beta)

    def forward(self, x_t, y_t, t_t):
        x_in = 2.0 * x_t - 1.0
        y_in = 2.0 * y_t - 1.0
        t_in = 2.0 * t_t - 1.0
        X = torch.cat([x_in, y_in, t_in], dim=1)
        Z = self.ff(X) if self.use_msfourier else X
        out = self.backbone(Z)
        p_tilde = out[:, 0:1]
        s_logit = out[:, 1:2]
        S_hat = torch.sigmoid(self.beta * s_logit)
        S_raw = self.s_eps + (self.s_co2_max - self.s_eps) * S_hat
        return p_tilde, self.apply_hard_ic(S_raw, t_t)


class TwoNetPINN(nn.Module, _HardICMixin):
    def __init__(self, cfg: dict[str, Any]):
        super().__init__()
        dtype = dtype_from_cfg(cfg)
        self.use_msfourier = bool(cfg.get("use_msfourier", False))
        self.decouple_ps_input = bool(cfg.get("decouple_ps_input", False))
        self.s_eps = float(cfg.get("S_eps", DEFAULT_S_EPS))
        self.s_co2_max = float(cfg.get("S_CO2_MAX", DEFAULT_S_CO2_MAX))
        self.use_hard_ic_s = bool(cfg.get("use_hard_ic_s", False))
        self.s_ic_co2 = float(cfg.get("s_ic_co2", DEFAULT_S_IC_CO2))
        self.t_ic_gate = float(cfg.get("t_ic_gate", DEFAULT_T_IC_GATE))
        self.p_ff = None
        self.s_ff = None
        if self.use_msfourier:
            self.s_ff = MultiScaleFourierFeatures(dtype, 3, cfg["ms_m_per_scale"], cfg["ms_scales"], cfg["ms_base_scale"])
            s_in_dim = self.s_ff.out_dim
        else:
            s_in_dim = 3
        if self.decouple_ps_input:
            p_in_dim = 3
        else:
            self.p_ff = self.s_ff
            p_in_dim = s_in_dim
        self.pnet = MLP(p_in_dim, 1, width=int(cfg["width"]), depth=6)
        self.snet = MLP(s_in_dim, 1, width=int(cfg["width"]), depth=8)
        self.beta = 1.0

    def set_beta(self, beta: float):
        self.beta = float(beta)

    def forward(self, x_t, y_t, t_t):
        x_in = 2.0 * x_t - 1.0
        y_in = 2.0 * y_t - 1.0
        t_in = 2.0 * t_t - 1.0
        X = torch.cat([x_in, y_in, t_in], dim=1)
        Xp = self.p_ff(X) if self.p_ff is not None else X
        Xs = self.s_ff(X) if self.s_ff is not None else X
        p_tilde = self.pnet(Xp)
        s_logit = self.snet(Xs)
        S_hat = torch.sigmoid(self.beta * s_logit)
        S_raw = self.s_eps + (self.s_co2_max - self.s_eps) * S_hat
        return p_tilde, self.apply_hard_ic(S_raw, t_t)


class TwoNetPINNCoarseDetail(nn.Module, _HardICMixin):
    def __init__(self, cfg: dict[str, Any]):
        super().__init__()
        dtype = dtype_from_cfg(cfg)
        self.decouple_ps_input = bool(cfg.get("decouple_ps_input", True))
        self.s_eps = float(cfg.get("S_eps", DEFAULT_S_EPS))
        self.s_co2_max = float(cfg.get("S_CO2_MAX", DEFAULT_S_CO2_MAX))
        self.use_hard_ic_s = bool(cfg.get("use_hard_ic_s", False))
        self.s_ic_co2 = float(cfg.get("s_ic_co2", DEFAULT_S_IC_CO2))
        self.t_ic_gate = float(cfg.get("t_ic_gate", DEFAULT_T_IC_GATE))
        self.detail_tanh = bool(cfg.get("e4_detail_tanh", DEFAULT_M3_DETAIL_TANH))
        self.detail_gain = float(cfg.get("e4_detail_gain", DEFAULT_M3_DETAIL_GAIN))
        self.p_ff = None
        if self.decouple_ps_input:
            p_in_dim = 3
        else:
            self.p_ff = MultiScaleFourierFeatures(dtype, 3, cfg["ms_m_per_scale"], cfg["ms_scales"], cfg["ms_base_scale"])
            p_in_dim = self.p_ff.out_dim
        self.s_coarse_ff = MultiScaleFourierFeatures(dtype, 3, cfg["e4_coarse_m_per_scale"], cfg["e4_coarse_scales"], cfg["e4_coarse_base_scale"])
        self.s_detail_ff = MultiScaleFourierFeatures(dtype, 3, cfg["e4_detail_m_per_scale"], cfg["e4_detail_scales"], cfg["e4_detail_base_scale"])
        self.pnet = MLP(p_in_dim, 1, width=int(cfg["width"]), depth=6)
        self.snet_coarse = MLP(self.s_coarse_ff.out_dim, 1, width=int(cfg["width"]), depth=int(cfg["e4_coarse_depth"]))
        self.snet_detail = MLP(self.s_detail_ff.out_dim, 1, width=int(cfg["width"]), depth=int(cfg["e4_detail_depth"]))
        self.beta = 1.0

    def set_beta(self, beta: float):
        self.beta = float(beta)

    def forward(self, x_t, y_t, t_t):
        x_in = 2.0 * x_t - 1.0
        y_in = 2.0 * y_t - 1.0
        t_in = 2.0 * t_t - 1.0
        X = torch.cat([x_in, y_in, t_in], dim=1)
        Xp = self.p_ff(X) if self.p_ff is not None else X
        Zc = self.s_coarse_ff(X)
        Zd = self.s_detail_ff(X)
        p_tilde = self.pnet(Xp)
        s_coarse = self.snet_coarse(Zc)
        s_detail = self.snet_detail(Zd)
        s_detail_term = torch.tanh(s_detail) if self.detail_tanh else s_detail
        s_logit = s_coarse + self.detail_gain * s_detail_term
        S_hat = torch.sigmoid(self.beta * s_logit)
        S_raw = self.s_eps + (self.s_co2_max - self.s_eps) * S_hat
        return p_tilde, self.apply_hard_ic(S_raw, t_t)


# =============================================================================
# Configuration from experiment/checkpoint
# =============================================================================
def default_cfg() -> dict[str, Any]:
    return dict(
        model_type="two", width=DEFAULT_WIDTH, dtype=DEFAULT_DTYPE,
        use_msfourier=False, decouple_ps_input=False, use_coarse_detail_s_branch=False,
        ms_m_per_scale=DEFAULT_MS_M_PER_SCALE, ms_scales=DEFAULT_MS_SCALES, ms_base_scale=DEFAULT_MS_BASE_SCALE,
        e4_coarse_m_per_scale=DEFAULT_M3_COARSE_M_PER_SCALE, e4_coarse_scales=DEFAULT_M3_COARSE_SCALES, e4_coarse_base_scale=DEFAULT_M3_COARSE_BASE_SCALE,
        e4_detail_m_per_scale=DEFAULT_M3_DETAIL_M_PER_SCALE, e4_detail_scales=DEFAULT_M3_DETAIL_SCALES, e4_detail_base_scale=DEFAULT_M3_DETAIL_BASE_SCALE,
        e4_coarse_depth=DEFAULT_M3_COARSE_DEPTH, e4_detail_depth=DEFAULT_M3_DETAIL_DEPTH, e4_detail_gain=DEFAULT_M3_DETAIL_GAIN, e4_detail_tanh=DEFAULT_M3_DETAIL_TANH,
        beta_eval=DEFAULT_BETA_EVAL, S_eps=DEFAULT_S_EPS, S_CO2_MAX=DEFAULT_S_CO2_MAX,
        use_hard_ic_s=False, s_ic_co2=DEFAULT_S_IC_CO2, t_ic_gate=DEFAULT_T_IC_GATE,
        L_ref=DEFAULT_L_REF, T_ref=DEFAULT_T_REF, mu_ref=DEFAULT_MU_REF, U_ref=DEFAULT_U_REF, k_ref=DEFAULT_K_REF,
        p0=DEFAULT_P0, P_ref_override=DEFAULT_P_REF_OVERRIDE,
        Sw_irr=DEFAULT_SW_IRR, Snr=DEFAULT_SNR,
        front_low=DEFAULT_FRONT_LOW, front_high=DEFAULT_FRONT_HIGH,
        front_center=DEFAULT_FRONT_CENTER, front_sigma=DEFAULT_FRONT_SIGMA, plume_threshold=DEFAULT_PLUME_THRESHOLD,
        background_threshold=DEFAULT_BACKGROUND_THRESHOLD, core_threshold=DEFAULT_CORE_THRESHOLD,
    )


def apply_exp_preset(cfg: dict[str, Any], exp_name: str):
    exp_name = str(exp_name).upper()
    n = int(exp_name[1:]) if re.fullmatch(r"M\d+", exp_name) else 0
    if n == 0:
        cfg.update(model_type="single", use_msfourier=False, decouple_ps_input=False, use_coarse_detail_s_branch=False, use_hard_ic_s=False)
    elif n == 1:
        cfg.update(model_type="two", use_msfourier=False, decouple_ps_input=False, use_coarse_detail_s_branch=False, use_hard_ic_s=False)
    elif n == 2:
        cfg.update(model_type="two", use_msfourier=True, decouple_ps_input=True, use_coarse_detail_s_branch=False, use_hard_ic_s=False)
    elif n >= 3:
        cfg.update(model_type="two_cd", use_msfourier=True, decouple_ps_input=True, use_coarse_detail_s_branch=True, use_hard_ic_s=False)
    cfg["P_ref"] = cfg.get("P_ref_override") or (float(cfg["mu_ref"]) * float(cfg["U_ref"]) * float(cfg["L_ref"]) / float(cfg["k_ref"]))


def load_checkpoint(ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        return ckpt["state_dict"], ckpt.get("config", {})
    return ckpt, {}


def cfg_update_from_ckpt(cfg: dict[str, Any], ckpt_cfg: dict[str, Any] | None):
    if not isinstance(ckpt_cfg, dict):
        return
    cfg["checkpoint_config"] = ckpt_cfg
    explicit_s_co2_max = False

    # Flat/top-level compatibility.
    flat_map = {
        "width": "width", "dtype": "dtype", "P_ref": "P_ref_override", "p0": "p0",
        "L_ref": "L_ref", "T_ref": "T_ref", "mu_ref": "mu_ref", "U_ref": "U_ref", "k_ref": "k_ref",
        "S_eps": "S_eps", "saturation_eps": "S_eps", "Snr": "Snr", "Sw_irr": "Sw_irr",
    }
    for src, dst in flat_map.items():
        if src in ckpt_cfg:
            cfg[dst] = ckpt_cfg[src]

    # Nested architecture block saved by M0-M7 training script.
    arch = ckpt_cfg.get("architecture", {})
    if isinstance(arch, dict):
        cfg["use_msfourier"] = bool(arch.get("use_msfourier", cfg.get("use_msfourier", False)))
        cfg["decouple_ps_input"] = bool(arch.get("decouple_ps_input", cfg.get("decouple_ps_input", False)))
        cfg["use_coarse_detail_s_branch"] = bool(arch.get("use_coarse_detail_s_branch", cfg.get("use_coarse_detail_s_branch", False)))
        if cfg["use_coarse_detail_s_branch"]:
            cfg["model_type"] = "two_cd"
        elif bool(arch.get("use_twonet", cfg.get("model_type") == "two")):
            cfg["model_type"] = "two"
        else:
            cfg["model_type"] = "single"
        if "msff_m_per_scale" in arch:
            cfg["ms_m_per_scale"] = int(arch["msff_m_per_scale"])
        if "msff_scales" in arch:
            cfg["ms_scales"] = tuple(float(v) for v in arch["msff_scales"])
        if "msff_base_scale" in arch:
            cfg["ms_base_scale"] = float(arch["msff_base_scale"])
        m3 = arch.get("m3_coarse_detail", {})
        if isinstance(m3, dict):
            pairs = {
                "coarse_m_per_scale": "e4_coarse_m_per_scale",
                "coarse_scales": "e4_coarse_scales",
                "coarse_base_scale": "e4_coarse_base_scale",
                "detail_m_per_scale": "e4_detail_m_per_scale",
                "detail_scales": "e4_detail_scales",
                "detail_base_scale": "e4_detail_base_scale",
                "coarse_depth": "e4_coarse_depth",
                "detail_depth": "e4_detail_depth",
                "detail_gain": "e4_detail_gain",
                "detail_tanh": "e4_detail_tanh",
            }
            for k, dst in pairs.items():
                if k in m3:
                    cfg[dst] = tuple(float(x) for x in m3[k]) if "scales" in k else m3[k]

    # Saturation IC/output bounds saved by the M0-M7 training script.
    sat_ic = ckpt_cfg.get("saturation_ic", {})
    if isinstance(sat_ic, dict):
        if "s_ic_co2" in sat_ic:
            cfg["s_ic_co2"] = float(sat_ic["s_ic_co2"])
        if "s_inj_co2" in sat_ic:
            cfg["s_inj_co2"] = float(sat_ic["s_inj_co2"])
        if "s_co2_output_min" in sat_ic:
            cfg["S_eps"] = float(sat_ic["s_co2_output_min"])
        if "s_co2_output_max" in sat_ic:
            cfg["S_CO2_MAX"] = float(sat_ic["s_co2_output_max"])
            explicit_s_co2_max = True

    # Legacy checkpoints may still contain the removed IC-gating block; new
    # M0-M7 checkpoints do not, and M4 now means front/plume loss.
    legacy_hard_ic = ckpt_cfg.get("m4_hard_ic", {})
    if isinstance(legacy_hard_ic, dict):
        cfg["use_hard_ic_s"] = bool(legacy_hard_ic.get("enabled", cfg.get("use_hard_ic_s", False)))
        if "s_ic_co2" in legacy_hard_ic:
            cfg["s_ic_co2"] = float(legacy_hard_ic["s_ic_co2"])
        if "t_ic_gate" in legacy_hard_ic:
            cfg["t_ic_gate"] = float(legacy_hard_ic["t_ic_gate"])

    # M4 front/plume region settings.
    m4_front = ckpt_cfg.get("m4_front_plume_loss", {})
    if isinstance(m4_front, dict):
        if "front_s_min" in m4_front:
            cfg["front_low"] = float(m4_front["front_s_min"])
        if "front_s_max" in m4_front:
            cfg["front_high"] = float(m4_front["front_s_max"])
        if "front_center" in m4_front:
            cfg["front_center"] = float(m4_front["front_center"])
        if "front_sigma" in m4_front:
            cfg["front_sigma"] = float(m4_front["front_sigma"])
        if "plume_threshold" in m4_front:
            cfg["plume_threshold"] = float(m4_front["plume_threshold"])

    # M5 pairwise front-gradient settings.
    m5_pair = ckpt_cfg.get("m5_pairwise_front_gradient", {})
    if isinstance(m5_pair, dict):
        for src, dst in {
            "s_min": "pairgrad_s_min",
            "s_max": "pairgrad_s_max",
            "front_center": "pairgrad_front_center",
            "front_sigma": "pairgrad_front_sigma",
            "front_weight_floor": "pairgrad_front_weight_floor",
            "min_dist_tilde": "pairgrad_min_dist",
            "max_dist_tilde": "pairgrad_max_dist",
        }.items():
            if src in m5_pair:
                cfg[dst] = float(m5_pair[src])

    # Training schedules: final beta only matters for BETA/BETA_DIFF checkpoints.
    sched = ckpt_cfg.get("training_schedules", {})
    if isinstance(sched, dict):
        if bool(sched.get("use_beta_curriculum", False)):
            cfg["beta_eval"] = float(sched.get("beta_max", cfg.get("beta_eval", 1.0)))
        else:
            cfg["beta_eval"] = 1.0

    phys = ckpt_cfg.get("physical_parameters", {})
    if isinstance(phys, dict):
        for src, dst in {
            "L_ref": "L_ref", "T_ref": "T_ref", "mu_ref": "mu_ref", "U_ref": "U_ref", "k_ref": "k_ref",
            "p0": "p0", "P_ref": "P_ref_override", "Sw_irr": "Sw_irr", "Snr": "Snr",
            "S_ic_co2": "s_ic_co2", "S_IC_CO2": "s_ic_co2",
            "S_inj_co2": "s_inj_co2", "S_CO2_MAX": "S_CO2_MAX",
        }.items():
            if src in phys:
                cfg[dst] = phys[src]
                if src == "S_CO2_MAX":
                    explicit_s_co2_max = True
    if (not explicit_s_co2_max) or cfg.get("S_CO2_MAX") is None:
        cfg["S_CO2_MAX"] = 1.0 - float(cfg.get("Sw_irr", DEFAULT_SW_IRR))

    cfg["P_ref"] = float(cfg["P_ref_override"]) if cfg.get("P_ref_override", None) is not None else (
        float(cfg["mu_ref"]) * float(cfg["U_ref"]) * float(cfg["L_ref"]) / float(cfg["k_ref"])
    )


def build_model(cfg: dict[str, Any]) -> nn.Module:
    if bool(cfg.get("use_coarse_detail_s_branch", False)) or cfg.get("model_type") == "two_cd":
        return TwoNetPINNCoarseDetail(cfg)
    if cfg.get("model_type") == "two":
        return TwoNetPINN(cfg)
    return SingleNetPINN(cfg)


# =============================================================================
# Dataset and run discovery
# =============================================================================
def load_dataset_pack(data_path: str):
    pack = torch.load(data_path, map_location="cpu", weights_only=False)
    if not isinstance(pack, dict) or "arrays" not in pack:
        raise ValueError(f"{data_path} does not look like a dataset pack with key 'arrays'.")
    arrays = pack["arrays"]
    for k in ("x", "y", "t", "p", "Sco2"):
        if k not in arrays:
            raise ValueError(f"Dataset missing required array: {k}")
    time_unique = pack.get("time_unique", None)
    if time_unique is None:
        time_unique = np.unique(to_np_1d(arrays["t"]))
    else:
        time_unique = to_np_1d(time_unique)
    return arrays, np.asarray(time_unique, dtype=np.float64)


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    exp_name: str
    training_strategy: str
    leave_out: str
    seed: str
    ckpt_path: str


def parse_run_from_filename(path: str) -> tuple[str | None, str, str, str]:
    name = os.path.basename(path)
    exp = None
    m = re.search(r"model_(M\d+)", name)
    if m:
        exp = m.group(1)
    strategy = "UNKNOWN"
    m = re.search(r"model_M\d+_([A-Z_]+?)(?:_LOO-|_seed)", name)
    if m:
        strategy = m.group(1)
    leave = "NONE"
    m = re.search(r"_LOO-([A-Za-z0-9_]+)_seed", name)
    if m:
        leave = m.group(1)
    seed = "NA"
    m = re.search(r"_seed(\d+)_", name)
    if m:
        seed = m.group(1)
    return exp, strategy, leave, seed


def find_checkpoint_for_run(ckpt_dir: str, exp: str, strategy: str, leave_out: str, seed: str | None = None) -> str | None:
    files = glob.glob(os.path.join(ckpt_dir, "*.pt"))
    cands = []
    for p in files:
        e, st, lo, sd = parse_run_from_filename(p)
        if e != exp:
            continue
        if strategy and strategy != "ANY" and st != strategy:
            continue
        if leave_out and leave_out != "ANY" and lo != leave_out:
            continue
        if seed is not None and seed != "ANY" and sd != str(seed):
            continue
        cands.append(p)
    if not cands:
        return None

    def score(p):
        name = os.path.basename(p).lower()
        return (
            1 if "final" in name else 0,
            1 if "seed0" in name else 0,
            os.path.getmtime(p),
        )
    return sorted(cands, key=score, reverse=True)[0]


def make_run_specs(ckpt_dir: str, experiments: list[str], strategy: str, include_loo: bool, loo_exp: str, leave_outs: list[str], seed: str | None) -> list[RunSpec]:
    specs: list[RunSpec] = []
    seen = set()
    for exp in experiments:
        ckpt = find_checkpoint_for_run(ckpt_dir, exp, strategy, "NONE", seed)
        if ckpt is None:
            print(f"[WARN] no checkpoint found for {exp} strategy={strategy} LOO=NONE")
            continue
        e, st, lo, sd = parse_run_from_filename(ckpt)
        run_id = f"{e}_{st}_LOO-{lo}_seed{sd}"
        if run_id not in seen:
            specs.append(RunSpec(run_id, e or exp, st, lo, sd, ckpt)); seen.add(run_id)
    if include_loo:
        for lo in leave_outs:
            ckpt = find_checkpoint_for_run(ckpt_dir, loo_exp, strategy, lo, seed)
            if ckpt is None:
                print(f"[WARN] no checkpoint found for {loo_exp} strategy={strategy} LOO={lo}")
                continue
            e, st, lo2, sd = parse_run_from_filename(ckpt)
            run_id = f"{e}_{st}_LOO-{lo2}_seed{sd}"
            if run_id not in seen:
                specs.append(RunSpec(run_id, e or loo_exp, st, lo2, sd, ckpt)); seen.add(run_id)
    return specs


def build_eval_plan(arrays: dict[str, Any], time_unique: np.ndarray, sample_per_time: int, max_time_steps: int | None, max_sort_n: int, seed: int, time_spec: str):
    x = to_np_1d(arrays["x"]).astype(np.float64, copy=False)
    y = to_np_1d(arrays["y"]).astype(np.float64, copy=False)
    t = to_np_1d(arrays["t"]).astype(np.float64, copy=False)
    p = to_np_1d(arrays["p"]).astype(np.float64, copy=False)
    s = np.clip(to_np_1d(arrays["Sco2"]).astype(np.float64, copy=False), 0.0, 1.0)

    tu = np.asarray(time_unique, dtype=np.float64).copy()
    tu.sort()
    if max_time_steps is not None:
        tu = tu[: int(max_time_steps)]
    Tn = int(tu.shape[0])
    if x.shape[0] > max_sort_n:
        raise MemoryError(f"N={x.shape[0]} exceeds --max-sort-n={max_sort_n}")
    rng = np.random.default_rng(seed)

    # Exact time bucket assignment: use nearest sorted unique time.
    pos = np.searchsorted(tu, t)
    pos = np.clip(pos, 0, Tn - 1)
    prev = np.clip(pos - 1, 0, Tn - 1)
    choose_prev = np.abs(t - tu[prev]) < np.abs(t - tu[pos])
    time_id = np.where(choose_prev, prev, pos)

    order = np.argsort(time_id, kind="mergesort")
    counts = np.bincount(time_id, minlength=Tn)
    offsets = np.concatenate([[0], np.cumsum(counts)])
    indices_per_time = []
    pooled = []
    for k in range(Tn):
        seg = order[offsets[k]:offsets[k + 1]]
        if seg.size == 0:
            indices_per_time.append(np.empty((0,), dtype=np.int64))
            continue
        if sample_per_time <= 0 or sample_per_time >= seg.size:
            idx = seg.astype(np.int64, copy=False)
        else:
            idx = rng.choice(seg, size=int(sample_per_time), replace=False).astype(np.int64)
            idx.sort()
        indices_per_time.append(idx)
        pooled.append(idx)

    selected_items = resolve_snapshot_items(tu, parse_time_indices(time_spec))
    selected_map = {idx: label for label, idx in selected_items}

    return dict(x=x, y=y, t=t, p=p, s=s, time_seconds=tu, indices_per_time=indices_per_time, pooled_idx=np.concatenate(pooled) if pooled else np.empty(0, dtype=np.int64), selected_items=selected_items, selected_map=selected_map)


# =============================================================================
# Metrics
# =============================================================================
def regression_basic(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    true = np.asarray(true, dtype=np.float64).reshape(-1)
    if pred.size == 0:
        return dict(
            mse=np.nan, rmse=np.nan, mae=np.nan, rel_l2=np.nan, maxae=np.nan,
            bias=np.nan, p95ae=np.nan, p99ae=np.nan, nrmse_range=np.nan,
            nrmse_iqr=np.nan, r2=np.nan, corr=np.nan,
        )
    err = pred - true
    ae = np.abs(err)
    mse = float(np.mean(err * err))
    true_range = float(np.max(true) - np.min(true)) if true.size else np.nan
    q25, q75 = np.percentile(true, [25.0, 75.0]) if true.size else (np.nan, np.nan)
    true_iqr = float(q75 - q25)
    ss_res = float(np.sum(err * err))
    ss_tot = float(np.sum((true - float(np.mean(true))) ** 2))
    if pred.size >= 2 and np.std(pred) > 0.0 and np.std(true) > 0.0:
        corr = float(np.corrcoef(pred, true)[0, 1])
    else:
        corr = np.nan
    return dict(
        mse=mse,
        rmse=float(math.sqrt(mse)),
        mae=float(np.mean(ae)),
        rel_l2=float(np.sqrt(ss_res) / (np.sqrt(np.sum(true * true)) + 1.0e-30)),
        maxae=float(np.max(ae)),
        bias=float(np.mean(err)),
        p95ae=float(np.percentile(ae, 95.0)),
        p99ae=float(np.percentile(ae, 99.0)),
        nrmse_range=float(math.sqrt(mse) / (true_range + 1.0e-30)) if true_range > 0.0 else np.nan,
        nrmse_iqr=float(math.sqrt(mse) / (true_iqr + 1.0e-30)) if true_iqr > 0.0 else np.nan,
        r2=float(1.0 - ss_res / (ss_tot + 1.0e-30)) if ss_tot > 0.0 else np.nan,
        corr=corr,
    )


def prefixed(prefix: str, met: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{k}": v for k, v in met.items()}


@torch.no_grad()
def predict_model(model: nn.Module, x_m: np.ndarray, y_m: np.ndarray, t_s: np.ndarray, cfg: dict[str, Any], device: torch.device, batch_size: int):
    L_ref = float(cfg["L_ref"])
    T_ref = float(cfg["T_ref"])
    infer_dtype = dtype_from_cfg(cfg)
    N = int(x_m.shape[0])
    p_out = np.empty(N, dtype=np.float64)
    s_out = np.empty(N, dtype=np.float64)
    for i0 in range(0, N, batch_size):
        i1 = min(N, i0 + batch_size)
        xb = torch.from_numpy(x_m[i0:i1]).to(device=device, dtype=infer_dtype).view(-1, 1) / L_ref
        yb = torch.from_numpy(y_m[i0:i1]).to(device=device, dtype=infer_dtype).view(-1, 1) / L_ref
        tb = torch.from_numpy(t_s[i0:i1]).to(device=device, dtype=infer_dtype).view(-1, 1) / T_ref
        pt, ss = model(xb, yb, tb)
        p_out[i0:i1] = pt.detach().cpu().to(torch.float64).numpy().reshape(-1)
        s_out[i0:i1] = ss.detach().cpu().to(torch.float64).numpy().reshape(-1)
    return p_out, s_out


def summarize_region_s(err_pred: np.ndarray, s_true: np.ndarray, cfg: dict[str, Any]) -> dict[str, float]:
    """Region-wise saturation diagnostics.

    The raw Sco2 field is still compared directly against raw Sco2 data.
    For nonzero initial CO2 saturation, plume/background
    diagnostics based on raw S>0.05 are not physically meaningful because the
    whole initial domain already satisfies that condition. Therefore this
    function reports both legacy raw-S regions and preferred excess-saturation
    regions using dS = S - S_IC_CO2.
    """
    s_pred = err_pred
    out = {}
    front_low = float(cfg.get("front_low", DEFAULT_FRONT_LOW))
    front_high = float(cfg.get("front_high", DEFAULT_FRONT_HIGH))
    s_ic = float(cfg.get("s_ic_co2", DEFAULT_S_IC_CO2))
    dS_true = s_true - s_ic
    dS_pred = s_pred - s_ic
    excess_plume_thr = float(cfg.get("excess_plume_threshold", DEFAULT_EXCESS_PLUME_THRESHOLD))
    excess_mobile_thr = float(cfg.get("excess_mobile_threshold", DEFAULT_EXCESS_MOBILE_THRESHOLD))
    excess_bg_thr = float(cfg.get("excess_background_threshold", DEFAULT_EXCESS_BACKGROUND_THRESHOLD))
    excess_lead_center = float(cfg.get("excess_leading_band_center", DEFAULT_EXCESS_LEADING_BAND_CENTER))
    excess_lead_half = float(cfg.get("excess_leading_band_half_width", DEFAULT_EXCESS_LEADING_BAND_HALF_WIDTH))
    masks = {
        # Raw-S diagnostic regions retained for backward compatibility.
        "front_band": (s_true >= front_low) & (s_true <= front_high),
        "plume_rawS005": s_true > float(cfg.get("plume_threshold", DEFAULT_PLUME_THRESHOLD)),
        "background_rawS002": s_true < float(cfg.get("background_threshold", DEFAULT_BACKGROUND_THRESHOLD)),
        "core": s_true > float(cfg.get("core_threshold", DEFAULT_CORE_THRESHOLD)),
        "near_sic": np.abs(s_true - s_ic) <= 0.02,
        "mobile_co2_training_raw": s_true > max(float(cfg.get("Snr", DEFAULT_SNR)), s_ic),
        "mobile_co2_raw": s_true > max(float(cfg.get("Snr", DEFAULT_SNR)), s_ic) + 0.02,
        # dS-based regions; with the current S_IC_CO2=0.0 training default
        # these coincide with raw-S regions while keeping future nonzero-IC
        # evaluations explicit.
        "excess_plume_dS005": dS_true > excess_plume_thr,
        "excess_mobile_dS002": dS_true > excess_mobile_thr,
        "excess_background_dS001": dS_true < excess_bg_thr,
        "excess_leading_band_dS005": np.abs(dS_true - excess_lead_center) <= excess_lead_half,
    }
    for name, mask in masks.items():
        if np.any(mask):
            met_raw = regression_basic(s_pred[mask], s_true[mask])
            met_excess = regression_basic(dS_pred[mask], dS_true[mask])
            # Raw and dS errors are numerically identical for additive S_IC, but
            # both names make downstream attribution explicit.
            out[f"S_RMSE_{name}"] = met_raw["rmse"]
            out[f"S_MAE_{name}"] = met_raw["mae"]
            out[f"S_Bias_{name}"] = met_raw["bias"]
            out[f"dS_RMSE_{name}"] = met_excess["rmse"]
            out[f"dS_MAE_{name}"] = met_excess["mae"]
            out[f"dS_Bias_{name}"] = met_excess["bias"]
            out[f"n_{name}"] = int(np.sum(mask))
        else:
            out[f"S_RMSE_{name}"] = np.nan
            out[f"S_MAE_{name}"] = np.nan
            out[f"S_Bias_{name}"] = np.nan
            out[f"dS_RMSE_{name}"] = np.nan
            out[f"dS_MAE_{name}"] = np.nan
            out[f"dS_Bias_{name}"] = np.nan
            out[f"n_{name}"] = 0
    return out


def evaluate_run(
    run: RunSpec,
    eval_plan: dict[str, Any],
    device: torch.device,
    batch_size: int,
    pressure_unit: str,
    s_ic_co2: float = DEFAULT_S_IC_CO2,
    Snr: float = DEFAULT_SNR,
    Sw_irr: float = DEFAULT_SW_IRR,
):
    state, ckpt_cfg = load_checkpoint(run.ckpt_path)
    cfg = default_cfg()
    # CLI values are fallbacks for legacy checkpoints; checkpoint config wins.
    cfg["s_ic_co2"] = float(s_ic_co2)
    cfg["Snr"] = float(Snr)
    cfg["Sw_irr"] = float(Sw_irr)
    apply_exp_preset(cfg, run.exp_name)
    cfg_update_from_ckpt(cfg, ckpt_cfg)
    cfg["P_ref"] = float(cfg.get("P_ref_override")) if cfg.get("P_ref_override", None) is not None else (
        float(cfg["mu_ref"]) * float(cfg["U_ref"]) * float(cfg["L_ref"]) / float(cfg["k_ref"])
    )

    model = build_model(cfg)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        # strict=False gives a clearer message when old checkpoints have extra metadata buffers.
        print(f"[WARN] {run.run_id}: missing={missing}, unexpected={unexpected}")
        if missing:
            raise RuntimeError(f"State_dict missing required keys for {run.run_id}: {missing}")
    model.to(device)
    model.eval()
    model.set_beta(float(cfg.get("beta_eval", 1.0)))

    P_ref = float(cfg["P_ref"])
    p0 = float(cfg["p0"])
    pf = pressure_unit_factor(pressure_unit)

    summary_acc = []
    time_rows = []

    # Accumulators for overall metrics.
    all_err_pt, all_true_pt = [], []
    all_err_pp, all_true_pp = [], []
    all_err_s, all_true_s = [], []
    all_pred_s = []

    for k, idx in enumerate(eval_plan["indices_per_time"]):
        if idx.size == 0:
            continue
        x = eval_plan["x"][idx]
        y = eval_plan["y"][idx]
        t = eval_plan["t"][idx]
        p_true_pa = eval_plan["p"][idx]
        p_true_tilde = (p_true_pa - p0) / P_ref
        s_true = eval_plan["s"][idx]
        p_pred_tilde, s_pred = predict_model(model, x, y, t, cfg, device, batch_size)
        p_pred_pa = p0 + P_ref * p_pred_tilde

        met_pt = regression_basic(p_pred_tilde, p_true_tilde)
        met_pp = regression_basic(p_pred_pa, p_true_pa)
        met_s = regression_basic(s_pred, s_true)
        reg_s = summarize_region_s(s_pred, s_true, cfg)

        all_err_pt.append(p_pred_tilde - p_true_tilde); all_true_pt.append(p_true_tilde)
        all_err_pp.append(p_pred_pa - p_true_pa); all_true_pp.append(p_true_pa)
        all_err_s.append(s_pred - s_true); all_true_s.append(s_true); all_pred_s.append(s_pred)

        row = dict(
            run_id=run.run_id, exp_name=run.exp_name, training_strategy=run.training_strategy,
            leave_out=run.leave_out, seed=run.seed, ckpt_path=run.ckpt_path,
            time_index=k, snapshot_label=eval_plan["selected_map"].get(k, ""),
            time_s=float(eval_plan["time_seconds"][k]),
            time_days=float(eval_plan["time_seconds"][k] / 86400.0),
            n_used=int(idx.size),
        )
        row.update(prefixed("p_tilde", met_pt))
        row.update(prefixed("p_phys_Pa", met_pp))
        row["p_phys_RMSE_MPa"] = met_pp["rmse"] * 1e-6
        row["p_phys_MAE_MPa"] = met_pp["mae"] * 1e-6
        row.update(prefixed("S", met_s))
        row.update(reg_s)
        # IC diagnostic: only meaningful at first time layer, but convenient to include.
        if k == 0:
            row["IC_S_RMSE_vs_S_IC_CO2"] = float(np.sqrt(np.mean((s_pred - float(cfg.get("s_ic_co2", DEFAULT_S_IC_CO2))) ** 2)))
            row["IC_S_Bias_vs_S_IC_CO2"] = float(np.mean(s_pred - float(cfg.get("s_ic_co2", DEFAULT_S_IC_CO2))))
        else:
            row["IC_S_RMSE_vs_S_IC_CO2"] = np.nan
            row["IC_S_Bias_vs_S_IC_CO2"] = np.nan
        time_rows.append(row)

    def concat(xs):
        return np.concatenate(xs) if xs else np.empty(0, dtype=np.float64)

    true_pt = concat(all_true_pt); pred_pt = true_pt + concat(all_err_pt)
    true_pp = concat(all_true_pp); pred_pp = true_pp + concat(all_err_pp)
    true_s = concat(all_true_s); pred_s = concat(all_pred_s)

    met_pt_all = regression_basic(pred_pt, true_pt)
    met_pp_all = regression_basic(pred_pp, true_pp)
    met_s_all = regression_basic(pred_s, true_s)
    region_all = summarize_region_s(pred_s, true_s, cfg)

    # IC metrics from by-time first layer.
    ic_rows = [r for r in time_rows if int(r["time_index"]) == 0]
    ic_rmse = ic_rows[0].get("IC_S_RMSE_vs_S_IC_CO2", np.nan) if ic_rows else np.nan
    ic_bias = ic_rows[0].get("IC_S_Bias_vs_S_IC_CO2", np.nan) if ic_rows else np.nan

    summary = dict(
        run_id=run.run_id, exp_name=run.exp_name, training_strategy=run.training_strategy,
        leave_out=run.leave_out, seed=run.seed, ckpt_path=run.ckpt_path,
        n_times=len([r for r in time_rows if r.get("n_used", 0) > 0]),
        n_points=int(sum(int(r["n_used"]) for r in time_rows)),
        model_type=cfg.get("model_type"), use_msfourier=bool(cfg.get("use_msfourier", False)),
        decouple_ps_input=bool(cfg.get("decouple_ps_input", False)),
        use_coarse_detail=bool(cfg.get("use_coarse_detail_s_branch", False)),
        use_hard_ic_s=bool(cfg.get("use_hard_ic_s", False)),
        S_IC_CO2=float(cfg.get("s_ic_co2", DEFAULT_S_IC_CO2)),
        Snr=float(cfg.get("Snr", DEFAULT_SNR)),
        Sw_irr=float(cfg.get("Sw_irr", DEFAULT_SW_IRR)),
        beta_eval=float(cfg.get("beta_eval", 1.0)),
        P_ref=float(cfg.get("P_ref", np.nan)),
        p0=float(cfg.get("p0", np.nan)),
        IC_S_RMSE_vs_S_IC_CO2=ic_rmse,
        IC_S_Bias_vs_S_IC_CO2=ic_bias,
    )
    summary.update(prefixed("p_tilde", met_pt_all))
    summary.update(prefixed("p_phys_Pa", met_pp_all))
    summary["p_phys_RMSE_MPa"] = met_pp_all["rmse"] * 1e-6
    summary["p_phys_MAE_MPa"] = met_pp_all["mae"] * 1e-6
    summary.update(prefixed("S", met_s_all))
    summary.update(region_all)

    selected_rows = [r for r in time_rows if r.get("snapshot_label", "")]
    return summary, time_rows, selected_rows, cfg


# =============================================================================
# I/O and plotting
# =============================================================================
def write_csv(path: str, rows: list[dict[str, Any]]):
    ensure_dir(os.path.dirname(path))
    if not rows:
        return
    keys = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                keys.append(k); seen.add(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def set_journal_style():
    """Publication-oriented matplotlib defaults.

    The settings are intentionally conservative: small font sizes, thin axes,
    and enough room for external legends.  The actual legend strings are kept
    compact by using plot_label rather than run_id.
    """
    setup_nature_rcparams(font_size=7.0, line_width=1.1)


def _savefig(fig, base: str, dpi: int, export_pdf: bool, export_tiff: bool):
    ensure_dir(os.path.dirname(base))
    save_pub_figure(fig, base, dpi=dpi, export_pdf=export_pdf, export_tiff=export_tiff, export_svg=EXPORT_SVG, export_png=True, pad_inches=0.035)


def _short_label_from_fields(exp_name: str, leave_out: str, mode: str = "auto", has_forward_context: bool = True) -> str:
    """Return compact figure label.

    mode="auto":
      - forward M0-M7 runs are labelled M0-M7;
      - leave-one-out runs with leave_out != NONE are labelled L1-L6;
      - if the run set contains only M7 LOO runs, LOO-NONE is labelled L0.
    mode="exp": always labels non-LOO runs by M*, and LOO removals by L*.
    mode="loo": labels M7 LOO-NONE as L0.
    """
    exp = str(exp_name or "").upper()
    leave = str(leave_out or "NONE").upper()
    mode = str(mode or "auto").lower()
    if mode == "loo" and exp == "M7":
        return LOO_LABELS.get(leave, leave)
    if leave != "NONE":
        return LOO_LABELS.get(leave, leave)
    if mode == "auto" and (not has_forward_context) and exp == "M7":
        return LOO_LABELS.get(leave, "L0")
    return exp if exp else "run"


def assign_plot_labels(rows: list[dict[str, Any]], label_mode: str = "auto") -> list[dict[str, Any]]:
    """Add compact plot_label and plot_group to rows without changing run_id.

    CSV files keep run_id for traceability, but all figures use plot_label.
    """
    if not rows:
        return rows
    exps = {str(r.get("exp_name", "")).upper() for r in rows if r.get("exp_name", "")}
    leaves = {str(r.get("leave_out", "NONE")).upper() for r in rows}
    has_forward_context = any(e != "M7" for e in exps) or ("NONE" in leaves and len(exps) > 1)
    for r in rows:
        exp = str(r.get("exp_name", "")).upper()
        leave = str(r.get("leave_out", "NONE")).upper()
        lab = _short_label_from_fields(exp, leave, label_mode, has_forward_context=has_forward_context)
        r["plot_label"] = lab
        r["plot_group"] = "LOO" if lab.startswith("L") else "M"
    return rows


def _dedup_legend(handles, labels):
    seen = set()
    h2, l2 = [], []
    for h, lab in zip(handles, labels):
        if lab in seen or lab == "_nolegend_":
            continue
        seen.add(lab)
        h2.append(h); l2.append(lab)
    return h2, l2


def _legend_ncol(n: int) -> int:
    if n <= 6:
        return n
    if n <= 10:
        return 5
    return 6


def _resolve_col(df, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _plot_grouped_lines(ax, df, y_col: str, time_unit: str):
    """Plot one curve per run using compact legend labels.

    Multiple seeds for the same plot_label are allowed. The legend is de-duplicated
    so the visible legend remains M0-M7 or L0-L6 only.
    """
    if "plot_label" not in df.columns:
        df = df.copy()
        df["plot_label"] = df["run_id"]
    plotted = {}
    for run_id, g in df.groupby("run_id", sort=False):
        lab = str(g["plot_label"].iloc[0])
        tdisp = [time_to_display(v, time_unit) for v in g["time_s"].to_numpy(dtype=float)]
        y = g[y_col].to_numpy(dtype=float)
        plotted[lab] = (np.asarray(tdisp, dtype=float), y)
        ax.plot(tdisp, y, label=lab, color=color_for_label(lab, None))
    return plotted


def plot_global_curves(by_time_rows: list[dict[str, Any]], out_dir: str, time_unit: str, pressure_unit: str, dpi: int, export_pdf: bool, export_tiff: bool, label_mode: str = "auto"):
    import pandas as pd
    if not by_time_rows:
        return
    set_journal_style()
    rows = [dict(r) for r in by_time_rows]
    assign_plot_labels(rows, label_mode=label_mode)
    df = pd.DataFrame(rows)
    fig_dir = os.path.join(out_dir, "figures")
    ensure_dir(fig_dir)

    # Preferred dS-based regional columns are used when available. Legacy raw-S
    # columns remain as fallback for old CSVs/checkpoints.
    metrics = [
        (["S_rmse"], "Saturation RMSE", "S RMSE"),
        (["p_phys_RMSE_MPa"] if pressure_unit.lower() == "mpa" else ["p_phys_Pa_rmse"], f"Pressure RMSE ({pressure_unit})", f"p RMSE ({pressure_unit})"),
        (["S_RMSE_front_band"], "Front-band saturation RMSE", "front-band S RMSE"),
        (["S_RMSE_excess_plume_dS005", "dS_RMSE_excess_plume_dS005", "S_RMSE_plume_rawS005"], "Excess-plume saturation RMSE", r"excess-plume $S$ RMSE"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(7.35, 4.85), constrained_layout=False)
    axes = axes.ravel()
    handles, labels = [], []
    for ax, (cands, title, ylabel) in zip(axes, metrics):
        c = _resolve_col(df, cands)
        if c is None:
            ax.set_visible(False)
            continue
        plotted = _plot_grouped_lines(ax, df, c, time_unit)
        ax.set_title(title, pad=2.0)
        ax.set_xlabel(time_axis_label(time_unit))
        ax.set_ylabel(ylabel)
        apply_nature_axis(ax, grid=True, grid_axis="both")
        apply_robust_time_axis(
            ax,
            plotted,
            x_cut=transient_cut_for_unit(time_unit),
            color_for_label=color_for_label,
        )
        h, l = ax.get_legend_handles_labels()
        handles += h; labels += l
    handles, labels = _dedup_legend(handles, labels)
    if handles:
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.005),
                   ncol=_legend_ncol(len(labels)), frameon=False, columnspacing=1.05,
                   handlelength=1.45, handletextpad=0.35, borderaxespad=0.0)
    fig.subplots_adjust(left=0.080, right=0.992, top=0.935, bottom=0.255, wspace=0.33, hspace=0.42)
    _savefig(fig, os.path.join(fig_dir, "global_accuracy_overview"), dpi, export_pdf, export_tiff)
    plt.close(fig)

    # Single-metric publication curves.  Legend is outside the axes and uses
    # only compact labels.
    singles = [
        (["S_rmse"], "Saturation RMSE", "S_RMSE_vs_time"),
        (["p_phys_RMSE_MPa"], "Pressure RMSE (MPa)", "p_RMSE_MPa_vs_time"),
        (["S_RMSE_front_band"], "Front-band saturation RMSE", "front_band_S_RMSE_vs_time"),
        (["S_RMSE_excess_plume_dS005", "dS_RMSE_excess_plume_dS005", "S_RMSE_plume_rawS005"], r"Excess-plume saturation RMSE", "excess_plume_dS005_S_RMSE_vs_time"),
        (["S_RMSE_excess_background_dS001", "dS_RMSE_excess_background_dS001", "S_RMSE_background_rawS002"], r"Excess-background saturation RMSE", "excess_background_dS001_S_RMSE_vs_time"),
    ]
    for cands, ylabel, fname in singles:
        col = _resolve_col(df, cands)
        if col is None:
            continue
        fig, ax = plt.subplots(figsize=(3.55, 2.45), constrained_layout=False)
        plotted = _plot_grouped_lines(ax, df, col, time_unit)
        ax.set_xlabel(time_axis_label(time_unit))
        ax.set_ylabel(ylabel)
        apply_nature_axis(ax, grid=True, grid_axis="both")
        apply_robust_time_axis(
            ax,
            plotted,
            x_cut=transient_cut_for_unit(time_unit),
            color_for_label=color_for_label,
        )
        h, l = _dedup_legend(*ax.get_legend_handles_labels())
        if h:
            ax.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, -0.30),
                      ncol=min(_legend_ncol(len(l)), 5), frameon=False,
                      columnspacing=0.90, handlelength=1.35, handletextpad=0.35)
        fig.subplots_adjust(left=0.165, right=0.985, top=0.955, bottom=0.365)
        _savefig(fig, os.path.join(fig_dir, fname), dpi, export_pdf, export_tiff)
        plt.close(fig)


def plot_selected_time_bars(selected_rows: list[dict[str, Any]], out_dir: str, dpi: int, export_pdf: bool, export_tiff: bool, label_mode: str = "auto"):
    import pandas as pd
    if not selected_rows:
        return
    set_journal_style()
    rows = [dict(r) for r in selected_rows]
    assign_plot_labels(rows, label_mode=label_mode)
    df = pd.DataFrame(rows)
    labels_order = ["t0", "early", "middle", "late", "final"]
    df = df[df["snapshot_label"].isin(labels_order)].copy()
    if df.empty:
        return
    fig_dir = os.path.join(out_dir, "figures")
    ensure_dir(fig_dir)

    metrics = [
        (["S_rmse"], "S RMSE"),
        (["S_RMSE_front_band"], "Front"),
        (["S_RMSE_excess_plume_dS005", "dS_RMSE_excess_plume_dS005", "S_RMSE_plume_rawS005"], "Excess plume"),
        (["S_RMSE_excess_background_dS001", "dS_RMSE_excess_background_dS001", "S_RMSE_background_rawS002"], "Excess background"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(7.35, 4.75), constrained_layout=False)
    axes = axes.ravel()
    legend_handles, legend_labels = [], []
    for ax, (cands, title) in zip(axes, metrics):
        col = _resolve_col(df, cands)
        if col is None:
            ax.set_visible(False)
            continue
        pivot = df.pivot_table(index="snapshot_label", columns="plot_label", values=col, aggfunc="mean").reindex(labels_order)
        x = np.arange(len(pivot.index))
        n = max(1, len(pivot.columns))
        width = min(0.82 / n, 0.105)
        for j, lab in enumerate(pivot.columns):
            ax.bar(x + (j - (n - 1) / 2) * width, pivot[lab].to_numpy(dtype=float),
                   width=width, label=str(lab), linewidth=0.2, color=color_for_label(lab, None))
        ax.set_title(title, pad=2.0)
        ax.set_xticks(x)
        ax.set_xticklabels(pivot.index, rotation=0)
        ax.set_ylabel("RMSE")
        apply_nature_axis(ax, grid=True, grid_axis="y")
        h, l = ax.get_legend_handles_labels()
        legend_handles += h; legend_labels += l
    legend_handles, legend_labels = _dedup_legend(legend_handles, legend_labels)
    if legend_handles:
        fig.legend(legend_handles, legend_labels, loc="lower center", bbox_to_anchor=(0.5, 0.006),
                   ncol=_legend_ncol(len(legend_labels)), frameon=False,
                   columnspacing=0.90, handlelength=1.0, handletextpad=0.35,
                   borderaxespad=0.0)
    fig.subplots_adjust(left=0.080, right=0.992, top=0.935, bottom=0.255, wspace=0.30, hspace=0.42)
    _savefig(fig, os.path.join(fig_dir, "selected_time_global_RMSE_bars"), dpi, export_pdf, export_tiff)
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================
def main():
    ap = argparse.ArgumentParser(description="Global accuracy evaluator for M0-M7 and M7 leave-one-out checkpoints.")
    ap.add_argument("--ckpt-dir", type=str, default=DEFAULT_CKPT_DIR)
    ap.add_argument("--data", type=str, default=DEFAULT_DATA_PATH)
    ap.add_argument("--out-dir", type=str, default=DEFAULT_OUT_DIR)
    ap.add_argument("--experiments", type=str, default=",".join(ALL_EXPS), help="Comma-separated M experiments, e.g. M0,...,M7")
    ap.add_argument("--training-strategy", type=str, default="BASE", help="BASE/BETA/DIFF/BETA_DIFF/ANY")
    ap.add_argument("--include-loo", type=int, default=0, help="Also evaluate leave-one-out checkpoints for --loo-exp.")
    ap.add_argument("--loo-exp", type=str, default="M7")
    ap.add_argument("--leave-outs", type=str, default="NONE,PLAIN_TWONET,FRONT_PLUME,PAIRGRAD,RAR,FV,COARSE_DETAIL")
    ap.add_argument("--seed", type=str, default="ANY")
    ap.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    ap.add_argument("--gpu-id", type=int, default=DEFAULT_GPU_ID)
    ap.add_argument("--dtype", type=str, default=DEFAULT_DTYPE)
    ap.add_argument("--sample-per-time", type=int, default=DEFAULT_SAMPLE_PER_TIME)
    ap.add_argument("--infer-batch-size", type=int, default=DEFAULT_INFER_BATCH_SIZE)
    ap.add_argument("--max-time-steps", type=int, default=DEFAULT_MAX_TIME_STEPS)
    ap.add_argument("--max-sort-n", type=int, default=DEFAULT_MAX_SORT_N)
    ap.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    ap.add_argument("--time-indices", type=str, default=DEFAULT_TIME_INDICES)
    ap.add_argument("--time-unit", type=str, default=DEFAULT_TIME_UNIT)
    ap.add_argument("--pressure-display-unit", type=str, default=DEFAULT_PRESSURE_DISPLAY_UNIT)
    ap.add_argument("--figure-dpi", type=int, default=DEFAULT_FIGURE_DPI)
    ap.add_argument("--export-pdf", type=int, default=int(DEFAULT_EXPORT_PDF))
    ap.add_argument("--export-tiff", type=int, default=int(DEFAULT_EXPORT_TIFF))
    ap.add_argument("--export-svg", type=int, default=int(DEFAULT_EXPORT_SVG),
                    help="Export editable SVG files in addition to screening PNG/PDF/TIFF.")
    ap.add_argument("--s-ic-co2", type=float, default=DEFAULT_S_IC_CO2,
                    help="Training-script initial CO2 saturation used for dS/IC metrics.")
    ap.add_argument("--Snr", type=float, default=DEFAULT_SNR,
                    help="Training-script CO2 residual saturation used for region diagnostics.")
    ap.add_argument("--Sw-irr", type=float, default=DEFAULT_SW_IRR,
                    help="Training-script irreducible water saturation recorded in resolved configs.")
    ap.add_argument("--plot-label-mode", type=str, default="auto", choices=("auto", "exp", "loo"),
                    help="Figure legend label mode. auto uses M0-M7 for forward runs and L1-L6 for leave-one-out; loo labels M7 LOO-NONE as L0.")
    args = ap.parse_args()
    global EXPORT_SVG
    EXPORT_SVG = bool(args.export_svg)

    ensure_dir(args.out_dir)
    ensure_dir(os.path.join(args.out_dir, "figures"))

    device = torch.device(f"cuda:{args.gpu_id}" if args.device.lower().startswith("cuda") and torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")
    print(f"[data] {args.data}")
    print(f"[ckpt-dir] {args.ckpt_dir}")
    arrays, time_unique = load_dataset_pack(args.data)
    eval_plan = build_eval_plan(arrays, time_unique, args.sample_per_time, args.max_time_steps, args.max_sort_n, args.random_seed, args.time_indices)
    print(f"[time] n={len(eval_plan['time_seconds'])}, selected={eval_plan['selected_items']}")

    experiments = [x.strip().upper() for x in args.experiments.split(",") if x.strip()]
    leave_outs = [x.strip().upper() for x in args.leave_outs.split(",") if x.strip()]
    seed = None if args.seed.upper() == "ANY" else args.seed
    specs = make_run_specs(args.ckpt_dir, experiments, args.training_strategy.upper(), bool(args.include_loo), args.loo_exp.upper(), leave_outs, seed)
    if not specs:
        raise RuntimeError("No checkpoints found. Check --ckpt-dir, --experiments, --training-strategy, and --leave-outs.")
    print(f"[runs] {len(specs)}")
    for s in specs:
        print(f"  - {s.run_id}: {s.ckpt_path}")

    summary_rows = []
    by_time_rows = []
    selected_rows = []
    resolved_configs = {}
    for spec in specs:
        print(f"[eval] {spec.run_id}")
        summary, time_rows, sel_rows, cfg = evaluate_run(
            spec,
            eval_plan,
            device,
            args.infer_batch_size,
            args.pressure_display_unit,
            args.s_ic_co2,
            args.Snr,
            args.Sw_irr,
        )
        summary_rows.append(summary)
        by_time_rows.extend(time_rows)
        selected_rows.extend(sel_rows)
        resolved_configs[spec.run_id] = {k: v for k, v in cfg.items() if k != "checkpoint_config"}

    # Add compact labels used by all figures. run_id remains in CSV for traceability.
    assign_plot_labels(summary_rows, label_mode=args.plot_label_mode)
    assign_plot_labels(by_time_rows, label_mode=args.plot_label_mode)
    assign_plot_labels(selected_rows, label_mode=args.plot_label_mode)

    # Sort rows for readability.
    summary_rows = sorted(summary_rows, key=lambda r: (r["exp_name"], r["training_strategy"], r["leave_out"], str(r["seed"])))
    by_time_rows = sorted(by_time_rows, key=lambda r: (r["run_id"], int(r["time_index"])))
    selected_rows = sorted(selected_rows, key=lambda r: (r["run_id"], int(r["time_index"])))

    write_csv(os.path.join(args.out_dir, "summary_global_accuracy.csv"), summary_rows)
    write_csv(os.path.join(args.out_dir, "by_time_global_accuracy.csv"), by_time_rows)
    write_csv(os.path.join(args.out_dir, "selected_time_global_accuracy.csv"), selected_rows)
    save_json({
        "args": vars(args),
        "selected_time_items": eval_plan["selected_items"],
        "region_definitions": {
            "front_band": "S_true in [front_low, front_high] = [0.05, 0.30] by default",
            "plume_rawS005": "raw S_true > plume_threshold (legacy diagnostic)",
            "excess_plume_dS005": "S_true - S_IC_CO2 > 0.05",
            "background_rawS002": "raw S_true < background_threshold (legacy diagnostic)",
            "excess_background_dS001": "S_true - S_IC_CO2 < 0.01",
            "core": "S_true > core_threshold",
            "near_sic": "abs(S_true - S_IC_CO2) <= 0.02",
            "mobile_co2_training_raw": "S_true > max(Snr, S_IC_CO2), aligned with the training mobile-CO2 mask",
            "mobile_co2": "S_true > max(Snr, S_IC_CO2)+0.02",
        },
        "resolved_configs": resolved_configs,
    }, os.path.join(args.out_dir, "global_evaluation_config.json"))

    plot_global_curves(by_time_rows, args.out_dir, args.time_unit, args.pressure_display_unit, args.figure_dpi, bool(args.export_pdf), bool(args.export_tiff), label_mode=args.plot_label_mode)
    plot_selected_time_bars(selected_rows, args.out_dir, args.figure_dpi, bool(args.export_pdf), bool(args.export_tiff), label_mode=args.plot_label_mode)

    print(f"[done] outputs written to {args.out_dir}")


if __name__ == "__main__":
    main()
