#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Front + plume evaluation for M0-M7 PINN checkpoints and M7 leave-one-out runs.

Outputs
-------
  <out-root>/front/summary_front_resolution.csv
  <out-root>/front/by_time_front_resolution.csv
  <out-root>/front/pairgrad_metrics.csv
  <out-root>/front/speckle_metrics.csv
  <out-root>/plume/summary_plume_support.csv
  <out-root>/plume/by_time_plume_support.csv
  <out-root>/front/figures/*.png|.pdf|.tiff
  <out-root>/plume/figures/*.png|.pdf|.tiff

Scope
-----
This script evaluates front geometry and plume support only:
  - front-band saturation errors
  - multi-level plume IoU/Dice/FP/FN/area metrics
  - contour Chamfer/Hausdorff distances on selected times
  - pairwise front-gradient finite-difference errors
  - low-saturation false-positive / speckle diagnostics

It intentionally does not compute global pressure/saturation fitting metrics,
PDE/FV/BC/IC physics residuals, or full snapshot fields. Those belong to the
corresponding global/physics/snapshot evaluators.

Design notes
------------
1. Time-layer selection is aligned with evaluate_snapshot_gridviz_M0_M7_publication_snapshots.py:
   t0, early, middle, late, final.
2. Sco2 initial saturation defaults to S_IC_CO2=0.0, matching the training script. Raw Sco2 is still
   compared against raw Sco2 data for value errors. Plume, FP/FN and speckle
   diagnostics are additionally reported using excess saturation dS = Sco2 - S_IC_CO2,
   which keeps the convention explicit if a future nonzero-initial-saturation dataset is evaluated.
3. Figure labels are short by construction: M0-M7 for forward ablation, and L0-L6
   for leave-one-out runs. Long checkpoint strings are not used in legends.

Dependency
----------
This script imports the model/checkpoint/dataset utilities from
`evaluate_global_accuracy_M0_M7_L6_dS_shortlabels.py`. Keep both scripts in the
same directory, or add that directory to PYTHONPATH.
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

from scipy.interpolate import griddata
from scipy.spatial import cKDTree
try:
    from scipy import ndimage as ndi
except Exception:  # pragma: no cover
    ndi = None

# Reuse checkpoint/model/data utilities from the global evaluator to avoid
# architecture drift between evaluation scripts.
from evaluate_global_accuracy_M0_M7_L6_dS_shortlabels import (  # type: ignore
    ALL_EXPS,
    ANALYSIS_T_MAX_S,
    DOMAIN_X_M,
    DOMAIN_Y_M,
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
DEFAULT_OUT_ROOT = None
DEFAULT_SAMPLE_PER_TIME = 20000
DEFAULT_GRID_NX = 420
DEFAULT_GRID_NY = 420
DEFAULT_MAX_INTERP_INPUT_POINTS = 120000
DEFAULT_FIGURE_DPI = 600
DEFAULT_EXPORT_PDF = True
DEFAULT_EXPORT_TIFF = True
DEFAULT_EXPORT_SVG = True
EXPORT_SVG = DEFAULT_EXPORT_SVG

# Front and plume defaults. These are aligned with the M-series objectives.
DEFAULT_FRONT_CENTER = 0.175
DEFAULT_FRONT_SIGMA = 0.075
DEFAULT_FRONT_LOW_WIDE = 0.05
DEFAULT_FRONT_HIGH_WIDE = 0.30
DEFAULT_FRONT_LOW_NARROW = 0.10
DEFAULT_FRONT_HIGH_NARROW = 0.25
DEFAULT_CONTOUR_LEVELS = (0.05, 0.10, 0.175, 0.30, 0.40, 0.50)
DEFAULT_PLUME_THRESHOLDS = (0.05, 0.10, 0.20, 0.40)  # raw-S legacy thresholds
DEFAULT_EXCESS_PLUME_THRESHOLDS = (0.02, 0.05, 0.10, 0.20)  # dS thresholds; identical to raw-S when S_IC_CO2=0

# Pairwise front-gradient diagnostics.
DEFAULT_PAIRGRAD_MAX_PAIRS_PER_TIME = 25000
DEFAULT_PAIRGRAD_S_MIN = 0.05
DEFAULT_PAIRGRAD_S_MAX = 0.30
DEFAULT_PAIRGRAD_K_NEIGHBORS = 4
DEFAULT_PAIRGRAD_MIN_DIST = 2.0e-4
DEFAULT_PAIRGRAD_MAX_DIST = 4.0 / 64.0
DEFAULT_PAIRGRAD_CLIP = 20.0
DEFAULT_PAIRGRAD_FRONT_CENTER = 0.175
DEFAULT_PAIRGRAD_FRONT_SIGMA = 0.075
DEFAULT_PAIRGRAD_FRONT_WEIGHT_FLOOR = 0.01

# Speckle definitions.
DEFAULT_SPECKLE_PRED_002 = 0.02
DEFAULT_SPECKLE_TRUE_BG_002 = 0.01
DEFAULT_SPECKLE_PRED_005 = 0.05
DEFAULT_SPECKLE_TRUE_BG_005 = 0.02
DEFAULT_SPECKLE_MIN_COMPONENT_PIXELS = 3
DEFAULT_S_IC_CO2_EVAL = DEFAULT_S_IC_CO2


# =============================================================================
# Small utilities
# =============================================================================
def parse_float_list(s: str) -> list[float]:
    return [float(x.strip()) for x in str(s).split(",") if x.strip()]


def short_label(row_or_spec, loo_labels: bool = False) -> str:
    exp = getattr(row_or_spec, "exp_name", None) if not isinstance(row_or_spec, dict) else row_or_spec.get("exp_name")
    leave = getattr(row_or_spec, "leave_out", None) if not isinstance(row_or_spec, dict) else row_or_spec.get("leave_out")
    exp = str(exp or "").upper()
    leave = str(leave or "NONE").upper()
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


def metric_basic(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return dict(mean=np.nan, median=np.nan, p90=np.nan, p95=np.nan, p99=np.nan, max=np.nan)
    return dict(
        mean=float(np.mean(values)),
        median=float(np.median(values)),
        p90=float(np.percentile(values, 90)),
        p95=float(np.percentile(values, 95)),
        p99=float(np.percentile(values, 99)),
        max=float(np.max(values)),
    )


def error_metrics(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    true = np.asarray(true, dtype=np.float64).reshape(-1)
    if pred.shape != true.shape:
        raise ValueError(f"Front metric arrays must have equal shapes; got {pred.shape} and {true.shape}.")
    if not np.isfinite(pred).all() or not np.isfinite(true).all():
        raise FloatingPointError("Nonfinite front metric input detected; evaluation is fail-closed.")
    if pred.size == 0:
        return dict(mse=np.nan, rmse=np.nan, mae=np.nan, bias=np.nan, p95ae=np.nan, n=0)
    err = pred - true
    ae = np.abs(err)
    mse = float(np.mean(err * err))
    return dict(
        mse=mse,
        rmse=float(math.sqrt(mse)),
        mae=float(np.mean(ae)),
        bias=float(np.mean(err)),
        p95ae=float(np.percentile(ae, 95.0)),
        n=int(pred.size),
    )


def metric_aliases(metrics: dict[str, Any], old_token: str, new_token: str) -> dict[str, Any]:
    """Keep legacy output columns while adding training-aligned column names."""
    out: dict[str, Any] = {}
    for key, value in metrics.items():
        out[key.replace(old_token, new_token)] = value
    return out


def add_prefixed(out: dict[str, Any], prefix: str, met: dict[str, Any]):
    for k, v in met.items():
        out[f"{prefix}_{k}"] = v


def set_journal_style():
    setup_nature_rcparams(font_size=7.0, line_width=1.1)


def savefig(fig, base: str, dpi: int, export_pdf: bool, export_tiff: bool):
    ensure_dir(os.path.dirname(base))
    save_pub_figure(fig, base, dpi=dpi, export_pdf=export_pdf, export_tiff=export_tiff, export_svg=EXPORT_SVG, export_png=True, pad_inches=0.035)


# =============================================================================
# Front/plume metrics on sampled data
# =============================================================================
def front_band_metrics(s_pred: np.ndarray, s_true: np.ndarray, low: float, high: float, name: str) -> dict[str, Any]:
    mask = (s_true >= low) & (s_true <= high)
    out = {}
    if np.any(mask):
        add_prefixed(out, f"front_band_{name}", error_metrics(s_pred[mask], s_true[mask]))
    else:
        add_prefixed(out, f"front_band_{name}", error_metrics(np.array([]), np.array([])))
    out[f"front_band_{name}_fraction"] = float(np.mean(mask)) if mask.size else np.nan
    return out


def plume_metrics(s_pred: np.ndarray, s_true: np.ndarray, thresholds: list[float]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    s_pred = np.asarray(s_pred, dtype=np.float64)
    s_true = np.asarray(s_true, dtype=np.float64)
    valid = np.isfinite(s_pred) & np.isfinite(s_true)
    n = max(1, int(np.sum(valid)))
    for tau in thresholds:
        tag = f"S{int(round(tau * 100)):03d}"
        mp = valid & (s_pred > tau)
        mt = valid & (s_true > tau)
        tp = mp & mt
        fp = mp & (~mt)
        fn = (~mp) & mt
        union = mp | mt
        pred_area = float(np.sum(mp)) / n
        true_area = float(np.sum(mt)) / n
        out[f"plume_IoU_{tag}"] = float(np.sum(tp) / max(1, np.sum(union))) if np.sum(union) > 0 else np.nan
        out[f"plume_Dice_{tag}"] = float(2 * np.sum(tp) / max(1, np.sum(mp) + np.sum(mt))) if (np.sum(mp) + np.sum(mt)) > 0 else np.nan
        out[f"plume_area_true_{tag}"] = true_area
        out[f"plume_area_pred_{tag}"] = pred_area
        out[f"plume_area_error_abs_{tag}"] = pred_area - true_area
        out[f"plume_area_error_rel_{tag}"] = (pred_area - true_area) / max(true_area, 1.0e-12)
        out[f"FP_area_ratio_{tag}"] = float(np.sum(fp)) / max(1, np.sum(mt)) if np.sum(mt) > 0 else np.nan
        out[f"FN_area_ratio_{tag}"] = float(np.sum(fn)) / max(1, np.sum(mt)) if np.sum(mt) > 0 else np.nan
        out[f"FP_fraction_domain_{tag}"] = float(np.sum(fp)) / n
        out[f"FN_fraction_domain_{tag}"] = float(np.sum(fn)) / n
    return out


def plume_metrics_excess(s_pred: np.ndarray, s_true: np.ndarray, s_ic: float, thresholds: list[float]) -> dict[str, Any]:
    """Plume support metrics based on excess saturation dS = S - S_IC_CO2.

    This is the preferred definition for the M-series final model because it
    keeps the initial-saturation convention explicit; with the current training
    data S_IC_CO2=0.0, dS and raw-S plume thresholds coincide.
    """
    out: dict[str, Any] = {}
    d_pred = np.asarray(s_pred, dtype=np.float64) - float(s_ic)
    d_true = np.asarray(s_true, dtype=np.float64) - float(s_ic)
    valid = np.isfinite(d_pred) & np.isfinite(d_true)
    n = max(1, int(np.sum(valid)))
    for tau in thresholds:
        tag = f"dS{int(round(tau * 100)):03d}"
        mp = valid & (d_pred > tau)
        mt = valid & (d_true > tau)
        tp = mp & mt
        fp = mp & (~mt)
        fn = (~mp) & mt
        union = mp | mt
        pred_area = float(np.sum(mp)) / n
        true_area = float(np.sum(mt)) / n
        out[f"plume_IoU_{tag}"] = float(np.sum(tp) / max(1, np.sum(union))) if np.sum(union) > 0 else np.nan
        out[f"plume_Dice_{tag}"] = float(2 * np.sum(tp) / max(1, np.sum(mp) + np.sum(mt))) if (np.sum(mp) + np.sum(mt)) > 0 else np.nan
        out[f"plume_area_true_{tag}"] = true_area
        out[f"plume_area_pred_{tag}"] = pred_area
        out[f"plume_area_error_abs_{tag}"] = pred_area - true_area
        out[f"plume_area_error_rel_{tag}"] = (pred_area - true_area) / max(true_area, 1.0e-12)
        out[f"FP_area_ratio_{tag}"] = float(np.sum(fp)) / max(1, np.sum(mt)) if np.sum(mt) > 0 else np.nan
        out[f"FN_area_ratio_{tag}"] = float(np.sum(fn)) / max(1, np.sum(mt)) if np.sum(mt) > 0 else np.nan
        out[f"FP_fraction_domain_{tag}"] = float(np.sum(fp)) / n
        out[f"FN_fraction_domain_{tag}"] = float(np.sum(fn)) / n
    return out


def pairgrad_metrics_xy(
    x: np.ndarray,
    y: np.ndarray,
    s_true: np.ndarray,
    s_pred: np.ndarray,
    rng: np.random.Generator,
    max_pairs: int,
    s_min: float,
    s_max: float,
    k_neighbors: int,
    min_dist: float,
    max_dist: float,
    clip_value: float,
    front_center: float,
    front_sigma: float,
    front_weight_floor: float,
) -> dict[str, Any]:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    s_true = np.asarray(s_true, dtype=np.float64).reshape(-1)
    s_pred = np.asarray(s_pred, dtype=np.float64).reshape(-1)
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(s_true) & np.isfinite(s_pred) & (s_true >= s_min) & (s_true <= s_max)
    idx_all = np.where(valid)[0]
    if idx_all.size < 8:
        return {"n_pairs": 0, "pairgrad_RMSE": np.nan, "pairgrad_MAE": np.nan, "pairgrad_Bias": np.nan, "pairgrad_Corr": np.nan, "pairgrad_P95_AE": np.nan, "pairgrad_RMSE_front_weighted": np.nan}
    # Cap anchors to keep the diagnostic bounded.
    max_anchors = min(idx_all.size, max(512, int(max_pairs // max(1, k_neighbors))))
    if idx_all.size > max_anchors:
        idx_anchor = rng.choice(idx_all, size=max_anchors, replace=False)
    else:
        idx_anchor = idx_all
    pts_all = np.column_stack([x[idx_all], y[idx_all]])
    pts_anchor = np.column_stack([x[idx_anchor], y[idx_anchor]])
    tree = cKDTree(pts_all)
    dists, neigh = tree.query(pts_anchor, k=min(k_neighbors + 1, idx_all.size))
    if dists.ndim == 1:
        dists = dists[:, None]
        neigh = neigh[:, None]
    rows_i, rows_j, rows_d = [], [], []
    for arow, ai in enumerate(idx_anchor):
        for jj in range(1, neigh.shape[1]):
            d = float(dists[arow, jj])
            if not np.isfinite(d) or d < min_dist or d > max_dist:
                continue
            bj = idx_all[int(neigh[arow, jj])]
            rows_i.append(ai); rows_j.append(bj); rows_d.append(d)
            if len(rows_i) >= max_pairs:
                break
        if len(rows_i) >= max_pairs:
            break
    if not rows_i:
        return {"n_pairs": 0, "pairgrad_RMSE": np.nan, "pairgrad_MAE": np.nan, "pairgrad_Bias": np.nan, "pairgrad_Corr": np.nan, "pairgrad_P95_AE": np.nan, "pairgrad_RMSE_front_weighted": np.nan}
    ii = np.asarray(rows_i, dtype=np.int64)
    jj = np.asarray(rows_j, dtype=np.int64)
    dd = np.asarray(rows_d, dtype=np.float64)
    gt = (s_true[jj] - s_true[ii]) / dd
    gp = (s_pred[jj] - s_pred[ii]) / dd
    gt = np.clip(gt, -clip_value, clip_value)
    gp = np.clip(gp, -clip_value, clip_value)
    err = gp - gt
    ae = np.abs(err)
    mid_s = 0.5 * (s_true[ii] + s_true[jj])
    w = np.exp(-((mid_s - front_center) ** 2) / (2.0 * front_sigma ** 2))
    w = np.maximum(w, front_weight_floor)
    corr = np.nan
    if gt.size > 2 and np.std(gt) > 1.0e-12 and np.std(gp) > 1.0e-12:
        corr = float(np.corrcoef(gp, gt)[0, 1])
    return {
        "n_pairs": int(gt.size),
        "pairgrad_RMSE": float(math.sqrt(np.mean(err * err))),
        "pairgrad_MAE": float(np.mean(ae)),
        "pairgrad_Bias": float(np.mean(err)),
        "pairgrad_Corr": corr,
        "pairgrad_P95_AE": float(np.percentile(ae, 95.0)),
        "pairgrad_RMSE_front_weighted": float(math.sqrt(np.sum(w * err * err) / (np.sum(w) + 1.0e-30))),
    }


# =============================================================================
# Grid utilities for selected-time contour and speckle diagnostics
# =============================================================================
def regular_grid_from_points(x: np.ndarray, y: np.ndarray, nx: int, ny: int):
    # Configuration K contours are evaluated on one immutable physical support.
    # Using random-subsample extrema lets the grid (and maximum possible error)
    # drift across runs even when the physical domain is unchanged.
    x_min, x_max = DOMAIN_X_M
    y_min, y_max = DOMAIN_Y_M
    gx = np.linspace(x_min, x_max, int(nx))
    gy = np.linspace(y_min, y_max, int(ny))
    Xg, Yg = np.meshgrid(gx, gy)
    return gx, gy, Xg, Yg


def interpolate_true_to_grid(x: np.ndarray, y: np.ndarray, values: np.ndarray, Xg: np.ndarray, Yg: np.ndarray, max_points: int, rng: np.random.Generator):
    """Map the complete unique native support to the fixed grid.

    ``max_points`` and ``rng`` remain in the public signature for compatibility
    with historical callers, but formal contour truth is never resampled.
    """
    del max_points, rng
    pts = np.column_stack([x, y])
    vals = np.asarray(values, dtype=np.float64)
    good = np.isfinite(pts).all(axis=1) & np.isfinite(vals)
    pts = pts[good]
    vals = vals[good]
    if pts.shape[0] == 0:
        return np.full_like(Xg, np.nan, dtype=np.float64)
    query = np.column_stack([Xg.reshape(-1), Yg.reshape(-1)])
    nearest = cKDTree(pts).query(query, k=1)[1]
    return vals[nearest].reshape(Xg.shape)


def apply_effective_domain_mask(
    Xg: np.ndarray,
    Yg: np.ndarray,
    true_grid: np.ndarray,
    pred_grid: np.ndarray,
    cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mask the two quarter-well holes in both reference and prediction grids."""

    if not (Xg.shape == Yg.shape == true_grid.shape == pred_grid.shape):
        raise ValueError("Contour coordinates, truth, and prediction grids must have identical shapes.")
    required = ("r_well", "inj_center", "out_center")
    missing = [key for key in required if key not in cfg]
    if missing:
        raise ValueError(f"Checkpoint geometry is missing contour-domain keys: {missing}.")
    radius = float(cfg["r_well"])
    inj = cfg["inj_center"]
    out = cfg["out_center"]
    if not np.isfinite(radius) or radius <= 0.0 or len(inj) != 2 or len(out) != 2:
        raise ValueError("Checkpoint well geometry must contain a positive radius and two 2-D centres.")
    valid = ((Xg - float(inj[0])) ** 2 + (Yg - float(inj[1])) ** 2) > radius**2
    valid &= ((Xg - float(out[0])) ** 2 + (Yg - float(out[1])) ** 2) > radius**2
    true_masked = np.asarray(true_grid, dtype=np.float64).copy()
    pred_masked = np.asarray(pred_grid, dtype=np.float64).copy()
    true_masked[~valid] = np.nan
    pred_masked[~valid] = np.nan
    return true_masked, pred_masked, valid


def _require_nondecreasing_reference_time(time_values: np.ndarray, *, chunk_size: int = 1_000_000) -> None:
    values = np.asarray(time_values, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Full contour reference requires nonempty finite time values.")
    previous = float(values[0])
    for start in range(0, values.size, int(chunk_size)):
        chunk = values[start:min(start + int(chunk_size), values.size)]
        if float(chunk[0]) < previous or (chunk.size > 1 and np.any(chunk[1:] < chunk[:-1])):
            raise ValueError("Full contour reference requires rows grouped in nondecreasing time order.")
        previous = float(chunk[-1])


def _collapse_unique_spatial_layer(x: np.ndarray, y: np.ndarray, values: np.ndarray):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if x.size == 0 or y.size != x.size or values.size != x.size:
        raise ValueError("Full contour reference x/y/S arrays must be nonempty and equally sized.")
    if not np.isfinite(x).all() or not np.isfinite(y).all() or not np.isfinite(values).all():
        raise ValueError("Full contour reference x/y/S arrays must be finite.")
    order = np.lexsort((y, x))
    x_sorted = x[order]
    y_sorted = y[order]
    values_sorted = values[order]
    is_start = np.ones(x.size, dtype=bool)
    is_start[1:] = (x_sorted[1:] != x_sorted[:-1]) | (y_sorted[1:] != y_sorted[:-1])
    starts = np.flatnonzero(is_start)
    counts = np.diff(np.append(starts, x.size)).astype(np.float64)
    collapsed = np.add.reduceat(values_sorted, starts) / counts
    return x_sorted[starts], y_sorted[starts], collapsed


def build_full_reference_layers(
    arrays: dict[str, Any],
    time_seconds: np.ndarray,
    *,
    selected_indices: set[int],
) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Return complete, duplicate-collapsed reference layers for contour times."""

    x_all = to_np_1d(arrays["x"]).astype(np.float64, copy=False)
    y_all = to_np_1d(arrays["y"]).astype(np.float64, copy=False)
    t_all = to_np_1d(arrays["t"]).astype(np.float64, copy=False)
    s_all = np.clip(to_np_1d(arrays["Sco2"]).astype(np.float64, copy=False), 0.0, 1.0)
    if not (x_all.size == y_all.size == t_all.size == s_all.size):
        raise ValueError("Full contour reference arrays x/y/t/Sco2 must have equal row counts.")
    _require_nondecreasing_reference_time(t_all)
    times = np.asarray(time_seconds, dtype=np.float64)
    layers: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for index in sorted(int(v) for v in selected_indices):
        if index < 0 or index >= times.size:
            raise ValueError(f"Selected contour time index {index} is outside 0..{times.size - 1}.")
        time_s = float(times[index])
        lo = int(np.searchsorted(t_all, time_s, side="left"))
        hi = int(np.searchsorted(t_all, time_s, side="right"))
        if hi <= lo:
            raise ValueError(f"Full contour reference found no layer at t={time_s!r} s.")
        layers[index] = _collapse_unique_spatial_layer(
            x_all[lo:hi], y_all[lo:hi], s_all[lo:hi]
        )
    return layers


def predict_grid(model, Xg: np.ndarray, Yg: np.ndarray, t_value: float, cfg: dict[str, Any], device: torch.device, batch_size: int):
    x_flat = Xg.reshape(-1)
    y_flat = Yg.reshape(-1)
    t_flat = np.full_like(x_flat, float(t_value), dtype=np.float64)
    _, s_pred = predict_model(model, x_flat, y_flat, t_flat, cfg, device, batch_size)
    return np.clip(s_pred.reshape(Xg.shape), 0.0, 1.0)


def contour_points(Xg: np.ndarray, Yg: np.ndarray, Z: np.ndarray, level: float) -> np.ndarray:
    if not np.isfinite(Z).any() or np.nanmin(Z) > level or np.nanmax(Z) < level:
        return np.empty((0, 2), dtype=np.float64)
    fig, ax = plt.subplots(figsize=(1, 1))
    try:
        cs = ax.contour(Xg, Yg, Z, levels=[level])
        pts = []
        for seg in cs.allsegs[0]:
            if seg.shape[0] > 0:
                pts.append(np.asarray(seg, dtype=np.float64))
        if not pts:
            raise RuntimeError(
                f"Contour field brackets level {level!r} but extraction returned no contour segments."
            )
        return np.concatenate(pts, axis=0)
    finally:
        plt.close(fig)


def saturation_level_tag(level: float) -> str:
    """Stable metric tag, preserving the preregistered S=0.175 level."""
    value = float(level)
    if np.isclose(value, 0.175, rtol=0.0, atol=1.0e-12):
        return "S0175"
    return f"S{int(round(value * 100)):03d}"


def contour_distance_metrics(Xg: np.ndarray, Yg: np.ndarray, true_grid: np.ndarray, pred_grid: np.ndarray, level: float) -> dict[str, Any]:
    pt = contour_points(Xg, Yg, true_grid, level)
    pp = contour_points(Xg, Yg, pred_grid, level)
    tag = saturation_level_tag(level)
    out = {f"contour_n_true_{tag}": int(pt.shape[0]), f"contour_n_pred_{tag}": int(pp.shape[0])}
    true_missing = pt.shape[0] == 0
    pred_missing = pp.shape[0] == 0
    if true_missing and pred_missing:
        out[f"contour_missing_status_{tag}"] = "both_missing"
        out.update({f"contour_chamfer_{tag}": np.nan, f"contour_hausdorff_{tag}": np.nan, f"contour_p95dist_{tag}": np.nan})
        return out
    if true_missing or pred_missing:
        # One-sided contour absence is a maximal miss, not a value that may be
        # silently dropped from the mean.  The fixed-domain diagonal supplies a
        # deterministic, unit-consistent penalty.
        diagonal = float(math.hypot(np.ptp(Xg), np.ptp(Yg)))
        out[f"contour_missing_status_{tag}"] = (
            "reference_missing" if true_missing else "predicted_missing"
        )
        out.update({
            f"contour_chamfer_{tag}": diagonal,
            f"contour_hausdorff_{tag}": diagonal,
            f"contour_p95dist_{tag}": diagonal,
        })
        return out
    out[f"contour_missing_status_{tag}"] = "present"
    tree_t = cKDTree(pt)
    tree_p = cKDTree(pp)
    d_p_to_t, _ = tree_t.query(pp, k=1)
    d_t_to_p, _ = tree_p.query(pt, k=1)
    all_d = np.concatenate([d_p_to_t, d_t_to_p])
    out[f"contour_chamfer_{tag}"] = float(0.5 * (np.mean(d_p_to_t) + np.mean(d_t_to_p)))
    out[f"contour_hausdorff_{tag}"] = float(max(np.max(d_p_to_t), np.max(d_t_to_p)))
    out[f"contour_p95dist_{tag}"] = float(np.percentile(all_d, 95.0))
    return out


def validate_canonical_contour_support(rows: list[dict[str, Any]]) -> None:
    """Require at least one informative S=0.175 snapshot for every run.

    A t=0 ``both_missing`` state is physically expected and is retained as
    telemetry.  The canonical mean nevertheless fails closed if an entire run
    has no present or one-sided (penalized) S=0.175 contour comparison.
    """

    by_run: dict[str, list[str]] = {}
    for row in rows:
        run_id = str(row.get("run_id", "")).strip()
        if not run_id:
            raise ValueError("Canonical contour telemetry is missing run_id.")
        status = str(row.get("contour_missing_status_S0175", "")).strip()
        if not status:
            raise ValueError(
                f"Run {run_id!r} is missing canonical S=0.175 contour status telemetry."
            )
        by_run.setdefault(run_id, []).append(status)
    if not by_run:
        raise ValueError("No canonical S=0.175 contour telemetry was produced.")
    informative = {"present", "reference_missing", "predicted_missing"}
    for run_id, statuses in by_run.items():
        if not any(status in informative for status in statuses):
            raise ValueError(
                f"Run {run_id!r} has no informative S=0.175 contour comparison at any selected time."
            )


def speckle_grid_metrics(true_grid: np.ndarray, pred_grid: np.ndarray, pred_thr: float, true_bg_thr: float, min_pixels: int, tag: str) -> dict[str, Any]:
    valid = np.isfinite(true_grid) & np.isfinite(pred_grid)
    denom = max(1, int(np.sum(valid)))
    speckle = valid & (pred_grid > pred_thr) & (true_grid < true_bg_thr)
    out = {
        f"speckle_area_ratio_{tag}": float(np.sum(speckle)) / denom,
        f"speckle_pixels_{tag}": int(np.sum(speckle)),
    }
    if ndi is None:
        out.update({f"speckle_component_count_{tag}": np.nan, f"speckle_component_count_small_{tag}": np.nan, f"speckle_mean_area_{tag}": np.nan, f"speckle_max_area_{tag}": np.nan})
        return out
    struct = np.ones((3, 3), dtype=np.int8)
    lab, nlab = ndi.label(speckle, structure=struct)
    if nlab == 0:
        out.update({f"speckle_component_count_{tag}": 0, f"speckle_component_count_small_{tag}": 0, f"speckle_mean_area_{tag}": 0.0, f"speckle_max_area_{tag}": 0.0})
        return out
    areas = np.bincount(lab.reshape(-1))[1:]
    out[f"speckle_component_count_{tag}"] = int(nlab)
    out[f"speckle_component_count_small_{tag}"] = int(np.sum(areas >= int(min_pixels)))
    out[f"speckle_mean_area_{tag}"] = float(np.mean(areas))
    out[f"speckle_max_area_{tag}"] = float(np.max(areas))
    return out




def speckle_grid_metrics_excess(true_grid: np.ndarray, pred_grid: np.ndarray, s_ic: float, pred_thr: float, true_bg_thr: float, min_pixels: int, tag: str) -> dict[str, Any]:
    d_true = np.asarray(true_grid, dtype=np.float64) - float(s_ic)
    d_pred = np.asarray(pred_grid, dtype=np.float64) - float(s_ic)
    out = speckle_grid_metrics(d_true, d_pred, pred_thr, true_bg_thr, min_pixels, f"dS{tag}")
    # Add explicit background dS diagnostics for clarity.
    bg = np.isfinite(d_true) & np.isfinite(d_pred) & (d_true < true_bg_thr)
    if np.any(bg):
        vals = d_pred[bg]
        out.update({
            f"background_dS_mean_pred_{tag}": float(np.mean(vals)),
            f"background_dS_p95_pred_{tag}": float(np.percentile(vals, 95.0)),
            f"background_dS_p99_pred_{tag}": float(np.percentile(vals, 99.0)),
            f"background_dS_max_pred_{tag}": float(np.max(vals)),
        })
    else:
        out.update({f"background_dS_mean_pred_{tag}": np.nan, f"background_dS_p95_pred_{tag}": np.nan, f"background_dS_p99_pred_{tag}": np.nan, f"background_dS_max_pred_{tag}": np.nan})
    return out


def contour_distance_metrics_excess(X: np.ndarray, Y: np.ndarray, true_grid: np.ndarray, pred_grid: np.ndarray, s_ic: float, excess_level: float) -> dict[str, Any]:
    raw_level = float(s_ic) + float(excess_level)
    raw_tag = f"S{int(round(raw_level * 100)):03d}"
    ds_tag = f"dS{int(round(float(excess_level) * 100)):03d}"
    res = contour_distance_metrics(X, Y, true_grid, pred_grid, raw_level)
    return {str(k).replace(raw_tag, ds_tag): v for k, v in res.items()}

def grid_front_plume_diagnostics(
    model,
    cfg: dict[str, Any],
    device: torch.device,
    batch_size: int,
    x: np.ndarray,
    y: np.ndarray,
    t_value: float,
    s_true_points: np.ndarray,
    nx: int,
    ny: int,
    max_interp_points: int,
    rng: np.random.Generator,
    contour_levels: list[float],
    plume_thresholds: list[float],
    excess_plume_thresholds: list[float],
    s_ic_co2: float,
    speckle_pred_002: float,
    speckle_true_bg_002: float,
    speckle_pred_005: float,
    speckle_true_bg_005: float,
    min_component_pixels: int,
) -> dict[str, Any]:
    gx, gy, Xg, Yg = regular_grid_from_points(x, y, nx, ny)
    s_true_grid = interpolate_true_to_grid(x, y, s_true_points, Xg, Yg, max_interp_points, rng)
    s_pred_grid = predict_grid(model, Xg, Yg, t_value, cfg, device, batch_size)
    s_true_grid, s_pred_grid, valid_domain = apply_effective_domain_mask(
        Xg, Yg, s_true_grid, s_pred_grid, cfg
    )
    out: dict[str, Any] = {
        "grid_effective_domain_cells": int(np.sum(valid_domain)),
        "grid_masked_well_cells": int(np.sum(~valid_domain)),
        "grid_effective_domain_mask": "outside-checkpoint-quarter-wells",
    }
    for lev in contour_levels:
        out.update(contour_distance_metrics(Xg, Yg, s_true_grid, s_pred_grid, lev))
    # Grid plume metrics are useful when point sampling is nonuniform.
    # Raw-S metrics are retained as legacy diagnostics; dS metrics are preferred
    # when S_IC_CO2 is nonzero.
    out.update({f"grid_{k}": v for k, v in plume_metrics(s_pred_grid.reshape(-1), s_true_grid.reshape(-1), plume_thresholds).items()})
    out.update({f"grid_{k}": v for k, v in plume_metrics_excess(s_pred_grid.reshape(-1), s_true_grid.reshape(-1), s_ic_co2, excess_plume_thresholds).items()})
    for lev in excess_plume_thresholds:
        out.update(contour_distance_metrics_excess(Xg, Yg, s_true_grid, s_pred_grid, s_ic_co2, lev))
    out.update(speckle_grid_metrics(s_true_grid, s_pred_grid, speckle_pred_002, speckle_true_bg_002, min_component_pixels, "002"))
    out.update(speckle_grid_metrics(s_true_grid, s_pred_grid, speckle_pred_005, speckle_true_bg_005, min_component_pixels, "005"))
    out.update(speckle_grid_metrics_excess(s_true_grid, s_pred_grid, s_ic_co2, speckle_pred_002, speckle_true_bg_002, min_component_pixels, "002"))
    out.update(speckle_grid_metrics_excess(s_true_grid, s_pred_grid, s_ic_co2, speckle_pred_005, speckle_true_bg_005, min_component_pixels, "005"))
    # Background predicted saturation diagnostics.
    bg = np.isfinite(s_true_grid) & np.isfinite(s_pred_grid) & (s_true_grid < speckle_true_bg_005)
    if np.any(bg):
        vals = s_pred_grid[bg]
        out.update({
            "background_S_mean_pred": float(np.mean(vals)),
            "background_S_p95_pred": float(np.percentile(vals, 95.0)),
            "background_S_p99_pred": float(np.percentile(vals, 99.0)),
            "background_S_max_pred": float(np.max(vals)),
        })
    else:
        out.update({"background_S_mean_pred": np.nan, "background_S_p95_pred": np.nan, "background_S_p99_pred": np.nan, "background_S_max_pred": np.nan})
    return out


# =============================================================================
# Run evaluation
# =============================================================================
def evaluate_run_front_plume(spec, eval_plan, full_reference_layers, device, batch_size, args):
    state, ckpt_cfg = load_checkpoint(spec.ckpt_path)
    cfg = default_cfg()
    apply_exp_preset(cfg, spec.exp_name)
    cfg_update_from_ckpt(cfg, ckpt_cfg)
    if args.dtype:
        cfg["dtype"] = args.dtype
    s_ic_eval = float(getattr(args, "s_ic_co2", DEFAULT_S_IC_CO2_EVAL))
    if "s_ic_co2" in cfg:
        s_ic_eval = float(cfg.get("s_ic_co2", s_ic_eval))
    model = build_model(cfg)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    model.set_beta(float(cfg.get("beta_eval", 1.0)))

    # Use the identical deterministic interpolation sample for every model so
    # cross-role differences cannot be caused by process-randomized hash salts.
    rng = np.random.default_rng(int(args.random_seed))
    front_rows = []
    plume_rows = []
    pair_rows = []
    speckle_rows = []

    front_low_wide = float(cfg.get("front_low", args.front_low_wide))
    front_high_wide = float(cfg.get("front_high", args.front_high_wide))
    pairgrad_s_min = float(cfg.get("pairgrad_s_min", args.pairgrad_s_min))
    pairgrad_s_max = float(cfg.get("pairgrad_s_max", args.pairgrad_s_max))
    pairgrad_front_center = float(cfg.get("pairgrad_front_center", cfg.get("front_center", args.pairgrad_front_center)))
    pairgrad_front_sigma = float(cfg.get("pairgrad_front_sigma", cfg.get("front_sigma", args.pairgrad_front_sigma)))
    pairgrad_front_weight_floor = float(cfg.get("pairgrad_front_weight_floor", args.pairgrad_front_weight_floor))
    thresholds = parse_float_list(args.plume_thresholds)
    if "plume_threshold" in cfg:
        plume_threshold = float(cfg["plume_threshold"])
        thresholds = [plume_threshold] + [v for v in thresholds if abs(float(v) - plume_threshold) > 1.0e-12]
    excess_thresholds = parse_float_list(args.excess_plume_thresholds)
    contour_levels = parse_float_list(args.contour_levels)

    for k, idx in enumerate(eval_plan["indices_per_time"]):
        if idx.size == 0:
            continue
        x = eval_plan["x"][idx]
        y = eval_plan["y"][idx]
        t = eval_plan["t"][idx]
        s_true = eval_plan["s"][idx]
        _, s_pred = predict_model(model, x, y, t, cfg, device, batch_size)
        base = dict(
            run_id=spec.run_id,
            label=short_label(spec, loo_labels=bool(args.loo_labels)),
            exp_name=spec.exp_name,
            training_strategy=spec.training_strategy,
            leave_out=spec.leave_out,
            seed=spec.seed,
            ckpt_path=spec.ckpt_path,
            time_index=k,
            snapshot_label=eval_plan["selected_map"].get(k, ""),
            time_s=float(eval_plan["time_seconds"][k]),
            time_days=float(eval_plan["time_seconds"][k] / 86400.0),
            n_used=int(idx.size),
        )
        frow = dict(base)
        wide_metrics = front_band_metrics(s_pred, s_true, front_low_wide, front_high_wide, "wide_005_030")
        narrow_metrics = front_band_metrics(s_pred, s_true, float(args.front_low_narrow), float(args.front_high_narrow), "narrow_010_025")
        frow.update(wide_metrics)
        frow.update(narrow_metrics)
        # Backward-compatible aliases for previously published CSV readers.
        frow.update(metric_aliases(wide_metrics, "wide_005_030", "wide_020_060"))
        frow.update(metric_aliases(narrow_metrics, "narrow_010_025", "narrow_021_041"))
        front_rows.append(frow)

        prow = dict(base)
        prow.update(plume_metrics(s_pred, s_true, thresholds))
        prow.update(plume_metrics_excess(s_pred, s_true, s_ic_eval, excess_thresholds))
        plume_rows.append(prow)

        pg = dict(base)
        pg.update(pairgrad_metrics_xy(
            x=x / float(cfg.get("L_ref", DEFAULT_L_REF)),
            y=y / float(cfg.get("L_ref", DEFAULT_L_REF)),
            s_true=s_true,
            s_pred=s_pred,
            rng=rng,
            max_pairs=int(args.pairgrad_max_pairs_per_time),
            s_min=pairgrad_s_min,
            s_max=pairgrad_s_max,
            k_neighbors=int(args.pairgrad_k_neighbors),
            min_dist=float(args.pairgrad_min_dist),
            max_dist=float(args.pairgrad_max_dist),
            clip_value=float(args.pairgrad_clip),
            front_center=pairgrad_front_center,
            front_sigma=pairgrad_front_sigma,
            front_weight_floor=pairgrad_front_weight_floor,
        ))
        pair_rows.append(pg)

        # Dense-grid contour/speckle diagnostics only for selected snapshot times.
        if k in eval_plan["selected_map"]:
            if k not in full_reference_layers:
                raise ValueError(f"Full contour reference layer {k} was not prepared.")
            x_full, y_full, s_true_full = full_reference_layers[k]
            grow = dict(base)
            grow.update(grid_front_plume_diagnostics(
                model=model,
                cfg=cfg,
                device=device,
                batch_size=batch_size,
                x=x_full,
                y=y_full,
                t_value=float(eval_plan["time_seconds"][k]),
                s_true_points=s_true_full,
                nx=int(args.grid_nx),
                ny=int(args.grid_ny),
                max_interp_points=int(args.max_interp_input_points),
                rng=rng,
                contour_levels=contour_levels,
                plume_thresholds=thresholds,
                excess_plume_thresholds=excess_thresholds,
                s_ic_co2=s_ic_eval,
                speckle_pred_002=float(args.speckle_pred_threshold_002),
                speckle_true_bg_002=float(args.speckle_true_background_002),
                speckle_pred_005=float(args.speckle_pred_threshold_005),
                speckle_true_bg_005=float(args.speckle_true_background_005),
                min_component_pixels=int(args.speckle_min_component_pixels),
            ))
            grow["contour_reference_unique_points"] = int(x_full.size)
            grow["contour_reference_mapping"] = "full-unique-native-nearest-to-fixed-grid"
            speckle_rows.append(grow)
    return front_rows, plume_rows, pair_rows, speckle_rows, cfg


def aggregate_summary(rows: list[dict[str, Any]], id_cols: list[str], prefixes_or_cols: list[str]) -> list[dict[str, Any]]:
    import pandas as pd
    if not rows:
        return []
    df = pd.DataFrame(rows)
    out = []
    for run_id, g in df.groupby("run_id", sort=False):
        first = g.iloc[0]
        row = {c: first.get(c, "") for c in id_cols if c in g.columns}
        row["run_id"] = run_id
        for col in g.columns:
            if col in row or col in ("ckpt_path",):
                continue
            if not any(col.startswith(p) or col == p for p in prefixes_or_cols):
                continue
            vals = pd.to_numeric(g[col], errors="coerce")
            if vals.notna().any():
                row[f"{col}_mean"] = float(vals.mean())
                row[f"{col}_median"] = float(vals.median())
                row[f"{col}_min"] = float(vals.min())
                row[f"{col}_max"] = float(vals.max())
        out.append(row)
    return out


def merge_run_summaries(
    primary: list[dict[str, Any]], additional: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Merge disjoint metric families without emitting duplicate run rows."""
    merged: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for source in (primary, additional):
        for row in source:
            run_id = str(row.get("run_id", "")).strip()
            if not run_id:
                raise ValueError("Summary row is missing a nonempty run_id.")
            if run_id not in merged:
                merged[run_id] = dict(row)
                order.append(run_id)
                continue
            target = merged[run_id]
            for key, value in row.items():
                if key in target and key != "run_id" and target[key] != value:
                    raise ValueError(
                        f"Conflicting summary identity/value for run_id={run_id!r}, key={key!r}."
                    )
                target[key] = value
    return [merged[run_id] for run_id in order]


# =============================================================================
# Plotting
# =============================================================================
def plot_lines(rows: list[dict[str, Any]], out_path: str, metric: str, ylabel: str, time_unit: str, dpi: int, export_pdf: bool, export_tiff: bool, title: str | None = None):
    import pandas as pd
    if not rows:
        return
    df = pd.DataFrame(rows)
    if metric not in df.columns:
        return
    set_journal_style()
    fig, ax = plt.subplots(figsize=(3.55, 2.45), constrained_layout=False)
    plotted = {}
    for label, g in df.groupby("label", sort=False):
        g = g.sort_values("time_s")
        tdisp = np.asarray([time_to_display(v, time_unit) for v in g["time_s"].to_numpy(dtype=float)], dtype=float)
        y = pd.to_numeric(g[metric], errors="coerce").to_numpy(dtype=float)
        plotted[str(label)] = (tdisp, y)
        ax.plot(tdisp, y, label=label, color=color_for_label(label, None))
    if title:
        ax.set_title(title, pad=2)
    ax.set_xlabel(time_axis_label(time_unit))
    ax.set_ylabel(ylabel)
    apply_nature_axis(ax, grid=True, grid_axis="both")
    apply_robust_time_axis(
        ax,
        plotted,
        x_cut=transient_cut_for_unit(time_unit),
        color_for_label=color_for_label,
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.31), ncol=min(5, len(df["label"].unique())), frameon=False, columnspacing=1.0, handlelength=1.6)
    fig.subplots_adjust(left=0.17, right=0.98, top=0.94, bottom=0.36)
    savefig(fig, out_path, dpi, export_pdf, export_tiff)
    plt.close(fig)


def plot_overview(front_rows, plume_rows, pair_rows, speckle_rows, front_dir, plume_dir, time_unit, dpi, export_pdf, export_tiff):
    import pandas as pd
    set_journal_style()
    ensure_dir(os.path.join(front_dir, "figures"))
    ensure_dir(os.path.join(plume_dir, "figures"))
    # Single metric curves.
    plot_lines(front_rows, os.path.join(front_dir, "figures", "front_band_RMSE_wide_vs_time"), "front_band_wide_005_030_rmse", "Front-band S RMSE", time_unit, dpi, export_pdf, export_tiff)
    plot_lines(pair_rows, os.path.join(front_dir, "figures", "pairgrad_RMSE_vs_time"), "pairgrad_RMSE", "Pair-gradient RMSE", time_unit, dpi, export_pdf, export_tiff)
    plot_lines(plume_rows, os.path.join(plume_dir, "figures", "plume_IoU_dS005_vs_time"), "plume_IoU_dS005", "Plume IoU (dS>0.05)", time_unit, dpi, export_pdf, export_tiff)
    plot_lines(plume_rows, os.path.join(plume_dir, "figures", "FP_area_ratio_dS005_vs_time"), "FP_area_ratio_dS005", "FP area / true plume", time_unit, dpi, export_pdf, export_tiff)
    plot_lines(plume_rows, os.path.join(plume_dir, "figures", "FN_area_ratio_dS005_vs_time"), "FN_area_ratio_dS005", "FN area / true plume", time_unit, dpi, export_pdf, export_tiff)
    plot_lines(speckle_rows, os.path.join(front_dir, "figures", "speckle_area_ratio_dS002_selected_times"), "speckle_area_ratio_dS002", "Speckle area ratio", time_unit, dpi, export_pdf, export_tiff)
    # Legacy raw-S figures retained for diagnostic compatibility.
    plot_lines(plume_rows, os.path.join(plume_dir, "figures", "plume_IoU_S005_vs_time"), "plume_IoU_S005", "Raw plume IoU (S>0.05)", time_unit, dpi, export_pdf, export_tiff)

    # Six-panel overview; use separate dataframes because some diagnostics are selected-time only.
    panels = [
        (front_rows, "front_band_wide_005_030_rmse", "Front-band RMSE"),
        (pair_rows, "pairgrad_RMSE", "Pair-gradient RMSE"),
        (plume_rows, "plume_IoU_dS005", "Plume IoU (dS>0.05)"),
        (plume_rows, "FP_area_ratio_dS005", "FP area / true plume"),
        (plume_rows, "FN_area_ratio_dS005", "FN area / true plume"),
        (speckle_rows, "speckle_area_ratio_dS002", "Speckle area ratio"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(7.4, 4.6), constrained_layout=False)
    handles, labels = None, None
    for ax, (rows, metric, ylabel) in zip(axes.ravel(), panels):
        if not rows:
            ax.set_visible(False); continue
        df = pd.DataFrame(rows)
        if metric not in df.columns:
            ax.set_visible(False); continue
        plotted = {}
        for label, g in df.groupby("label", sort=False):
            g = g.sort_values("time_s")
            tdisp = np.asarray([time_to_display(v, time_unit) for v in g["time_s"].to_numpy(dtype=float)], dtype=float)
            y = pd.to_numeric(g[metric], errors="coerce").to_numpy(dtype=float)
            plotted[str(label)] = (tdisp, y)
            ax.plot(tdisp, y, label=label, color=color_for_label(label, None))
        ax.set_ylabel(ylabel)
        ax.set_xlabel(time_axis_label(time_unit))
        apply_nature_axis(ax, grid=True, grid_axis="both")
        apply_robust_time_axis(
            ax,
            plotted,
            x_cut=transient_cut_for_unit(time_unit),
            color_for_label=color_for_label,
        )
        handles, labels = ax.get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.015), ncol=min(6, len(labels)), frameon=False, columnspacing=1.05, handlelength=1.5)
    fig.subplots_adjust(left=0.075, right=0.99, top=0.965, bottom=0.22, wspace=0.42, hspace=0.45)
    base = os.path.join(front_dir, "figures", "front_plume_overview")
    savefig(fig, base, dpi, export_pdf, export_tiff)
    # Duplicate the overview under plume/figures for convenience.
    savefig(fig, os.path.join(plume_dir, "figures", "front_plume_overview"), dpi, export_pdf, export_tiff)
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================
def main():
    ap = argparse.ArgumentParser(description="Front + plume evaluator for M0-M7 and M7 leave-one-out checkpoints.")
    ap.add_argument("--ckpt-dir", type=str, required=True)
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--analysis-t-max-s", type=float, default=ANALYSIS_T_MAX_S,
                    help="Frozen H/K common physical-time cutoff in seconds.")
    ap.add_argument("--out-root", type=str, required=True, help="Root eval directory; front/ and plume/ are created inside this directory.")
    ap.add_argument("--experiments", type=str, default=",".join(ALL_EXPS))
    ap.add_argument("--training-strategy", type=str, default="BASE")
    ap.add_argument("--include-loo", type=int, default=0)
    ap.add_argument("--loo-exp", type=str, default="M7")
    ap.add_argument("--leave-outs", type=str, default="NONE,PLAIN_TWONET,FRONT_PLUME,PAIRGRAD,RAR,FV,COARSE_DETAIL")
    ap.add_argument("--loo-labels", type=int, default=0, help="If 1, label M7 LOO-NONE as L0 in figures.")
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
    ap.add_argument("--grid-nx", type=int, default=DEFAULT_GRID_NX)
    ap.add_argument("--grid-ny", type=int, default=DEFAULT_GRID_NY)
    ap.add_argument("--max-interp-input-points", type=int, default=DEFAULT_MAX_INTERP_INPUT_POINTS,
                    help="Legacy compatibility option; formal contour truth always uses the full unique layer.")
    ap.add_argument("--front-low-wide", type=float, default=DEFAULT_FRONT_LOW_WIDE)
    ap.add_argument("--front-high-wide", type=float, default=DEFAULT_FRONT_HIGH_WIDE)
    ap.add_argument("--front-low-narrow", type=float, default=DEFAULT_FRONT_LOW_NARROW)
    ap.add_argument("--front-high-narrow", type=float, default=DEFAULT_FRONT_HIGH_NARROW)
    ap.add_argument("--contour-levels", type=str, default=",".join(str(v) for v in DEFAULT_CONTOUR_LEVELS))
    ap.add_argument("--plume-thresholds", type=str, default=",".join(str(v) for v in DEFAULT_PLUME_THRESHOLDS), help="Raw-S thresholds. With S_IC_CO2=0, these are the primary plume thresholds; with nonzero S_IC_CO2, use dS metrics.")
    ap.add_argument("--excess-plume-thresholds", type=str, default=",".join(str(v) for v in DEFAULT_EXCESS_PLUME_THRESHOLDS), help="Preferred thresholds applied to dS = S - S_IC_CO2.")
    ap.add_argument("--s-ic-co2", type=float, default=DEFAULT_S_IC_CO2_EVAL)
    ap.add_argument("--pairgrad-max-pairs-per-time", type=int, default=DEFAULT_PAIRGRAD_MAX_PAIRS_PER_TIME)
    ap.add_argument("--pairgrad-s-min", type=float, default=DEFAULT_PAIRGRAD_S_MIN)
    ap.add_argument("--pairgrad-s-max", type=float, default=DEFAULT_PAIRGRAD_S_MAX)
    ap.add_argument("--pairgrad-k-neighbors", type=int, default=DEFAULT_PAIRGRAD_K_NEIGHBORS)
    ap.add_argument("--pairgrad-min-dist", type=float, default=DEFAULT_PAIRGRAD_MIN_DIST)
    ap.add_argument("--pairgrad-max-dist", type=float, default=DEFAULT_PAIRGRAD_MAX_DIST)
    ap.add_argument("--pairgrad-clip", type=float, default=DEFAULT_PAIRGRAD_CLIP)
    ap.add_argument("--pairgrad-front-center", type=float, default=DEFAULT_PAIRGRAD_FRONT_CENTER)
    ap.add_argument("--pairgrad-front-sigma", type=float, default=DEFAULT_PAIRGRAD_FRONT_SIGMA)
    ap.add_argument("--pairgrad-front-weight-floor", type=float, default=DEFAULT_PAIRGRAD_FRONT_WEIGHT_FLOOR)
    ap.add_argument("--speckle-pred-threshold-002", type=float, default=DEFAULT_SPECKLE_PRED_002)
    ap.add_argument("--speckle-true-background-002", type=float, default=DEFAULT_SPECKLE_TRUE_BG_002)
    ap.add_argument("--speckle-pred-threshold-005", type=float, default=DEFAULT_SPECKLE_PRED_005)
    ap.add_argument("--speckle-true-background-005", type=float, default=DEFAULT_SPECKLE_TRUE_BG_005)
    ap.add_argument("--speckle-min-component-pixels", type=int, default=DEFAULT_SPECKLE_MIN_COMPONENT_PIXELS)
    ap.add_argument("--figure-dpi", type=int, default=DEFAULT_FIGURE_DPI)
    ap.add_argument("--export-pdf", type=int, default=int(DEFAULT_EXPORT_PDF))
    ap.add_argument("--export-tiff", type=int, default=int(DEFAULT_EXPORT_TIFF))
    ap.add_argument("--export-svg", type=int, default=int(DEFAULT_EXPORT_SVG),
                    help="Export editable SVG files in addition to PNG/PDF/TIFF.")
    args = ap.parse_args()
    if float(args.analysis_t_max_s) != ANALYSIS_T_MAX_S:
        raise ValueError(
            f"Configuration K analysis_t_max_s is frozen at {ANALYSIS_T_MAX_S}; "
            f"got {args.analysis_t_max_s}."
        )
    global EXPORT_SVG
    EXPORT_SVG = bool(args.export_svg)

    front_dir = os.path.join(args.out_root, "front")
    plume_dir = os.path.join(args.out_root, "plume")
    ensure_dir(front_dir); ensure_dir(plume_dir)
    ensure_dir(os.path.join(front_dir, "figures")); ensure_dir(os.path.join(plume_dir, "figures"))

    device = torch.device(f"cuda:{args.gpu_id}" if args.device.lower().startswith("cuda") and torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")
    arrays, time_unique = load_dataset_pack(args.data, analysis_t_max_s=args.analysis_t_max_s)
    eval_plan = build_eval_plan(arrays, time_unique, args.sample_per_time, args.max_time_steps, args.max_sort_n, args.random_seed, args.time_indices)
    full_reference_layers = build_full_reference_layers(
        arrays,
        eval_plan["time_seconds"],
        selected_indices=set(eval_plan["selected_map"]),
    )
    print(f"[time] n={len(eval_plan['time_seconds'])}, selected={eval_plan['selected_items']}")

    experiments = [x.strip().upper() for x in args.experiments.split(",") if x.strip()]
    leave_outs = [x.strip().upper() for x in args.leave_outs.split(",") if x.strip()]
    seed = None if args.seed.upper() == "ANY" else args.seed
    specs = make_run_specs(args.ckpt_dir, experiments, args.training_strategy.upper(), bool(args.include_loo), args.loo_exp.upper(), leave_outs, seed)
    if not specs:
        raise RuntimeError("No checkpoints found. Check --ckpt-dir, --experiments, --training-strategy, and --leave-outs.")
    print(f"[runs] {len(specs)}")
    for s in specs:
        print(f"  - {short_label(s, loo_labels=bool(args.loo_labels))}: {s.ckpt_path}")

    all_front, all_plume, all_pair, all_speckle = [], [], [], []
    resolved_cfgs = {}
    for spec in specs:
        print(f"[eval] {spec.run_id}")
        fr, pr, pg, sp, cfg = evaluate_run_front_plume(
            spec,
            eval_plan,
            full_reference_layers,
            device,
            int(args.infer_batch_size),
            args,
        )
        all_front.extend(fr); all_plume.extend(pr); all_pair.extend(pg); all_speckle.extend(sp)
        resolved_cfgs[spec.run_id] = {k: v for k, v in cfg.items() if k != "checkpoint_config"}

    id_cols = ["run_id", "label", "exp_name", "training_strategy", "leave_out", "seed"]
    validate_canonical_contour_support(all_speckle)
    front_summary = aggregate_summary(all_front + all_speckle, id_cols, ["front_band_", "contour_"])
    # Add pair summaries into front summary by appending rows if needed; easier to keep separate pairgrad file.
    pair_summary = aggregate_summary(all_pair, id_cols, ["pairgrad_"])
    plume_summary = aggregate_summary(all_plume, id_cols, ["plume_", "FP_", "FN_"])

    write_csv(os.path.join(front_dir, "by_time_front_resolution.csv"), sorted(all_front, key=lambda r: (r["run_id"], int(r["time_index"]))))
    write_csv(os.path.join(front_dir, "pairgrad_metrics.csv"), sorted(all_pair, key=lambda r: (r["run_id"], int(r["time_index"]))))
    write_csv(os.path.join(front_dir, "speckle_metrics.csv"), sorted(all_speckle, key=lambda r: (r["run_id"], int(r["time_index"]))))
    write_csv(os.path.join(plume_dir, "by_time_plume_support.csv"), sorted(all_plume, key=lambda r: (r["run_id"], int(r["time_index"]))))
    combined_front_summary = merge_run_summaries(front_summary, pair_summary)
    write_csv(os.path.join(front_dir, "summary_front_resolution.csv"), sorted(combined_front_summary, key=lambda r: r.get("run_id", "")))
    write_csv(os.path.join(plume_dir, "summary_plume_support.csv"), sorted(plume_summary, key=lambda r: r.get("run_id", "")))

    save_json({
        "args": vars(args),
        "selected_time_items": eval_plan["selected_items"],
        "front_definitions": {
            "wide_front_band": [args.front_low_wide, args.front_high_wide],
            "narrow_front_band": [args.front_low_narrow, args.front_high_narrow],
            "contour_levels": parse_float_list(args.contour_levels),
            "contour_reference_mapping": "full duplicate-collapsed native layer to fixed grid by nearest neighbour",
            "both_missing_policy": "allowed per time; every run must contain at least one informative S=0.175 comparison",
        },
        "plume_definitions": {
            "raw_thresholds_legacy": parse_float_list(args.plume_thresholds),
            "excess_thresholds_preferred": parse_float_list(args.excess_plume_thresholds),
            "dS_definition": "dS = Sco2 - S_IC_CO2",
            "s_ic_co2": float(args.s_ic_co2),
        },
        "speckle_definitions": {
            "speckle_002_raw_legacy": f"S_pred>{args.speckle_pred_threshold_002} and S_true<{args.speckle_true_background_002}",
            "speckle_005_raw_legacy": f"S_pred>{args.speckle_pred_threshold_005} and S_true<{args.speckle_true_background_005}",
            "speckle_dS002_preferred": f"dS_pred>{args.speckle_pred_threshold_002} and dS_true<{args.speckle_true_background_002}",
            "speckle_dS005_preferred": f"dS_pred>{args.speckle_pred_threshold_005} and dS_true<{args.speckle_true_background_005}",
        },
        "Sco2_initial_default": DEFAULT_S_IC_CO2,
        "CO2_residual_saturation_default": DEFAULT_SNR,
        "resolved_configs": resolved_cfgs,
    }, os.path.join(front_dir, "front_plume_evaluation_config.json"))

    plot_overview(all_front, all_plume, all_pair, all_speckle, front_dir, plume_dir, args.time_unit, int(args.figure_dpi), bool(args.export_pdf), bool(args.export_tiff))
    print(f"[done] outputs written to {args.out_root}/front and {args.out_root}/plume")


if __name__ == "__main__":
    main()
