#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Nature-style snapshot exporter for M0-M7 PINN checkpoints and M7 leave-one-out runs.

Outputs are placed under:
  eval_M0_M7/snapshots/<M0..M7 or L0..L5>/<t0|early|middle|late|final>/

Key design choices
------------------
- Uses dense regular-grid rendering for PINN and reference fields.
- Uses the same time-layer selector as the other M0-M7 evaluators:
  t0, early, middle, late, final.
- Handles the physical initial CO2 saturation S_IC_CO2=0.0 from the training script.
- Uses excess saturation dS = S - S_IC_CO2 by default for plume masks,
  leading-edge contours, FP/FN maps and speckle diagnostics. This avoids the
  convention remains explicit if a future dataset starts from nonzero CO2 saturation.
- Raw saturation is still used for the main front contour S=0.175 and for basic
  S_true/S_pred/S_error fields.
- Empty masks are explicitly annotated as "none detected" rather than exported
  as visually blank figures.
"""
from __future__ import annotations

import os
import json
import math
import argparse
from typing import Any

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

try:
    from _nature_plot_style_M0_M7 import setup_nature_rcparams, save_pub_figure
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
        if export_svg:
            fig.savefig(str(base_without_ext) + ".svg", bbox_inches="tight", pad_inches=pad_inches)

# Reuse the M0-M7 model/checkpoint/time utilities from the canonical global evaluator.
from evaluate_global_accuracy_M0_M7_L6_dS_shortlabels import (
    ALL_EXPS,
    RunSpec,
    default_cfg,
    load_dataset_pack,
    load_checkpoint,
    apply_exp_preset,
    cfg_update_from_ckpt,
    build_model,
    predict_model,
    parse_time_indices,
    resolve_snapshot_items,
    make_run_specs,
    dtype_from_cfg,
    pressure_unit_factor,
    time_to_display,
)

# =============================================================================
# Defaults
# =============================================================================
DEFAULT_DATA_PATH = "./data/tables_cache_0_775_step1.pt"
DEFAULT_CKPT_DIR = "./runs"
DEFAULT_OUT_ROOT = "./evaluation_supplement/snapshots"
DEFAULT_DEVICE = "cpu"
DEFAULT_GPU_ID = 0
DEFAULT_TIME_INDICES = "t0,early,middle,late,final"
DEFAULT_TIME_UNIT = "days"
DEFAULT_GRID_NX = 500
DEFAULT_GRID_NY = 500
DEFAULT_INFER_BATCH_SIZE = 65536
DEFAULT_FIGURE_DPI = 600
DEFAULT_EXPORT_PDF = True
DEFAULT_EXPORT_TIFF = True
DEFAULT_EXPORT_SVG = True
DEFAULT_TITLE_MODE = "compact"
DEFAULT_PRESSURE_DISPLAY_UNIT = "MPa"
DEFAULT_PRESSURE_CMAP = "viridis"
DEFAULT_SATURATION_CMAP = "viridis"
DEFAULT_ERROR_CMAP = "RdBu_r"
DEFAULT_SATURATION_VMIN = 0.0
DEFAULT_SATURATION_VMAX = 0.8
DEFAULT_SATURATION_ERR_ABS = None
DEFAULT_PRESSURE_ERR_ABS = None
DEFAULT_S_IC_CO2 = 0.0
DEFAULT_SNR = 0.0
DEFAULT_SW_IRR = 0.20
DEFAULT_THRESHOLD_MODE = "excess"   # excess or raw
DEFAULT_PLUME_THRESHOLD = 0.05       # applied to dS when threshold_mode=excess
DEFAULT_LEADING_EXCESS_LEVEL = 0.05
DEFAULT_MAIN_RAW_CONTOUR_LEVEL = 0.175
DEFAULT_SPECKLE_PRED_THRESHOLD_002 = 0.02   # applied to dS by default
DEFAULT_SPECKLE_TRUE_BACKGROUND_002 = 0.01
DEFAULT_SPECKLE_PRED_THRESHOLD_005 = 0.05
DEFAULT_SPECKLE_TRUE_BACKGROUND_005 = 0.02
DEFAULT_ISOLATED_SPECKLE_THRESHOLD = 0.05
DEFAULT_ISOLATED_SPECKLE_MIN_PIXELS = 4

LOO_LABELS = {
    "NONE": "L0",
    "PLAIN_TWONET": "L1",
    "FRONT_PLUME": "L2",
    "PAIRGRAD": "L3",
    "RAR": "L4",
    "FV": "L5",
    "COARSE_DETAIL": "L6",
}

# =============================================================================
# Plotting utilities
# =============================================================================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def apply_publication_rcparams():
    setup_nature_rcparams(font_size=7.0, line_width=1.05)


def _journal_cmap(name: str, bad=(1.0, 1.0, 1.0, 1.0)):
    cmap = plt.get_cmap(str(name)).copy()
    cmap.set_bad(color=bad)
    return cmap


def save_multi(fig, root: str, args, png: bool = True):
    dpi = int(args.figure_dpi)
    save_pub_figure(fig, root, dpi=dpi, export_pdf=bool(args.export_pdf), export_tiff=bool(args.export_tiff), export_svg=bool(args.export_svg), export_png=png, pad_inches=0.025)


def style_axis(ax, L: float, xlabel=True, ylabel=True):
    ax.set_xlim(0.0, L)
    ax.set_ylim(0.0, L)
    ax.set_aspect("equal", adjustable="box")
    if xlabel:
        ax.set_xlabel(r"$x$ (m)")
    else:
        ax.set_xticklabels([])
    if ylabel:
        ax.set_ylabel(r"$y$ (m)")
    else:
        ax.set_yticklabels([])
    ax.tick_params(direction="out", length=2.5, width=0.55, pad=1.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def panel_label(ax, label: str):
    ax.text(0.025, 0.975, label, transform=ax.transAxes,
            ha="left", va="top", fontsize=7.0, fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=1.2))


def masked(a):
    return np.ma.masked_invalid(np.asarray(a, dtype=np.float64))


def format_colorbar(cb, label: str | None = None, labelpad: float = 3.0):
    """Apply compact, non-overlapping journal-style colorbar formatting."""
    cb.ax.tick_params(labelsize=6.0, width=0.5, length=2.2, pad=1.2)
    if label:
        cb.set_label(label, labelpad=labelpad, fontsize=7.0)
    cb.outline.set_linewidth(0.55)
    return cb


def compact_legend(ax, handles, loc="upper right", ncol: int = 1, outside: bool = False):
    """Short, compact legend used by all snapshot figures.

    Labels are intentionally short (Ref., PINN, TP, FP, FN) and never use
    checkpoint file names.  The default position is inside the axes because
    the domain corners are usually inactive well/background regions.  For
    crowded panels this helper keeps padding and handle lengths small.
    """
    if outside:
        return ax.legend(
            handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.13),
            ncol=ncol, frameon=False, fontsize=5.8, columnspacing=0.75,
            handlelength=1.35, handletextpad=0.35, borderpad=0.15
        )
    return ax.legend(
        handles=handles, loc=loc, ncol=ncol, frameon=True, framealpha=0.88,
        facecolor="white", edgecolor="0.80", fontsize=5.8,
        borderpad=0.18, labelspacing=0.18, handlelength=1.35,
        handletextpad=0.35, borderaxespad=0.20
    )


# =============================================================================
# Domain / interpolation / prediction
# =============================================================================
def to_np(a) -> np.ndarray:
    if isinstance(a, torch.Tensor):
        return a.detach().cpu().numpy()
    return np.asarray(a)


def get_arrays_for_time(arrays: dict[str, Any], time_value: float) -> dict[str, np.ndarray]:
    x = to_np(arrays["x"]).reshape(-1)
    y = to_np(arrays["y"]).reshape(-1)
    t = to_np(arrays["t"]).reshape(-1)
    p = to_np(arrays["p"]).reshape(-1)
    s = to_np(arrays["Sco2"]).reshape(-1)
    idx = np.isclose(t, float(time_value), rtol=0.0, atol=max(1e-9, 1e-10 * max(1.0, abs(float(time_value)))))
    if not np.any(idx):
        # fallback to nearest time bucket
        k = int(np.argmin(np.abs(t - float(time_value))))
        tv = t[k]
        idx = np.isclose(t, tv, rtol=0.0, atol=max(1e-9, 1e-10 * max(1.0, abs(float(tv)))))
    return {"x": x[idx].astype(np.float64), "y": y[idx].astype(np.float64),
            "p": p[idx].astype(np.float64), "S": np.clip(s[idx].astype(np.float64), 0.0, 1.0)}


def regular_grid(L: float, nx: int, ny: int):
    xv = np.linspace(0.0, L, int(nx))
    yv = np.linspace(0.0, L, int(ny))
    X, Y = np.meshgrid(xv, yv)
    return X, Y


def well_mask(X: np.ndarray, Y: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    r = float(cfg.get("r_well", 0.5))
    inj = cfg.get("inj_center", (0.0, 0.0))
    out = cfg.get("out_center", (float(cfg.get("L_ref", 5.0)), float(cfg.get("L_ref", 5.0))))
    xi, yi = float(inj[0]), float(inj[1])
    xo, yo = float(out[0]), float(out[1])
    return ((X - xi) ** 2 + (Y - yi) ** 2 <= r ** 2) | ((X - xo) ** 2 + (Y - yo) ** 2 <= r ** 2)


def interpolate_to_grid(x: np.ndarray, y: np.ndarray, values: np.ndarray, X: np.ndarray, Y: np.ndarray, mask: np.ndarray):
    vals = np.asarray(values, dtype=np.float64).reshape(-1)
    good = np.isfinite(x) & np.isfinite(y) & np.isfinite(vals)
    if not good.all():
        raise ValueError("Snapshot reference contains nonfinite coordinates or values.")
    # Match clean's spatial diagnostics: average exact duplicates before mapping.
    from evaluate_physics_M0_M7_L6_dS import _spatial_collapse_plan, _collapse_spatial_values
    order, starts, counts, xu, yu = _spatial_collapse_plan(x, y)
    vu = _collapse_spatial_values(vals, order, starts, counts, "snapshot reference")
    if xu.size < 3:
        raise ValueError("Snapshot requires at least three unique reference points.")
    tri = mtri.Triangulation(xu, yu)
    interp = mtri.LinearTriInterpolator(tri, vu)
    Z = np.ma.asarray(interp(X, Y)).filled(np.nan)
    # Fill remaining holes by nearest-neighbor only inside the physical square, then re-mask wells.
    if np.isnan(Z).any():
        try:
            from scipy.interpolate import NearestNDInterpolator
            nn = NearestNDInterpolator(np.column_stack([xu, yu]), vu)
            Z_near = np.asarray(nn(X, Y), dtype=np.float64)
            Z = np.where(np.isfinite(Z), Z, Z_near)
        except Exception as exc:
            raise ValueError("Snapshot nearest-neighbour fill failed.") from exc
    if not np.isfinite(Z[~mask]).all():
        raise ValueError("Snapshot has nonfinite values outside well masks.")
    Z[mask] = np.nan
    return Z


def predict_on_grid(model, cfg: dict[str, Any], X: np.ndarray, Y: np.ndarray, t_seconds: float, device: torch.device, batch_size: int):
    mask = well_mask(X, Y, cfg)
    x_flat = X.reshape(-1)
    y_flat = Y.reshape(-1)
    t_flat = np.full_like(x_flat, float(t_seconds), dtype=np.float64)
    valid = ~mask.reshape(-1)
    p_tilde = np.full(x_flat.shape, np.nan, dtype=np.float64)
    S = np.full(x_flat.shape, np.nan, dtype=np.float64)
    p_valid, s_valid = predict_model(model, x_flat[valid], y_flat[valid], t_flat[valid], cfg, device, batch_size)
    p_tilde[valid] = p_valid
    S[valid] = np.clip(s_valid, 0.0, 1.0)
    return p_tilde.reshape(X.shape), S.reshape(X.shape)


# =============================================================================
# Saturation threshold logic
# =============================================================================
def s_ic(cfg: dict[str, Any], args) -> float:
    return float(cfg.get("s_ic_co2", args.s_ic_co2))


def excess(S: np.ndarray, cfg: dict[str, Any], args) -> np.ndarray:
    return np.asarray(S, dtype=np.float64) - s_ic(cfg, args)


def threshold_field(S: np.ndarray, cfg: dict[str, Any], args, mode: str | None = None) -> np.ndarray:
    mode = (mode or args.threshold_mode).lower()
    if mode == "excess":
        return excess(S, cfg, args)
    if mode == "raw":
        return np.asarray(S, dtype=np.float64)
    raise ValueError(f"Unsupported threshold mode: {mode}")


def mask_threshold(S: np.ndarray, cfg: dict[str, Any], args, threshold: float, mode: str | None = None):
    A = threshold_field(S, cfg, args, mode=mode)
    return (A > float(threshold)) & np.isfinite(A)


def level_tag(prefix: str, value: float):
    return f"{prefix}{int(round(100.0 * float(value))):03d}"


def contour_if_valid(ax, X, Y, Z, levels, colors, linestyles, linewidths=1.0, zorder=3):
    Z = np.asarray(Z, dtype=np.float64)
    if not np.isfinite(Z).any():
        return None
    zmin, zmax = np.nanmin(Z), np.nanmax(Z)
    lev = [float(v) for v in levels if zmin <= float(v) <= zmax]
    if not lev:
        return None
    return ax.contour(X, Y, Z, levels=lev, colors=colors[:len(lev)] if isinstance(colors, list) else colors,
                      linestyles=linestyles, linewidths=linewidths, zorder=zorder)


# =============================================================================
# Figure exporters
# =============================================================================
def save_field(Z, root: str, args, L: float, title: str, cmap: str, vmin=None, vmax=None, cbar_label: str | None = None):
    fig, ax = plt.subplots(figsize=(3.35, 3.10), constrained_layout=False)
    fig.subplots_adjust(left=0.16, right=0.86, bottom=0.14, top=0.89)
    im = ax.imshow(masked(Z), origin="lower", extent=(0, L, 0, L), cmap=_journal_cmap(cmap),
                   vmin=vmin, vmax=vmax, interpolation="bilinear", aspect="equal")
    style_axis(ax, L, xlabel=True, ylabel=True)
    if args.title_mode != "none":
        ax.set_title(title, pad=3.0)
    cax = fig.add_axes([0.885, 0.16, 0.026, 0.68])
    cb = fig.colorbar(im, cax=cax)
    format_colorbar(cb, cbar_label, labelpad=3.0)
    save_multi(fig, root, args)
    plt.close(fig)


def save_front_overlay(X, Y, s_true, s_pred, cfg, args, root: str, mode: str, level: float, title: str):
    L = float(cfg["L_ref"])
    Zt = threshold_field(s_true, cfg, args, mode=mode)
    Zp = threshold_field(s_pred, cfg, args, mode=mode)
    fig, ax = plt.subplots(figsize=(3.35, 3.10), constrained_layout=False)
    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.14, top=0.88)
    ax.imshow(masked(s_true), origin="lower", extent=(0, L, 0, L), cmap=_journal_cmap("Greys"),
              vmin=0.0, vmax=1.0, interpolation="bilinear", alpha=0.16)
    contour_if_valid(ax, X, Y, Zt, [level], colors=["#000000"], linestyles="solid", linewidths=1.05, zorder=4)
    contour_if_valid(ax, X, Y, Zp, [level], colors=["#000000"], linestyles="dashed", linewidths=1.05, zorder=5)
    style_axis(ax, L, xlabel=True, ylabel=True)
    if args.title_mode != "none":
        ax.set_title(title, pad=3.0)
    handles = [Line2D([0], [0], color="black", lw=1.0, ls="solid", label="Ref."),
               Line2D([0], [0], color="black", lw=1.0, ls="dashed", label="PINN")]
    compact_legend(ax, handles, loc="upper right")
    save_multi(fig, root, args)
    plt.close(fig)


def save_front_overlay_multi(X, Y, s_true, s_pred, cfg, args, root: str, title: str):
    L = float(cfg["L_ref"])
    fig, ax = plt.subplots(figsize=(3.35, 3.10), constrained_layout=False)
    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.14, top=0.88)
    ax.imshow(masked(s_true), origin="lower", extent=(0, L, 0, L), cmap=_journal_cmap("Greys"),
              vmin=0.0, vmax=1.0, interpolation="bilinear", alpha=0.16)
    lead = float(args.leading_excess_level)
    main = float(args.main_raw_contour_level)
    contour_if_valid(ax, X, Y, excess(s_true, cfg, args), [lead], colors=["#000000"], linestyles="solid", linewidths=1.05, zorder=4)
    contour_if_valid(ax, X, Y, excess(s_pred, cfg, args), [lead], colors=["#000000"], linestyles="dashed", linewidths=1.05, zorder=5)
    contour_if_valid(ax, X, Y, s_true, [main], colors=["#D55E00"], linestyles="solid", linewidths=1.05, zorder=4)
    contour_if_valid(ax, X, Y, s_pred, [main], colors=["#D55E00"], linestyles="dashed", linewidths=1.05, zorder=5)
    style_axis(ax, L, xlabel=True, ylabel=True)
    if args.title_mode != "none":
        ax.set_title(title, pad=3.0)
    handles = [
        Line2D([0], [0], color="black", lw=1.0, ls="solid", label="Ref."),
        Line2D([0], [0], color="black", lw=1.0, ls="dashed", label="PINN"),
        Line2D([0], [0], color="black", lw=1.0, ls="solid", label=rf"$\Delta S={lead:g}$"),
        Line2D([0], [0], color="#D55E00", lw=1.0, ls="solid", label=rf"$S={main:g}$"),
    ]
    compact_legend(ax, handles, loc="upper right")
    save_multi(fig, root, args)
    plt.close(fig)


def binary_mask_plot(mask, cfg, args, root: str, title: str, color=(0.80, 0.10, 0.10, 1.0)):
    L = float(cfg["L_ref"])
    m = np.asarray(mask, dtype=bool)
    arr = np.where(m, 1.0, np.nan)
    cmap = ListedColormap([(1, 1, 1, 0), color])
    fig, ax = plt.subplots(figsize=(3.35, 3.10), constrained_layout=False)
    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.14, top=0.88)
    ax.imshow(np.zeros_like(arr), origin="lower", extent=(0, L, 0, L), cmap=ListedColormap(["#FAFAFA"]), vmin=0, vmax=1)
    ax.imshow(np.ma.masked_invalid(arr), origin="lower", extent=(0, L, 0, L), cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
    style_axis(ax, L, xlabel=True, ylabel=True)
    if args.title_mode != "none":
        ax.set_title(title, pad=3.0)
    n = int(np.sum(m))
    if n == 0:
        ax.text(0.5, 0.5, "none detected", transform=ax.transAxes, ha="center", va="center", fontsize=7.0, color="0.35",
                bbox=dict(facecolor="white", edgecolor="0.85", boxstyle="round,pad=0.23", alpha=0.92))
    else:
        ax.text(0.985, 0.02, f"n={n}", transform=ax.transAxes, ha="right", va="bottom",
                fontsize=6.0, bbox=dict(facecolor="white", edgecolor="none", alpha=0.78, pad=1.0))
    save_multi(fig, root, args)
    plt.close(fig)


def mask_categories(s_true, s_pred, cfg, args):
    th = float(args.plume_threshold)
    tm = mask_threshold(s_true, cfg, args, th, mode=args.threshold_mode)
    pm = mask_threshold(s_pred, cfg, args, th, mode=args.threshold_mode)
    fp = (~tm) & pm & np.isfinite(s_true) & np.isfinite(s_pred)
    fn = tm & (~pm) & np.isfinite(s_true) & np.isfinite(s_pred)
    tp = tm & pm
    cat = np.zeros_like(s_true, dtype=np.float64)
    cat[tp] = 1.0
    cat[fp] = 2.0
    cat[fn] = 3.0
    cat[~np.isfinite(s_true) | ~np.isfinite(s_pred)] = np.nan
    return cat, tp, fp, fn


def save_plume_overlay(X, Y, s_true, s_pred, cfg, args, root: str, title: str):
    L = float(cfg["L_ref"])
    cat, tp, fp, fn = mask_categories(s_true, s_pred, cfg, args)
    cmap = ListedColormap(["white", "#D9D9D9", "#E69F00", "#56B4E9"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
    fig, ax = plt.subplots(figsize=(3.35, 3.10), constrained_layout=False)
    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.14, top=0.88)
    ax.imshow(np.ma.masked_invalid(cat), origin="lower", extent=(0, L, 0, L), cmap=cmap, norm=norm, interpolation="nearest")
    Zt = threshold_field(s_true, cfg, args, mode=args.threshold_mode)
    Zp = threshold_field(s_pred, cfg, args, mode=args.threshold_mode)
    contour_if_valid(ax, X, Y, Zt, [float(args.plume_threshold)], colors=["black"], linestyles="solid", linewidths=1.0, zorder=4)
    contour_if_valid(ax, X, Y, Zp, [float(args.plume_threshold)], colors=["black"], linestyles="dashed", linewidths=1.0, zorder=5)
    style_axis(ax, L, xlabel=True, ylabel=True)
    if args.title_mode != "none":
        ax.set_title(title, pad=3.0)
    handles = [Line2D([0], [0], color="black", lw=1.0, ls="solid", label="Ref."),
               Line2D([0], [0], color="black", lw=1.0, ls="dashed", label="PINN")]
    compact_legend(ax, handles, loc="upper right")
    save_multi(fig, root, args)
    plt.close(fig)


def save_fpfn_combined(s_true, s_pred, cfg, args, root: str, title: str):
    L = float(cfg["L_ref"])
    cat, tp, fp, fn = mask_categories(s_true, s_pred, cfg, args)
    colors = ["white", "#BDBDBD", "#D55E00", "#0072B2"]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
    fig, ax = plt.subplots(figsize=(3.35, 3.10), constrained_layout=False)
    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.14, top=0.88)
    ax.imshow(np.ma.masked_invalid(cat), origin="lower", extent=(0, L, 0, L), cmap=cmap, norm=norm, interpolation="nearest")
    style_axis(ax, L, xlabel=True, ylabel=True)
    if args.title_mode != "none":
        ax.set_title(title, pad=3.0)
    compact_legend(ax, [Patch(facecolor=colors[1], label="TP"), Patch(facecolor=colors[2], label="FP"), Patch(facecolor=colors[3], label="FN")], loc="upper right")
    save_multi(fig, root, args)
    plt.close(fig)


def label_components(mask: np.ndarray):
    try:
        from scipy import ndimage
        return ndimage.label(mask.astype(bool), structure=np.ones((3, 3), dtype=np.int8))
    except Exception:
        return np.where(mask, 1, 0).astype(np.int32), int(1 if np.any(mask) else 0)


def isolated_speckles(s_true, s_pred, cfg, args):
    tm = mask_threshold(s_true, cfg, args, float(args.isolated_speckle_threshold), mode=args.threshold_mode)
    pm = mask_threshold(s_pred, cfg, args, float(args.isolated_speckle_threshold), mode=args.threshold_mode)
    labels, nlab = label_components(pm)
    iso = np.zeros_like(pm, dtype=bool)
    for lab in range(1, int(nlab) + 1):
        comp = labels == lab
        if int(np.sum(comp)) < int(args.isolated_speckle_min_pixels):
            continue
        if not np.any(comp & tm):
            iso |= comp
    return iso


def speckle_masks(s_true, s_pred, cfg, args):
    St = threshold_field(s_true, cfg, args, mode=args.threshold_mode)
    Sp = threshold_field(s_pred, cfg, args, mode=args.threshold_mode)
    m002 = (Sp > float(args.speckle_pred_threshold_002)) & (St < float(args.speckle_true_background_002)) & np.isfinite(Sp) & np.isfinite(St)
    m005 = (Sp > float(args.speckle_pred_threshold_005)) & (St < float(args.speckle_true_background_005)) & np.isfinite(Sp) & np.isfinite(St)
    miso = isolated_speckles(s_true, s_pred, cfg, args)
    return m002, m005, miso


def save_diagnostic_panel(X, Y, s_true, s_pred, s_err, cfg, args, root: str, title: str):
    L = float(cfg["L_ref"])
    s_vmin, s_vmax = float(args.saturation_vmin), float(args.saturation_vmax)
    err_lim = float(args.saturation_err_abs)
    cat, tp, fp, fn = mask_categories(s_true, s_pred, cfg, args)
    m002, m005, miso = speckle_masks(s_true, s_pred, cfg, args)
    fig, axes = plt.subplots(2, 3, figsize=(7.85, 4.95), constrained_layout=False)
    fig.subplots_adjust(left=0.065, right=0.855, bottom=0.095, top=0.925, wspace=0.225, hspace=0.310)
    axs = axes.ravel()
    im0 = axs[0].imshow(masked(s_true), origin="lower", extent=(0, L, 0, L), cmap=_journal_cmap(args.saturation_cmap), vmin=s_vmin, vmax=s_vmax, interpolation="bilinear")
    axs[1].imshow(masked(s_pred), origin="lower", extent=(0, L, 0, L), cmap=_journal_cmap(args.saturation_cmap), vmin=s_vmin, vmax=s_vmax, interpolation="bilinear")
    im2 = axs[2].imshow(masked(s_err), origin="lower", extent=(0, L, 0, L), cmap=_journal_cmap(args.error_cmap), vmin=-err_lim, vmax=err_lim, interpolation="bilinear")
    axs[3].imshow(masked(s_true), origin="lower", extent=(0, L, 0, L), cmap=_journal_cmap("Greys"), vmin=0, vmax=1, interpolation="bilinear", alpha=0.16)
    contour_if_valid(axs[3], X, Y, excess(s_true, cfg, args), [float(args.leading_excess_level)], colors=["#000000"], linestyles="solid", linewidths=1.0, zorder=4)
    contour_if_valid(axs[3], X, Y, excess(s_pred, cfg, args), [float(args.leading_excess_level)], colors=["#000000"], linestyles="dashed", linewidths=1.0, zorder=5)
    contour_if_valid(axs[3], X, Y, s_true, [float(args.main_raw_contour_level)], colors=["#D55E00"], linestyles="solid", linewidths=1.0, zorder=4)
    contour_if_valid(axs[3], X, Y, s_pred, [float(args.main_raw_contour_level)], colors=["#D55E00"], linestyles="dashed", linewidths=1.0, zorder=5)
    cmap_cat = ListedColormap(["white", "#BDBDBD", "#D55E00", "#0072B2"])
    norm_cat = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap_cat.N)
    axs[4].imshow(np.ma.masked_invalid(cat), origin="lower", extent=(0, L, 0, L), cmap=cmap_cat, norm=norm_cat, interpolation="nearest")
    axs[5].imshow(np.zeros_like(m005, dtype=float), origin="lower", extent=(0, L, 0, L), cmap=ListedColormap(["#FAFAFA"]), vmin=0, vmax=1)
    axs[5].imshow(np.ma.masked_invalid(np.where(m005, 1.0, np.nan)), origin="lower", extent=(0, L, 0, L), cmap=ListedColormap(["white", "#7B3294"]), vmin=0, vmax=1, interpolation="nearest")
    if int(np.sum(m005)) == 0:
        axs[5].text(0.5, 0.5, "none detected", transform=axs[5].transAxes, ha="center", va="center", fontsize=7.0, color="0.35", bbox=dict(facecolor="white", edgecolor="0.85", boxstyle="round,pad=0.22", alpha=0.92))
    titles = ["Reference $S$", "PINN $S$", r"$\Delta S$", "front contours", "TP / FP / FN", "speckle mask"]
    for i, ax in enumerate(axs):
        style_axis(ax, L, xlabel=(i >= 3), ylabel=(i in (0, 3)))
        panel_label(ax, chr(ord("a") + i))
        if args.title_mode != "none":
            ax.set_title(titles[i], pad=3.0)
    cax0 = fig.add_axes([0.882, 0.575, 0.014, 0.300])
    cb0 = fig.colorbar(im0, cax=cax0); format_colorbar(cb0, r"$S_{\mathrm{CO_2}}$ (-)", labelpad=3.0)
    cax2 = fig.add_axes([0.882, 0.135, 0.014, 0.300])
    cb2 = fig.colorbar(im2, cax=cax2); format_colorbar(cb2, r"$\Delta S$ (-)", labelpad=3.0)
    compact_legend(axs[3], [Line2D([0], [0], color="black", ls="solid", lw=1.0, label="Ref."), Line2D([0], [0], color="black", ls="dashed", lw=1.0, label="PINN")], loc="upper right")
    compact_legend(axs[4], [Patch(facecolor="#BDBDBD", label="TP"), Patch(facecolor="#D55E00", label="FP"), Patch(facecolor="#0072B2", label="FN")], loc="upper right")
    if args.title_mode != "none":
        fig.suptitle(title, y=0.992, fontsize=8.5)
    save_multi(fig, root, args)
    plt.close(fig)


def save_paper_panel(p_true, p_pred, p_err, s_true, s_pred, s_err, args, cfg, root: str, p_scale: float, p_label: str, p_err_label: str):
    """Export the 2x3 pressure/saturation paper panel.

    The previous version placed four vertical colorbars too close to one another
    at the right edge of the figure, so the colorbar tick labels and axis labels
    could overlap.  This version reserves a wider right margin and uses two
    well-separated colorbar columns: one column for field values and one column
    for error fields.  The image panels keep equal aspect ratio, while the
    colorbar labels remain readable in PNG/PDF/TIFF outputs.
    """
    L = float(cfg["L_ref"])

    # Wider canvas and a reduced right boundary for the 2x3 image grid leave
    # enough room for four colorbars without label overlap.
    fig, axes = plt.subplots(2, 3, figsize=(8.65, 4.95), constrained_layout=False)
    fig.subplots_adjust(
        left=0.065, right=0.765, bottom=0.095, top=0.925,
        wspace=0.215, hspace=0.310
    )
    axs = axes.ravel()

    p_vmin = None if args.pressure_vmin is None else float(args.pressure_vmin)
    p_vmax = None if args.pressure_vmax is None else float(args.pressure_vmax)
    p_lim = np.nanmax(np.abs(p_err * p_scale)) if args.pressure_err_abs is None else float(args.pressure_err_abs)
    p_lim = max(float(p_lim), 1e-12)
    s_lim = float(args.saturation_err_abs)

    im0 = axs[0].imshow(masked(p_true * p_scale), origin="lower", extent=(0, L, 0, L),
                        cmap=_journal_cmap(args.pressure_cmap), vmin=p_vmin, vmax=p_vmax,
                        interpolation="bilinear")
    axs[1].imshow(masked(p_pred * p_scale), origin="lower", extent=(0, L, 0, L),
                  cmap=_journal_cmap(args.pressure_cmap), vmin=p_vmin, vmax=p_vmax,
                  interpolation="bilinear")
    im2 = axs[2].imshow(masked(p_err * p_scale), origin="lower", extent=(0, L, 0, L),
                        cmap=_journal_cmap(args.error_cmap), vmin=-p_lim, vmax=p_lim,
                        interpolation="bilinear")
    im3 = axs[3].imshow(masked(s_true), origin="lower", extent=(0, L, 0, L),
                        cmap=_journal_cmap(args.saturation_cmap),
                        vmin=args.saturation_vmin, vmax=args.saturation_vmax,
                        interpolation="bilinear")
    axs[4].imshow(masked(s_pred), origin="lower", extent=(0, L, 0, L),
                  cmap=_journal_cmap(args.saturation_cmap),
                  vmin=args.saturation_vmin, vmax=args.saturation_vmax,
                  interpolation="bilinear")
    im5 = axs[5].imshow(masked(s_err), origin="lower", extent=(0, L, 0, L),
                        cmap=_journal_cmap(args.error_cmap), vmin=-s_lim, vmax=s_lim,
                        interpolation="bilinear")

    titles = ["Reference $p$", "PINN $p$", r"$\Delta p$", "Reference $S$", "PINN $S$", r"$\Delta S$"]
    for i, ax in enumerate(axs):
        style_axis(ax, L, xlabel=(i >= 3), ylabel=(i in (0, 3)))
        panel_label(ax, chr(ord("a") + i))
        if args.title_mode != "none":
            ax.set_title(titles[i], pad=3.0)

    # Two separated colorbar columns.  The first column is for field values
    # (p/S); the second is for differences (Δp/ΔS).  The gap is deliberately
    # large enough for vertical labels and tick labels.
    cbar_w = 0.014
    field_x = 0.800
    error_x = 0.905
    top_y, bot_y, cbar_h = 0.575, 0.135, 0.300

    caxp = fig.add_axes([field_x, top_y, cbar_w, cbar_h])
    cbp = fig.colorbar(im0, cax=caxp)
    format_colorbar(cbp, p_label, labelpad=3.2)

    caxpe = fig.add_axes([error_x, top_y, cbar_w, cbar_h])
    cbpe = fig.colorbar(im2, cax=caxpe)
    format_colorbar(cbpe, p_err_label, labelpad=3.2)

    caxs = fig.add_axes([field_x, bot_y, cbar_w, cbar_h])
    cbs = fig.colorbar(im3, cax=caxs)
    format_colorbar(cbs, r"$S_{\mathrm{CO_2}}$ (-)", labelpad=3.2)

    caxse = fig.add_axes([error_x, bot_y, cbar_w, cbar_h])
    cbse = fig.colorbar(im5, cax=caxse)
    format_colorbar(cbse, r"$\Delta S$ (-)", labelpad=3.2)

    save_multi(fig, root, args)
    plt.close(fig)


# =============================================================================
# Export logic
# =============================================================================
def pressure_labels(unit: str):
    u = unit.lower()
    if u == "mpa": return 1e-6, r"$p$ (MPa)", r"$\Delta p$ (MPa)"
    if u == "kpa": return 1e-3, r"$p$ (kPa)", r"$\Delta p$ (kPa)"
    return 1.0, r"$p$ (Pa)", r"$\Delta p$ (Pa)"


def export_run(run: RunSpec, label: str, arrays: dict[str, Any], time_unique: np.ndarray, args):
    base_cfg = default_cfg()
    # CLI values are fallbacks for legacy checkpoints; checkpoint config wins.
    base_cfg["s_ic_co2"] = float(args.s_ic_co2)
    base_cfg["Snr"] = float(args.Snr)
    base_cfg["Sw_irr"] = DEFAULT_SW_IRR
    apply_exp_preset(base_cfg, run.exp_name)
    state, ckpt_cfg = load_checkpoint(run.ckpt_path)
    cfg_update_from_ckpt(base_cfg, ckpt_cfg)
    base_cfg["s_ic_co2"] = float(base_cfg.get("s_ic_co2", args.s_ic_co2))
    base_cfg.setdefault("Snr", DEFAULT_SNR)
    base_cfg.setdefault("Sw_irr", DEFAULT_SW_IRR)
    cfg = base_cfg
    run_args = argparse.Namespace(**vars(args))
    if "plume_threshold" in cfg:
        run_args.plume_threshold = float(cfg["plume_threshold"])
        run_args.leading_excess_level = float(cfg["plume_threshold"])
    if "front_center" in cfg:
        run_args.main_raw_contour_level = float(cfg["front_center"])
    if "S_CO2_MAX" in cfg:
        run_args.saturation_vmax = float(cfg["S_CO2_MAX"])
    args = run_args

    device = torch.device("cpu")
    if str(args.device).lower() == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable.")
        device = torch.device(f"cuda:{int(args.gpu_id)}")
    model = build_model(cfg).to(device=device, dtype=dtype_from_cfg(cfg))
    model.load_state_dict(state, strict=True)
    model.eval()
    if hasattr(model, "set_beta"):
        model.set_beta(float(cfg.get("beta_eval", 1.0)))

    L = float(cfg["L_ref"])
    X, Y = regular_grid(L, args.grid_nx, args.grid_ny)
    mask = well_mask(X, Y, cfg)
    p_scale, p_label, p_err_label = pressure_labels(args.pressure_display_unit)
    out_base = os.path.join(args.out_root, label)
    ensure_dir(out_base)
    selected = resolve_snapshot_items(np.asarray(time_unique, dtype=np.float64), parse_time_indices(args.time_indices))
    record = {"label": label, "run_id": run.run_id, "exp_name": run.exp_name, "training_strategy": run.training_strategy, "leave_out": run.leave_out, "checkpoint": run.ckpt_path, "times": []}
    for tag, tidx in selected:
        t_seconds = float(np.asarray(time_unique, dtype=np.float64)[tidx])
        t_disp = time_to_display(t_seconds, args.time_unit)
        t_label = f"{t_disp:.3g} {args.time_unit}"
        tdir = os.path.join(out_base, tag)
        ensure_dir(tdir)
        dat = get_arrays_for_time(arrays, t_seconds)
        p_true_grid = interpolate_to_grid(dat["x"], dat["y"], dat["p"], X, Y, mask)
        s_true_grid = interpolate_to_grid(dat["x"], dat["y"], dat["S"], X, Y, mask)
        p_tilde_pred, s_pred_grid = predict_on_grid(model, cfg, X, Y, t_seconds, device, args.infer_batch_size)
        P_ref = float(cfg.get("P_ref"))
        p0 = float(cfg.get("p0"))
        p_pred_grid = p0 + P_ref * p_tilde_pred
        p_err_grid = p_pred_grid - p_true_grid
        s_err_grid = s_pred_grid - s_true_grid
        # Use matching scales for the reference/prediction pair.
        args = argparse.Namespace(**vars(run_args))
        if args.saturation_err_abs is None:
            args.saturation_err_abs = max(float(np.nanmax(np.abs(s_err_grid))), 1e-12)
        if args.pressure_vmin is None:
            args.pressure_vmin = float(min(np.nanmin(p_true_grid), np.nanmin(p_pred_grid))) * p_scale
        if args.pressure_vmax is None:
            args.pressure_vmax = float(max(np.nanmax(p_true_grid), np.nanmax(p_pred_grid))) * p_scale
        # Basic fields
        save_field(s_true_grid, os.path.join(tdir, "S_true"), args, L, f"{label}, reference $S$, {tag}", args.saturation_cmap, args.saturation_vmin, args.saturation_vmax, r"$S_{\mathrm{CO_2}}$ (-)")
        save_field(s_pred_grid, os.path.join(tdir, "S_pred"), args, L, f"{label}, PINN $S$, {tag}", args.saturation_cmap, args.saturation_vmin, args.saturation_vmax, r"$S_{\mathrm{CO_2}}$ (-)")
        save_field(s_err_grid, os.path.join(tdir, "S_error"), args, L, f"{label}, $\\Delta S$, {tag}", args.error_cmap, -args.saturation_err_abs, args.saturation_err_abs, r"$\Delta S$ (-)")
        p_vmin = None if args.pressure_vmin is None else float(args.pressure_vmin)
        p_vmax = None if args.pressure_vmax is None else float(args.pressure_vmax)
        p_abs = np.nanmax(np.abs(p_err_grid * p_scale)) if args.pressure_err_abs is None else float(args.pressure_err_abs)
        p_abs = max(p_abs, 1e-12)
        save_field(p_true_grid * p_scale, os.path.join(tdir, "p_true"), args, L, f"{label}, reference $p$, {tag}", args.pressure_cmap, p_vmin, p_vmax, p_label)
        save_field(p_pred_grid * p_scale, os.path.join(tdir, "p_pred"), args, L, f"{label}, PINN $p$, {tag}", args.pressure_cmap, p_vmin, p_vmax, p_label)
        save_field(p_err_grid * p_scale, os.path.join(tdir, "p_error"), args, L, f"{label}, $\\Delta p$, {tag}", args.error_cmap, -p_abs, p_abs, p_err_label)
        # Panels
        save_paper_panel(p_true_grid, p_pred_grid, p_err_grid, s_true_grid, s_pred_grid, s_err_grid, args, cfg, os.path.join(tdir, "paper_snapshot_panel"), p_scale, p_label, p_err_label)
        # Front overlays
        lead_tag = level_tag("dS", args.leading_excess_level)
        main_tag = level_tag("S", args.main_raw_contour_level)
        save_front_overlay(X, Y, s_true_grid, s_pred_grid, cfg, args, os.path.join(tdir, f"front_overlay_{lead_tag}"), "excess", args.leading_excess_level, rf"{label}, front $\Delta S={args.leading_excess_level:g}$, {tag}")
        save_front_overlay(X, Y, s_true_grid, s_pred_grid, cfg, args, os.path.join(tdir, f"front_overlay_{main_tag}"), "raw", args.main_raw_contour_level, rf"{label}, front $S={args.main_raw_contour_level:g}$, {tag}")
        save_front_overlay_multi(X, Y, s_true_grid, s_pred_grid, cfg, args, os.path.join(tdir, "front_overlay_multi"), f"{label}, front overlays, {tag}")
        # Masks
        plume_tag = level_tag("dS" if args.threshold_mode == "excess" else "S", args.plume_threshold)
        cat, tp, fp, fn = mask_categories(s_true_grid, s_pred_grid, cfg, args)
        save_plume_overlay(X, Y, s_true_grid, s_pred_grid, cfg, args, os.path.join(tdir, f"plume_mask_overlay_{plume_tag}"), f"{label}, plume {plume_tag}, {tag}")
        binary_mask_plot(fp, cfg, args, os.path.join(tdir, f"false_positive_mask_{plume_tag}"), f"{label}, FP {plume_tag}, {tag}", color=(0.84, 0.33, 0.00, 1.0))
        binary_mask_plot(fn, cfg, args, os.path.join(tdir, f"false_negative_mask_{plume_tag}"), f"{label}, FN {plume_tag}, {tag}", color=(0.00, 0.45, 0.70, 1.0))
        save_fpfn_combined(s_true_grid, s_pred_grid, cfg, args, os.path.join(tdir, f"FP_FN_combined_{plume_tag}"), f"{label}, TP/FP/FN {plume_tag}, {tag}")
        # Speckle
        m002, m005, miso = speckle_masks(s_true_grid, s_pred_grid, cfg, args)
        binary_mask_plot(m002, cfg, args, os.path.join(tdir, "speckle_mask_002"), f"{label}, speckle dS002, {tag}", color=(0.85, 0.20, 0.15, 1.0))
        binary_mask_plot(m005, cfg, args, os.path.join(tdir, "speckle_mask_005"), f"{label}, speckle dS005, {tag}", color=(0.70, 0.05, 0.05, 1.0))
        binary_mask_plot(miso, cfg, args, os.path.join(tdir, "isolated_speckle_components"), f"{label}, isolated speckles, {tag}", color=(0.49, 0.18, 0.56, 1.0))
        save_diagnostic_panel(X, Y, s_true_grid, s_pred_grid, s_err_grid, cfg, args, os.path.join(tdir, "diagnostic_panel"), f"{label}, diagnostics, {tag}")
        np.savez_compressed(os.path.join(tdir, "snapshot_grid.npz"), X=X, Y=Y, p_true=p_true_grid, p_pred=p_pred_grid, p_error=p_err_grid, S_true=s_true_grid, S_pred=s_pred_grid, S_error=s_err_grid, time_seconds=t_seconds, time_display=t_disp, time_unit=args.time_unit)
        record["times"].append({"tag": tag, "time_seconds": t_seconds, "time_display": t_disp})
    with open(os.path.join(out_base, "snapshot_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, default=str)
    from k_spatial_diagnostics import export_diagnostics
    export_diagnostics(out_base, arrays, time_unique, model, cfg, device, args)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()


# =============================================================================
# CLI / main
# =============================================================================
def parse_args():
    ap = argparse.ArgumentParser(description="Nature-style snapshot visualization for M0-M7 and L0-L6 PINN runs.")
    ap.add_argument("--data", dest="data_path", default=DEFAULT_DATA_PATH)
    ap.add_argument("--ckpt-dir", default=DEFAULT_CKPT_DIR)
    ap.add_argument("--ckpt", default=None, help="Evaluate a single checkpoint instead of a checkpoint directory.")
    ap.add_argument("--run-label", default=None, help="Explicit K role label in single-checkpoint mode.")
    ap.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    ap.add_argument("--experiments", default="M0,M1,M2,M3,M4,M5,M6,M7")
    ap.add_argument("--training-strategy", default="BASE")
    ap.add_argument("--include-loo", type=int, default=1)
    ap.add_argument("--loo-exp", default="M7")
    ap.add_argument("--leave-outs", default="NONE,PLAIN_TWONET,FRONT_PLUME,PAIRGRAD,RAR,FV,COARSE_DETAIL")
    ap.add_argument("--seed", default="ANY")
    ap.add_argument("--device", default=DEFAULT_DEVICE, choices=["cpu", "cuda"])
    ap.add_argument("--gpu-id", type=int, default=DEFAULT_GPU_ID)
    ap.add_argument("--time-indices", default=DEFAULT_TIME_INDICES)
    ap.add_argument("--time-unit", default=DEFAULT_TIME_UNIT)
    ap.add_argument("--grid-nx", type=int, default=DEFAULT_GRID_NX)
    ap.add_argument("--grid-ny", type=int, default=DEFAULT_GRID_NY)
    ap.add_argument("--infer-batch-size", type=int, default=DEFAULT_INFER_BATCH_SIZE)
    ap.add_argument("--figure-dpi", type=int, default=DEFAULT_FIGURE_DPI)
    ap.add_argument("--export-pdf", type=int, default=int(DEFAULT_EXPORT_PDF))
    ap.add_argument("--export-tiff", type=int, default=int(DEFAULT_EXPORT_TIFF))
    ap.add_argument("--export-svg", type=int, default=int(DEFAULT_EXPORT_SVG))
    ap.add_argument("--title-mode", default=DEFAULT_TITLE_MODE, choices=["compact", "none"])
    ap.add_argument("--pressure-display-unit", default=DEFAULT_PRESSURE_DISPLAY_UNIT, choices=["Pa", "kPa", "MPa"])
    ap.add_argument("--pressure-cmap", default=DEFAULT_PRESSURE_CMAP)
    ap.add_argument("--saturation-cmap", default=DEFAULT_SATURATION_CMAP)
    ap.add_argument("--error-cmap", default=DEFAULT_ERROR_CMAP)
    ap.add_argument("--saturation-vmin", type=float, default=DEFAULT_SATURATION_VMIN)
    ap.add_argument("--saturation-vmax", type=float, default=DEFAULT_SATURATION_VMAX)
    ap.add_argument("--saturation-err-abs", type=float, default=DEFAULT_SATURATION_ERR_ABS)
    ap.add_argument("--pressure-err-abs", type=float, default=DEFAULT_PRESSURE_ERR_ABS)
    ap.add_argument("--pressure-vmin", type=float, default=None)
    ap.add_argument("--pressure-vmax", type=float, default=None)
    ap.add_argument("--s-ic-co2", type=float, default=DEFAULT_S_IC_CO2)
    ap.add_argument("--Snr", type=float, default=DEFAULT_SNR, help="CO2 residual saturation; used only for metadata/physical interpretation, not for remapping raw Sco2.")
    ap.add_argument("--threshold-mode", default=DEFAULT_THRESHOLD_MODE, choices=["excess", "raw"])
    ap.add_argument("--plume-threshold", type=float, default=DEFAULT_PLUME_THRESHOLD, help="In excess mode, threshold is applied to dS=S-S_IC.")
    ap.add_argument("--leading-excess-level", type=float, default=DEFAULT_LEADING_EXCESS_LEVEL)
    ap.add_argument("--main-raw-contour-level", type=float, default=DEFAULT_MAIN_RAW_CONTOUR_LEVEL)
    ap.add_argument("--speckle-pred-threshold-002", type=float, default=DEFAULT_SPECKLE_PRED_THRESHOLD_002)
    ap.add_argument("--speckle-true-background-002", type=float, default=DEFAULT_SPECKLE_TRUE_BACKGROUND_002)
    ap.add_argument("--speckle-pred-threshold-005", type=float, default=DEFAULT_SPECKLE_PRED_THRESHOLD_005)
    ap.add_argument("--speckle-true-background-005", type=float, default=DEFAULT_SPECKLE_TRUE_BACKGROUND_005)
    ap.add_argument("--isolated-speckle-threshold", type=float, default=DEFAULT_ISOLATED_SPECKLE_THRESHOLD)
    ap.add_argument("--isolated-speckle-min-pixels", type=int, default=DEFAULT_ISOLATED_SPECKLE_MIN_PIXELS)
    return ap.parse_args()


def main():
    args = parse_args()
    apply_publication_rcparams()
    ensure_dir(args.out_root)
    arrays, time_unique = load_dataset_pack(args.data_path)
    runs: list[tuple[RunSpec, str]] = []
    if args.ckpt:
        # Single checkpoint mode; infer exp from filename via global parser if possible.
        from evaluate_global_accuracy_M0_M7_L6_dS_shortlabels import parse_run_from_filename
        exp, st, lo, sd = parse_run_from_filename(args.ckpt)
        exp = exp or args.loo_exp
        st = st or args.training_strategy
        lo = lo or "NONE"
        sd = sd or "NA"
        has_loo = "LOO-" in os.path.basename(args.ckpt).upper()
        label = LOO_LABELS.get(lo, exp) if has_loo else exp
        runs.append((RunSpec(f"{exp}_{st}_LOO-{lo}_seed{sd}", exp, st, lo, sd, args.ckpt), args.run_label or label))
    else:
        from evaluate_global_accuracy_M0_M7_L6_dS_shortlabels import parse_run_from_filename
        exps = [x.strip().upper() for x in args.experiments.split(",") if x.strip()]
        leave_outs = [x.strip().upper() for x in args.leave_outs.split(",") if x.strip()]

        def _score(path):
            base = os.path.basename(path).lower()
            return (0 if "final" in base else 1, 0 if "seed0" in base else 1, len(base), base)

        def _find(exp, strategy, leave_out=None, require_loo=False, exclude_loo=False):
            cands = []
            for fn in os.listdir(args.ckpt_dir):
                if not fn.endswith(".pt"):
                    continue
                pth = os.path.join(args.ckpt_dir, fn)
                e, st, lo, sd = parse_run_from_filename(pth)
                if e != exp:
                    continue
                if strategy and strategy != "ANY" and st != strategy:
                    continue
                upper = fn.upper()
                has_loo = "LOO-" in upper
                is_non_none_loo = has_loo and str(lo).upper() != "NONE"
                if require_loo and not has_loo:
                    continue
                if exclude_loo and is_non_none_loo:
                    continue
                if leave_out is not None and lo != leave_out:
                    continue
                if args.seed and args.seed != "ANY" and sd != str(args.seed):
                    continue
                cands.append(pth)
            if not cands:
                return None
            return sorted(cands, key=_score)[0]

        # Main M0-M7 runs: exclude all leave-one-out checkpoints.
        for exp in exps:
            pth = _find(exp, args.training_strategy.upper(), leave_out="NONE", exclude_loo=True)
            if pth is None:
                print(f"[WARN] no main checkpoint found for {exp} strategy={args.training_strategy}")
                continue
            e, st, lo, sd = parse_run_from_filename(pth)
            runs.append((RunSpec(f"{e}_{st}_LOO-{lo}_seed{sd}", e or exp, st, lo, sd, pth), exp))

        # Leave-one-out runs: L0 is the final M7 model.  Prefer an explicit
        # LOO-NONE checkpoint when it exists, but fall back to the ordinary M7
        # final checkpoint if no LOO-NONE file was saved.  For L1-L6, require
        # explicit LOO-* checkpoint names.
        if bool(args.include_loo):
            for lo_req in leave_outs:
                if lo_req == "NONE":
                    pth = _find(args.loo_exp.upper(), args.training_strategy.upper(), leave_out="NONE", require_loo=True)
                    if pth is None:
                        pth = _find(args.loo_exp.upper(), args.training_strategy.upper(), leave_out="NONE", exclude_loo=True)
                else:
                    pth = _find(args.loo_exp.upper(), args.training_strategy.upper(), leave_out=lo_req, require_loo=True)
                if pth is None:
                    print(f"[WARN] no LOO checkpoint found for {args.loo_exp} LOO-{lo_req}")
                    continue
                e, st, lo, sd = parse_run_from_filename(pth)
                label = LOO_LABELS.get(lo_req, f"LOO-{lo_req}")
                runs.append((RunSpec(f"{e}_{st}_LOO-{lo_req}_seed{sd}", e or args.loo_exp.upper(), st, lo_req, sd, pth), label))
    selected = {label: r.__dict__ for r, label in runs}
    with open(os.path.join(args.out_root, "selected_checkpoints.json"), "w", encoding="utf-8") as f:
        json.dump(selected, f, indent=2)
    with open(os.path.join(args.out_root, "snapshot_visualization_config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)
    for r, label in runs:
        print(f"[snapshot] exporting {label}: {r.ckpt_path}")
        export_run(r, label, arrays, time_unique, args)


if __name__ == "__main__":
    main()
