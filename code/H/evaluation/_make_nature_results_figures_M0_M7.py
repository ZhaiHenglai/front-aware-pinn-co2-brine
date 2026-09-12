#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Nature-style result figures for the M0-M7 front-aware PINN manuscript.

This script is intentionally a plotting-only layer. It reads the evaluated CSV
tables and snapshot NPZ files produced by the existing Results/ evaluators and
exports clean, manuscript-oriented figures without recomputing predictions or
changing any source metrics.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Circle

from _nature_plot_style_M0_M7 import (
    METHOD_PALETTE,
    apply_nature_axis,
    panel_label as shared_panel_label,
    save_pub_figure,
    setup_nature_rcparams,
)
from _time_series_scaling import (
    apply_robust_time_axis,
    axis_scaling_report,
)


METHOD_LABELS = [f"M{i}" for i in range(8)]
OUTLIER_ZOOM_THRESHOLD = 5.0

METHOD_DESCRIPTIONS = {
    "M0": "single-branch vanilla PINN",
    "M1": "two-branch pressure-saturation surrogate",
    "M2": "MS Fourier features + pressure/saturation input decoupling",
    "M3": "coarse/detail saturation representation",
    "M4": "front/plume-supervised loss",
    "M5": "pairwise front-gradient refinement",
    "M6": "residual adaptive resampling",
    "M7": "finite-volume mass regularisation",
}

LOO_MODULE_LABELS = {
    "L1": "plain TwoNet",
    "L2": "front/plume\nloss",
    "L3": "pair-gradient",
    "L4": "RAR",
    "L5": "FV mass",
    "L6": "coarse/detail\nbranch",
}

PALETTE = dict(METHOD_PALETTE)


def palette_color(label: str, fallback: str | None = None) -> str | None:
    return PALETTE.get(str(label), fallback)


def setup_nature_style() -> None:
    setup_nature_rcparams(font_size=7.0, line_width=0.75)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return pd.read_csv(path)


def first_existing_col(df: pd.DataFrame, candidates: Iterable[str], *, required: bool = True) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    if required:
        raise KeyError(f"None of these columns were found: {list(candidates)}")
    return None


def existing_cols(df: pd.DataFrame, candidates: Iterable[str]) -> list[str]:
    return [col for col in candidates if col in df.columns]


def save_figure(fig: plt.Figure, out_base: Path, dpi: int = 600) -> list[Path]:
    ensure_dir(out_base.parent)
    return save_pub_figure(
        fig,
        out_base,
        dpi=dpi,
        export_pdf=True,
        export_tiff=True,
        export_svg=True,
        export_png=True,
        pad_inches=0.03,
    )


def panel_label(ax: plt.Axes, label: str) -> None:
    shared_panel_label(ax, label, x=-0.13, y=1.08, fontsize=8.0)


def nice_axis(ax: plt.Axes, ylabel: str, xlabel: str = "time (days)") -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    apply_nature_axis(ax, grid=True, grid_axis="both")


def _finite_positive(values: Iterable[float]) -> np.ndarray:
    arr = pd.to_numeric(pd.Series(list(values)), errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    return arr[arr > 0]


def _outlier_scaling_report(series: dict[str, tuple[np.ndarray, np.ndarray]]) -> dict[str, object]:
    rows = []
    all_values = []
    for lab, (_, y) in series.items():
        yy = np.asarray(y, dtype=float)
        yy = yy[np.isfinite(yy)]
        if yy.size:
            rows.append((lab, float(np.nanmax(yy))))
            all_values.extend(yy.tolist())
    vals = _finite_positive(all_values)
    if not rows or vals.size == 0:
        return {"needs_zoom": False, "outlier_labels": [], "max_to_median": np.nan}
    max_to_median = float(np.nanmax(vals) / max(np.nanmedian(vals), 1.0e-12))
    label_max = pd.Series({lab: val for lab, val in rows}).sort_values(ascending=False)
    med_label = float(np.nanmedian(label_max.to_numpy(dtype=float)))
    outliers = [
        lab for lab, val in label_max.items()
        if val / max(med_label, 1.0e-12) >= 2.5
    ]
    if max_to_median >= OUTLIER_ZOOM_THRESHOLD and not outliers:
        outliers = [str(label_max.index[0])]
    needs_zoom = bool(max_to_median >= OUTLIER_ZOOM_THRESHOLD)
    return {"needs_zoom": needs_zoom, "outlier_labels": outliers, "max_to_median": max_to_median}


def maybe_add_cluster_zoom(
    ax: plt.Axes,
    series: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    xlabel: str = "time (days)",
    categorical_xticklabels: list[str] | None = None,
) -> None:
    """Add an inset when one/few extreme methods flatten the main cluster."""
    report = _outlier_scaling_report(series)
    if not report["needs_zoom"]:
        return

    outliers = set(report["outlier_labels"])
    cluster_values: list[float] = []
    for lab, (_, y) in series.items():
        yy = np.asarray(y, dtype=float)
        yy = yy[np.isfinite(yy)]
        if yy.size == 0:
            continue
        if lab not in outliers:
            cluster_values.extend(yy.tolist())
    if len(cluster_values) < 4:
        all_values = np.concatenate([np.asarray(y, dtype=float) for _, y in series.values()])
        all_values = all_values[np.isfinite(all_values)]
        if all_values.size == 0:
            return
        cluster_values = all_values[all_values <= np.nanpercentile(all_values, 85.0)].tolist()
    cluster = np.asarray(cluster_values, dtype=float)
    cluster = cluster[np.isfinite(cluster)]
    if cluster.size == 0:
        return

    ymin = 0.0 if np.nanmin(cluster) >= 0 else float(np.nanpercentile(cluster, 2.0))
    pos_cluster = cluster[cluster > 0]
    if pos_cluster.size and float(np.nanmax(pos_cluster) / max(np.nanmedian(pos_cluster), 1.0e-12)) >= OUTLIER_ZOOM_THRESHOLD:
        ymax = float(np.nanpercentile(cluster, 90.0))
    else:
        ymax = float(np.nanpercentile(cluster, 98.0))
    if ymax <= ymin:
        return
    pad = 0.14 * (ymax - ymin)
    ymax += pad
    all_max = max(float(np.nanmax(np.asarray(y, dtype=float))) for _, y in series.values() if len(y))
    if ymax >= 0.90 * all_max:
        return

    iax = ax.inset_axes([0.54, 0.50, 0.42, 0.42])
    for lab, (x, y) in series.items():
        iax.plot(
            x, y,
            color=PALETTE.get(lab, "#666666"),
            lw=0.65 if lab not in {"M7", "M2"} else 0.9,
            alpha=0.82,
        )
    iax.set_ylim(ymin, ymax)
    finite_x = np.concatenate([np.asarray(x, dtype=float) for x, _ in series.values() if len(x)])
    if finite_x.size:
        iax.set_xlim(float(np.nanmin(finite_x)), float(np.nanmax(finite_x)))
    iax.grid(axis="y", color="#D0D5DD", lw=0.25, alpha=0.75)
    iax.grid(axis="x", color="#E5E7EB", lw=0.18, alpha=0.45)
    iax.tick_params(labelsize=4.8, length=1.6, width=0.35, pad=1)
    if categorical_xticklabels is not None:
        iax.set_xticks(np.arange(len(categorical_xticklabels)))
        iax.set_xticklabels(categorical_xticklabels, rotation=0)
    else:
        iax.set_xlabel(xlabel, fontsize=4.8, labelpad=0.5)
    iax.set_title("cluster zoom", fontsize=5.2, pad=1.5)
    for spine in iax.spines.values():
        spine.set_linewidth(0.35)
        spine.set_color("#667085")


def plot_method_lines(
    ax: plt.Axes,
    df: pd.DataFrame,
    y_col: str,
    ylabel: str,
    *,
    ylim: tuple[float, float] | None = None,
    x_col: str = "time_days",
    xlabel: str = "time (days)",
    labels: Iterable[str] = METHOD_LABELS,
    cluster_zoom: bool = True,
) -> None:
    labels = list(labels)
    plotted: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for lab in labels:
        sub = df[df["plot_label"].fillna(df.get("label", "")).eq(lab)]
        if sub.empty and "label" in df.columns:
            sub = df[df["label"].eq(lab)]
        if sub.empty or y_col not in sub.columns:
            continue
        sub = sub[[x_col, y_col]].dropna().sort_values(x_col)
        if sub.empty:
            continue
        x = sub[x_col].to_numpy(dtype=float)
        y = sub[y_col].to_numpy(dtype=float)
        plotted[lab] = (x, y)
        lw = 1.35 if lab in {"M2", "M7"} else 0.8
        alpha = 0.96 if lab in {"M2", "M5", "M6", "M7"} else 0.72
        z = 4 if lab in {"M2", "M7"} else 2
        ax.plot(
            x, y,
            color=PALETTE.get(lab, "#666666"),
            lw=lw, alpha=alpha, label=lab, zorder=z
        )
    nice_axis(ax, ylabel, xlabel=xlabel)
    if ylim is not None:
        ymax = max((float(np.nanmax(y)) for _, y in plotted.values() if len(y)), default=ylim[1])
        ax.set_ylim(ylim[0], max(ylim[1], ymax * 1.04))
    if cluster_zoom and plotted:
        report = apply_robust_time_axis(
            ax,
            plotted,
            x_cut=0.01 if x_col == "time_days" else None,
            color_for_label=palette_color,
        )
        if not report.get("applied"):
            maybe_add_cluster_zoom(ax, plotted, xlabel=xlabel)


def write_source_data(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, index=False)


def figure_global_progression(eval_root: Path, out_dir: Path, dpi: int) -> list[Path]:
    data = read_csv(eval_root / "M0_M7" / "global" / "by_time_global_accuracy.csv")
    data = data[data["plot_label"].isin(METHOD_LABELS)].copy()
    write_source_data(
        data[[
            "plot_label", "time_days", "S_rmse", "p_phys_RMSE_MPa",
            "S_RMSE_front_band", "S_RMSE_excess_plume_dS005"
        ]],
        out_dir / "source_data" / "FigR1_M0_M7_global_progression.csv",
    )

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.35), constrained_layout=True)
    specs = [
        ("S_rmse", "Saturation RMSE", (0, 0.38), "a"),
        ("p_phys_RMSE_MPa", "Pressure RMSE (MPa)", (0, 1.48), "b"),
        ("S_RMSE_front_band", "Front-band S RMSE", (0, 0.38), "c"),
        ("S_RMSE_excess_plume_dS005", "Excess-plume S RMSE", (0, 0.36), "d"),
    ]
    for ax, (col, ylabel, ylim, lab) in zip(axes.ravel(), specs):
        plot_method_lines(ax, data, col, ylabel, ylim=ylim)
        panel_label(ax, lab)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels, ncol=9, loc="upper center", bbox_to_anchor=(0.5, 1.03),
        frameon=False, handlelength=1.4, handletextpad=0.35, columnspacing=0.7
    )
    return save_figure(fig, out_dir / "main_text" / "FigR1_M0_M7_global_progression", dpi=dpi)


def figure_front_plume_diagnostics(eval_root: Path, out_dir: Path, dpi: int) -> list[Path]:
    front = read_csv(eval_root / "M0_M7" / "front" / "by_time_front_resolution.csv")
    pair = read_csv(eval_root / "M0_M7" / "front" / "pairgrad_metrics.csv")
    plume = read_csv(eval_root / "M0_M7" / "plume" / "by_time_plume_support.csv")
    speckle = read_csv(eval_root / "M0_M7" / "front" / "speckle_metrics.csv")
    front = front.copy()
    pair = pair.copy()
    plume = plume.copy()
    speckle = speckle.copy()
    for df in (front, pair, plume, speckle):
        if "plot_label" not in df.columns and "label" in df.columns:
            df.loc[:, "plot_label"] = df["label"]

    front_band_col = first_existing_col(front, ["front_band_wide_005_030_rmse", "front_band_wide_020_060_rmse"])
    contour_col = first_existing_col(speckle, ["contour_chamfer_S0175", "contour_chamfer_front", "contour_chamfer_S040"], required=False)
    source = (
        front[["plot_label", "time_days", front_band_col]]
        .merge(pair[["plot_label", "time_days", "pairgrad_RMSE"]], on=["plot_label", "time_days"], how="outer")
        .merge(plume[["plot_label", "time_days", "plume_IoU_dS005", "FN_area_ratio_dS005"]], on=["plot_label", "time_days"], how="outer")
    )
    write_source_data(source, out_dir / "source_data" / "FigR2_front_plume_diagnostics_timeseries.csv")
    speckle_source_cols = ["plot_label", "snapshot_label", "time_days", "speckle_area_ratio_dS002"]
    if contour_col is not None:
        speckle_source_cols.append(contour_col)
    write_source_data(
        speckle[speckle_source_cols],
        out_dir / "source_data" / "FigR2_front_plume_diagnostics_selected_times.csv",
    )

    fig = plt.figure(figsize=(7.2, 4.65), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)
    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[0, 2]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1:]),
    ]

    plot_method_lines(
        axes[0], front, front_band_col,
        "Front-band S RMSE", ylim=(0, 0.38)
    )
    panel_label(axes[0], "a")

    plot_method_lines(axes[1], pair, "pairgrad_RMSE", "Pair-gradient RMSE", ylim=(0, 10.2))
    panel_label(axes[1], "b")

    plot_method_lines(axes[2], plume, "plume_IoU_dS005", "Plume IoU ($\\Delta S>0.05$)", ylim=(0, 1.02), cluster_zoom=False)
    panel_label(axes[2], "c")

    plot_method_lines(axes[3], plume, "FN_area_ratio_dS005", "FN area / true plume", ylim=(0, 0.36))
    panel_label(axes[3], "d")

    ax = axes[4]
    sel = speckle[speckle["plot_label"].isin(METHOD_LABELS)].copy()
    order = ["early", "middle", "late", "final"]
    x = np.arange(len(order))
    width = 0.085
    for i, lab in enumerate(METHOD_LABELS):
        sub = sel[sel["plot_label"].eq(lab)].copy()
        y = []
        for tag in order:
            vals = sub.loc[sub["snapshot_label"].eq(tag), "speckle_area_ratio_dS002"].dropna()
            y.append(float(vals.iloc[0]) if len(vals) else np.nan)
        ax.plot(
            x + (i - 4) * width * 0.22, y, marker="o", ms=2.6, lw=0.9,
            color=PALETTE[lab], alpha=0.88, label=lab
        )
    speckle_series = {}
    for lab in METHOD_LABELS:
        sub = sel[sel["plot_label"].eq(lab)].copy()
        y = []
        for tag in order:
            vals = sub.loc[sub["snapshot_label"].eq(tag), "speckle_area_ratio_dS002"].dropna()
            y.append(float(vals.iloc[0]) if len(vals) else np.nan)
        speckle_series[lab] = (x.astype(float), np.asarray(y, dtype=float))
    ax.set_xticks(x)
    ax.set_xticklabels(order)
    nice_axis(ax, "Speckle area ratio", xlabel="selected time")
    ax.set_ylim(0, 0.56)
    maybe_add_cluster_zoom(ax, speckle_series, xlabel="selected time", categorical_xticklabels=order)
    panel_label(ax, "e")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, ncol=9, loc="upper center", bbox_to_anchor=(0.5, 1.03),
        frameon=False, handlelength=1.4, handletextpad=0.35, columnspacing=0.7
    )
    return save_figure(fig, out_dir / "main_text" / "FigR2_front_plume_diagnostics", dpi=dpi)


def mask_wells(arr: np.ndarray, x: np.ndarray, y: np.ndarray, radius: float = 0.5) -> np.ma.MaskedArray:
    mask = (x ** 2 + y ** 2 <= radius ** 2) | ((x - 5.0) ** 2 + (y - 5.0) ** 2 <= radius ** 2)
    return np.ma.array(arr, mask=mask)


def add_well_patches(ax: plt.Axes, radius: float = 0.5) -> None:
    for xy in [(0, 0), (5, 5)]:
        ax.add_patch(Circle(xy, radius=radius, facecolor="white", edgecolor="white", lw=0.0, zorder=10))


def field_panel(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *,
    title: str,
    cmap: str,
    vmin: float,
    vmax: float,
    contour: np.ndarray | None = None,
    contour_level: float = 0.175,
) -> mpl.image.AxesImage:
    im = ax.imshow(
        mask_wells(z, x, y), origin="lower", extent=[0, 5, 0, 5],
        cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest"
    )
    if contour is not None:
        ax.contour(x, y, mask_wells(contour, x, y), levels=[contour_level], colors="white", linewidths=0.55)
        ax.contour(x, y, mask_wells(contour, x, y), levels=[contour_level], colors="black", linewidths=0.28)
    add_well_patches(ax)
    ax.set_title(title, pad=3)
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 5)
    ax.set_aspect("equal")
    ax.set_xlabel("$x$ (m)")
    ax.set_ylabel("$y$ (m)")
    ax.tick_params(length=2)
    return im


def figure_m7_snapshot(eval_root: Path, out_dir: Path, dpi: int) -> list[Path]:
    npz_path = eval_root / "M0_M7" / "snapshots" / "M7" / "final" / "snapshot_grid.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Snapshot NPZ not found: {npz_path}")
    z = np.load(npz_path)
    x, y = z["X"], z["Y"]
    p_true = z["p_true"] / 1e6
    p_pred = z["p_pred"] / 1e6
    p_err = z["p_error"] / 1e6
    s_true = z["S_true"]
    s_pred = z["S_pred"]
    s_err = z["S_error"]
    write_source_data(
        pd.DataFrame({
            "metric": ["time_days", "p_error_min_MPa", "p_error_max_MPa", "S_error_min", "S_error_max"],
            "value": [
                float(z["time_display"]),
                float(np.nanmin(p_err)), float(np.nanmax(p_err)),
                float(np.nanmin(s_err)), float(np.nanmax(s_err)),
            ],
        }),
        out_dir / "source_data" / "FigR3_M7_final_snapshot_metadata.csv",
    )

    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.55), constrained_layout=True)
    p_min = float(np.nanmin(p_true))
    p_max = float(np.nanmax(p_true))
    p_err_lim = max(abs(float(np.nanpercentile(p_err, 1))), abs(float(np.nanpercentile(p_err, 99))))
    s_err_lim = max(abs(float(np.nanpercentile(s_err, 1))), abs(float(np.nanpercentile(s_err, 99))))

    ims = []
    ims.append(field_panel(axes[0, 0], x, y, p_true, title="Reference pressure", cmap="viridis", vmin=p_min, vmax=p_max))
    ims.append(field_panel(axes[0, 1], x, y, p_pred, title="M7 pressure", cmap="viridis", vmin=p_min, vmax=p_max))
    ims.append(field_panel(axes[0, 2], x, y, p_err, title="$\\Delta p$", cmap="RdBu_r", vmin=-p_err_lim, vmax=p_err_lim))
    ims.append(field_panel(axes[1, 0], x, y, s_true, title="Reference saturation", cmap="viridis", vmin=0.0, vmax=0.8, contour=s_true))
    ims.append(field_panel(axes[1, 1], x, y, s_pred, title="M7 saturation", cmap="viridis", vmin=0.0, vmax=0.8, contour=s_pred))
    ims.append(field_panel(axes[1, 2], x, y, s_err, title="$\\Delta S$", cmap="RdBu_r", vmin=-s_err_lim, vmax=s_err_lim))

    for ax, lab in zip(axes.ravel(), list("abcdef")):
        panel_label(ax, lab)

    cb1 = fig.colorbar(ims[1], ax=axes[0, :2], shrink=0.78, pad=0.015)
    cb1.set_label("pressure (MPa)")
    cb2 = fig.colorbar(ims[2], ax=axes[0, 2], shrink=0.78, pad=0.015)
    cb2.set_label("$\\Delta p$ (MPa)")
    cb3 = fig.colorbar(ims[4], ax=axes[1, :2], shrink=0.78, pad=0.015)
    cb3.set_label("$S_{CO_2}$ (-)")
    cb4 = fig.colorbar(ims[5], ax=axes[1, 2], shrink=0.78, pad=0.015)
    cb4.set_label("$\\Delta S$ (-)")

    return save_figure(fig, out_dir / "main_text" / "FigR3_M7_final_field_reconstruction", dpi=dpi)


def figure_loo_attribution(eval_root: Path, out_dir: Path, dpi: int) -> list[Path]:
    loo = read_csv(eval_root / "L0_L6" / "attribution" / "leave_one_out_attribution_L0_L6.csv")
    loo = loo[loo["loo_label"].isin(["L1", "L2", "L3", "L4", "L5", "L6"])].copy()
    metrics = [
        (["S_RMSE_deterioration_rel"], "S RMSE"),
        (["front_band_RMSE_deterioration_rel"], "Front-band\nRMSE"),
        (["contour_chamfer_front_deterioration_rel", "contour_chamfer_S0175_deterioration_rel", "contour_chamfer_S040_deterioration_rel"], "Front\nChamfer"),
        (["plume_IoU_dS005_deterioration_rel"], "Plume IoU"),
        (["pairgrad_RMSE_deterioration_rel"], "Pair-gradient\nRMSE"),
        (["PDE_total_p95_deterioration_rel"], "PDE total\np95"),
        (["local_FV_CO2_RMSE_deterioration_rel"], "Local FV\nCO$_2$ RMSE"),
        (["CO2_mass_rel_error_deterioration_rel"], "CO$_2$ mass\nrel. err."),
    ]
    metric_cols = [first_existing_col(loo, candidates, required=False) for candidates, _ in metrics]
    mat = np.array([[float(row.get(col, np.nan)) * 100 if col else np.nan for col in metric_cols] for _, row in loo.iterrows()])
    source_cols = ["loo_label", "removed_module"] + [col for col in metric_cols if col]
    write_source_data(
        loo[source_cols],
        out_dir / "source_data" / "FigR4_LOO_module_attribution.csv",
    )

    fig, ax = plt.subplots(figsize=(7.2, 3.05), constrained_layout=True)
    cap = 80.0
    clipped = np.clip(mat, -cap, cap)
    im = ax.imshow(clipped, cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-cap, vcenter=0.0, vmax=cap), aspect="auto")
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels([m[1] for m in metrics], rotation=35, ha="right", rotation_mode="anchor")
    ax.set_yticks(np.arange(len(loo)))
    ax.set_yticklabels([f"{r.loo_label}: {LOO_MODULE_LABELS.get(r.loo_label, r.removed_module)}" for _, r in loo.iterrows()])
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if not np.isfinite(v):
                label = "n/a"
            elif abs(v) > cap:
                label = f"{'+' if v > 0 else ''}{v:.0f}"
            else:
                label = f"{v:+.0f}"
            color = "white" if abs(clipped[i, j]) > 48 else "#111827"
            ax.text(j, i, label, ha="center", va="center", fontsize=5.6, color=color)
    ax.set_title("Module necessity from M7 leave-one-out tests", pad=6)
    cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.015)
    cb.set_label("relative deterioration (%)")
    return save_figure(fig, out_dir / "main_text" / "FigR4_LOO_module_attribution", dpi=dpi)


def figure_training_strategy(eval_root: Path, out_dir: Path, dpi: int) -> list[Path]:
    rows = []
    for strategy in ["BASE", "BETA", "DIFF", "BETA_DIFF"]:
        base = eval_root / "train" / strategy
        g = read_csv(base / "global" / "summary_global_accuracy.csv").iloc[0]
        fr = read_csv(base / "front" / "summary_front_resolution.csv")
        front_band_mean_col = first_existing_col(fr, ["front_band_wide_005_030_rmse_mean", "front_band_wide_020_060_rmse_mean"])
        fr_front = fr[pd.to_numeric(fr[front_band_mean_col], errors="coerce").notna()].iloc[0]
        fr_pair = fr[fr["pairgrad_RMSE_mean"].notna()].iloc[0]
        pl = read_csv(base / "plume" / "summary_plume_support.csv").iloc[0]
        ph = read_csv(base / "physics" / "summary_physics_consistency.csv").iloc[0]
        rows.append({
            "strategy": strategy.replace("_", "+"),
            "S RMSE": float(g["S_rmse"]),
            "front-band RMSE": float(fr_front[front_band_mean_col]),
            "plume IoU": float(pl["plume_IoU_dS005_mean"]),
            "pair-gradient RMSE": float(fr_pair["pairgrad_RMSE_mean"]),
            "PDE p95": float(ph["PDE_total_p95"]),
            "local FV CO2 RMSE": float(ph["local_FV_CO2_RMSE"]),
        })
    df = pd.DataFrame(rows)
    write_source_data(df, out_dir / "source_data" / "FigS_training_strategy_comparison.csv")

    lower_is_better = ["S RMSE", "front-band RMSE", "pair-gradient RMSE", "PDE p95", "local FV CO2 RMSE"]
    norm = df.copy()
    for col in lower_is_better:
        best = norm[col].min()
        norm[col] = norm[col] / best
    best_iou = norm["plume IoU"].max()
    norm["plume IoU"] = best_iou / norm["plume IoU"]

    metrics = ["S RMSE", "front-band RMSE", "plume IoU", "pair-gradient RMSE", "PDE p95", "local FV CO2 RMSE"]
    fig, ax = plt.subplots(figsize=(6.4, 2.85), constrained_layout=True)
    x = np.arange(len(metrics))
    width = 0.18
    colors = ["#111827", "#4F7FB9", "#A96B50", "#7E7E7E"]
    for i, (_, row) in enumerate(norm.iterrows()):
        ax.bar(x + (i - 1.5) * width, [row[m] for m in metrics], width=width, color=colors[i], label=row["strategy"])
    ax.axhline(1.0, color="#111827", lw=0.6, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=25, ha="right")
    ax.set_ylabel("normalised score\n(best = 1)")
    ax.set_ylim(0, max(1.35, float(norm[metrics].max().max()) * 1.12))
    ax.grid(axis="y", color="#D0D5DD", lw=0.35, alpha=0.75)
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    ax.set_title("Training-continuation variants do not dominate the base M7 model", pad=6)
    return save_figure(fig, out_dir / "supplementary" / "FigS_training_strategy_comparison", dpi=dpi)


def audit_scaling(eval_root: Path, out_dir: Path) -> Path:
    """Screen manuscript candidate panels for outlier-driven axis compression."""
    specs = [
        ("FigR1", eval_root / "M0_M7" / "global" / "by_time_global_accuracy.csv", "plot_label", "S_rmse"),
        ("FigR1", eval_root / "M0_M7" / "global" / "by_time_global_accuracy.csv", "plot_label", "p_phys_RMSE_MPa"),
        ("FigR1", eval_root / "M0_M7" / "global" / "by_time_global_accuracy.csv", "plot_label", "S_RMSE_front_band"),
        ("FigR1", eval_root / "M0_M7" / "global" / "by_time_global_accuracy.csv", "plot_label", "S_RMSE_excess_plume_dS005"),
        ("FigR2", eval_root / "M0_M7" / "front" / "by_time_front_resolution.csv", "label", "front_band_wide_005_030_rmse"),
        ("FigR2", eval_root / "M0_M7" / "front" / "by_time_front_resolution.csv", "label", "front_band_wide_020_060_rmse"),
        ("FigR2", eval_root / "M0_M7" / "front" / "pairgrad_metrics.csv", "label", "pairgrad_RMSE"),
        ("FigR2", eval_root / "M0_M7" / "plume" / "by_time_plume_support.csv", "label", "plume_IoU_dS005"),
        ("FigR2", eval_root / "M0_M7" / "plume" / "by_time_plume_support.csv", "label", "FN_area_ratio_dS005"),
        ("FigR2", eval_root / "M0_M7" / "front" / "speckle_metrics.csv", "label", "speckle_area_ratio_dS002"),
        ("FigR4", eval_root / "L0_L6" / "attribution" / "leave_one_out_attribution_L0_L6.csv", "loo_label", "primary_deterioration_rel"),
    ]
    rows = []
    for fig, path, label_col, metric in specs:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if label_col not in df.columns and "plot_label" in df.columns:
            label_col = "plot_label"
        if label_col not in df.columns or metric not in df.columns:
            continue
        series = {}
        x_col = "time_days" if "time_days" in df.columns else None
        for lab, sub in df.groupby(label_col, sort=False):
            y = pd.to_numeric(sub[metric], errors="coerce").to_numpy(dtype=float)
            if x_col is not None:
                x = pd.to_numeric(sub[x_col], errors="coerce").to_numpy(dtype=float)
            else:
                x = np.arange(len(y), dtype=float)
            series[str(lab)] = (x, y)
        cluster_report = _outlier_scaling_report(series)
        robust_report = axis_scaling_report(series, x_cut=0.01 if x_col == "time_days" else None)
        vals = _finite_positive(pd.to_numeric(df[metric], errors="coerce").dropna().tolist())
        rows.append({
            "figure": fig,
            "metric": metric,
            "source": str(path),
            "n": int(vals.size),
            "max": float(np.nanmax(vals)) if vals.size else np.nan,
            "median": float(np.nanmedian(vals)) if vals.size else np.nan,
            "max_to_median": cluster_report["max_to_median"],
            "max_to_post_p95": robust_report["max_to_post_p95"],
            "max_to_post_median": robust_report["max_to_post_median"],
            "dominant_labels": ";".join(cluster_report["outlier_labels"]),
            "robust_axis_needed": bool(robust_report["needs_rescale"]),
            "robust_axis_reason": robust_report["reason"],
            "cluster_zoom_added": bool(cluster_report["needs_zoom"]),
        })
    audit = pd.DataFrame(rows)
    out_path = out_dir / "source_data" / "figure_scaling_audit.csv"
    write_source_data(audit, out_path)
    return out_path


def write_readme(out_dir: Path, eval_root: Path, generated: list[Path]) -> None:
    ensure_dir(out_dir)
    manifest = pd.DataFrame({
        "file": [str(p) for p in generated],
        "figure": [p.stem for p in generated],
        "format": [p.suffix.lstrip(".") for p in generated],
    })
    manifest.to_csv(out_dir / "nature_figure_manifest.csv", index=False)
    readme = {
        "purpose": "Nature-style plotting-only layer for the M0-M7 manuscript results.",
        "eval_root": str(eval_root),
        "output_dir": str(out_dir),
        "figure_logic": {
            "FigR1": "M0-M7 quantitative progression in pressure, saturation and front-band errors.",
            "FigR2": "Front-specific diagnostics: front-band, pair-gradient, plume IoU and speckle behaviour.",
            "FigR3": "Final M7 pressure-saturation field reconstruction from snapshot_grid.npz.",
            "FigR4": "M7 leave-one-out module necessity from attribution CSV.",
            "FigS_training": "Training-continuation sensitivity for BASE/BETA/DIFF/BETA_DIFF.",
        },
        "method_map": METHOD_DESCRIPTIONS,
        "notes": [
            "This script does not recompute model outputs.",
            "Panels with early transient spikes or full-range outliers use a robust post-transient/cluster main axis and show the compressed range in an inset.",
            "Extreme heatmap values in FigR4 are colour-clipped at +/-80% but exact annotations and source data are exported.",
            "figure_scaling_audit.csv records which panels triggered the robust-axis and cluster-zoom rules.",
        ],
    }
    (out_dir / "README_nature_figures.json").write_text(json.dumps(readme, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    default_project = Path(__file__).resolve().parents[1]
    default_eval = default_project / "Results2" / "eval_M0_M7_split"
    default_out = default_eval / "nature_figures"
    ap = argparse.ArgumentParser(description="Create Nature-style manuscript figures from evaluated M0-M7 CSV/NPZ outputs.")
    ap.add_argument("--eval-root", type=Path, default=default_eval, help="Root containing M0_M7/, L0_L6/ and train/ evaluation folders.")
    ap.add_argument("--out-dir", type=Path, default=default_out, help="Output directory for Nature-style figures.")
    ap.add_argument("--dpi", type=int, default=600)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    setup_nature_style()
    ensure_dir(args.out_dir / "main_text")
    ensure_dir(args.out_dir / "supplementary")
    ensure_dir(args.out_dir / "source_data")
    generated: list[Path] = []
    generated += figure_global_progression(args.eval_root, args.out_dir, args.dpi)
    generated += figure_front_plume_diagnostics(args.eval_root, args.out_dir, args.dpi)
    generated += figure_m7_snapshot(args.eval_root, args.out_dir, args.dpi)
    generated += figure_loo_attribution(args.eval_root, args.out_dir, args.dpi)
    generated += figure_training_strategy(args.eval_root, args.out_dir, args.dpi)
    audit_scaling(args.eval_root, args.out_dir)
    write_readme(args.out_dir, args.eval_root, generated)
    print(f"[nature-figures] eval root: {args.eval_root}")
    print(f"[nature-figures] output:    {args.out_dir}")
    print(f"[nature-figures] exported:  {len(generated)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
