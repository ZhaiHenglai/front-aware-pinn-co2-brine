#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Robust axis scaling helpers for evaluation time-series figures."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np


ColorFunc = Callable[[str, str | None], str | None]


def transient_cut_for_unit(time_unit: str, cut_days: float = 0.01) -> float | None:
    """Return the display-axis value equivalent to ``cut_days``."""
    unit = str(time_unit or "").strip().lower()
    if not unit or "day" in unit or unit in {"d"}:
        return float(cut_days)
    if "hour" in unit or unit in {"h", "hr", "hrs"}:
        return float(cut_days * 24.0)
    if "min" in unit:
        return float(cut_days * 24.0 * 60.0)
    if unit in {"s", "sec", "secs", "second", "seconds"}:
        return float(cut_days * 24.0 * 3600.0)
    return None


def _finite_xy(x: Any, y: Any) -> tuple[np.ndarray, np.ndarray]:
    xx = np.asarray(x, dtype=float)
    yy = np.asarray(y, dtype=float)
    n = min(xx.size, yy.size)
    if n == 0:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    xx = xx[:n]
    yy = yy[:n]
    mask = np.isfinite(xx) & np.isfinite(yy)
    return xx[mask], yy[mask]


def _series_arrays(series: dict[str, tuple[np.ndarray, np.ndarray]]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for lab, (x, y) in series.items():
        xx, yy = _finite_xy(x, y)
        if yy.size:
            out[str(lab)] = (xx, yy)
    return out


def _positive(values: list[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return arr[arr > 0]


def _post_mask(x: np.ndarray, x_cut: float | None) -> np.ndarray:
    if x_cut is None or not np.isfinite(x_cut):
        return np.ones_like(x, dtype=bool)
    return x >= float(x_cut)


def _fallback_post_values(arrays: dict[str, tuple[np.ndarray, np.ndarray]], x_cut: float | None) -> np.ndarray:
    vals: list[float] = []
    for x, y in arrays.values():
        yy = y[_post_mask(x, x_cut)]
        yy = yy[np.isfinite(yy)]
        vals.extend(yy.tolist())
    post = _positive(vals)
    if post.size >= 2:
        return post

    vals = []
    for x, y in arrays.values():
        ux = np.unique(x[np.isfinite(x)])
        if ux.size > 3:
            yy = y[np.isin(x, ux[2:])]
        else:
            yy = y
        yy = yy[np.isfinite(yy)]
        vals.extend(yy.tolist())
    return _positive(vals)


def axis_scaling_report(
    series: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    x_cut: float | None = 0.01,
    early_p95_threshold: float = 1.35,
    early_median_threshold: float = 2.0,
    cluster_median_threshold: float = 4.0,
) -> dict[str, Any]:
    """Diagnose whether a time-series panel is visually compressed by extremes."""
    arrays = _series_arrays(series)
    all_values: list[float] = []
    label_max: dict[str, float] = {}
    max_x = np.nan
    max_label = ""
    max_y = -np.inf
    for lab, (x, y) in arrays.items():
        yy = y[np.isfinite(y)]
        if not yy.size:
            continue
        all_values.extend(yy.tolist())
        local_idx = int(np.nanargmax(yy))
        local_max = float(yy[local_idx])
        label_max[lab] = local_max
        if local_max > max_y:
            max_y = local_max
            finite_idx = np.flatnonzero(np.isfinite(y))[local_idx]
            max_x = float(x[finite_idx])
            max_label = lab

    pos = _positive(all_values)
    if not arrays or pos.size == 0:
        return {
            "needs_rescale": False,
            "reason": "",
            "max_to_post_p95": np.nan,
            "max_to_post_median": np.nan,
            "max_x": np.nan,
            "max_label": "",
            "outlier_labels": [],
        }

    post = _fallback_post_values(arrays, x_cut)
    if post.size == 0:
        post = pos
    post_p95 = float(np.nanpercentile(post, 95.0))
    post_median = float(np.nanmedian(post))
    max_to_post_p95 = float(np.nanmax(pos) / max(post_p95, 1.0e-12))
    max_to_post_median = float(np.nanmax(pos) / max(post_median, 1.0e-12))

    label_vals = np.asarray(list(label_max.values()), dtype=float)
    med_label_max = float(np.nanmedian(label_vals)) if label_vals.size else np.nan
    outlier_labels = [
        lab for lab, val in sorted(label_max.items(), key=lambda item: item[1], reverse=True)
        if np.isfinite(med_label_max) and val / max(med_label_max, 1.0e-12) >= 2.5
    ]

    early_max = bool(x_cut is not None and np.isfinite(x_cut) and np.isfinite(max_x) and max_x < float(x_cut))
    early_issue = early_max and max_to_post_p95 >= early_p95_threshold and max_to_post_median >= early_median_threshold
    cluster_issue = max_to_post_median >= cluster_median_threshold
    reason = "early transient" if early_issue else ("full-range outlier" if cluster_issue else "")
    return {
        "needs_rescale": bool(early_issue or cluster_issue),
        "reason": reason,
        "early_max": early_max,
        "max_to_post_p95": max_to_post_p95,
        "max_to_post_median": max_to_post_median,
        "post_p95": post_p95,
        "post_median": post_median,
        "max_x": max_x,
        "max_label": max_label,
        "outlier_labels": outlier_labels,
    }


def _main_axis_values(
    arrays: dict[str, tuple[np.ndarray, np.ndarray]],
    report: dict[str, Any],
    x_cut: float | None,
) -> np.ndarray:
    outliers = set(map(str, report.get("outlier_labels", [])))
    vals: list[float] = []
    for lab, (x, y) in arrays.items():
        if lab in outliers and report.get("reason") == "full-range outlier":
            continue
        yy = y[_post_mask(x, x_cut)]
        yy = yy[np.isfinite(yy)]
        vals.extend(yy.tolist())
    arr = _positive(vals)
    if arr.size >= 4:
        return arr
    return _fallback_post_values(arrays, x_cut)


def _color(label: str, color_for_label: ColorFunc | None) -> str:
    if color_for_label is None:
        return "#666666"
    return str(color_for_label(label, "#666666") or "#666666")


def _draw_inset(
    ax: plt.Axes,
    arrays: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    title: str,
    x_cut: float | None,
    color_for_label: ColorFunc | None,
    categorical_xticklabels: list[str] | None,
) -> None:
    iax = ax.inset_axes([0.55, 0.53, 0.40, 0.38])
    y_values: list[float] = []
    x_values: list[float] = []
    for lab, (x, y) in arrays.items():
        if title == "early transient" and x_cut is not None and np.isfinite(x_cut):
            mask = x <= float(x_cut)
            xx = x[mask]
            yy = y[mask]
        else:
            xx, yy = x, y
        if yy.size == 0:
            continue
        iax.plot(xx, yy, color=_color(lab, color_for_label), lw=0.55, alpha=0.78)
        y_values.extend(yy[np.isfinite(yy)].tolist())
        x_values.extend(xx[np.isfinite(xx)].tolist())
    yy = np.asarray(y_values, dtype=float)
    xx = np.asarray(x_values, dtype=float)
    if yy.size:
        ymin = 0.0 if np.nanmin(yy) >= 0 else float(np.nanpercentile(yy, 2.0))
        ymax = float(np.nanpercentile(yy, 99.0))
        if ymax <= ymin:
            ymax = float(np.nanmax(yy))
        pad = 0.10 * max(ymax - ymin, 1.0e-12)
        iax.set_ylim(ymin, ymax + pad)
    if xx.size:
        iax.set_xlim(float(np.nanmin(xx)), float(np.nanmax(xx)))
    if categorical_xticklabels is not None:
        iax.set_xticks(np.arange(len(categorical_xticklabels)))
        iax.set_xticklabels(categorical_xticklabels)
    iax.set_title(title, fontsize=5.2, pad=1.5)
    iax.grid(axis="y", color="#D0D5DD", lw=0.25, alpha=0.70)
    iax.grid(axis="x", color="#E5E7EB", lw=0.18, alpha=0.40)
    iax.tick_params(labelsize=4.8, length=1.5, width=0.35, pad=1)
    for spine in iax.spines.values():
        spine.set_linewidth(0.35)
        spine.set_color("#667085")


def apply_robust_time_axis(
    ax: plt.Axes,
    series: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    x_cut: float | None = 0.01,
    color_for_label: ColorFunc | None = None,
    categorical_xticklabels: list[str] | None = None,
    y_floor: float | None = 0.0,
    y_quantile: float = 98.0,
    headroom: float = 0.16,
) -> dict[str, Any]:
    """Set a post-transient/cluster y-scale and add an explanatory inset if needed."""
    arrays = _series_arrays(series)
    report = axis_scaling_report(arrays, x_cut=x_cut)
    if not report.get("needs_rescale"):
        report["applied"] = False
        return report

    main = _main_axis_values(arrays, report, x_cut)
    if main.size < 4:
        report["applied"] = False
        return report

    ymax = float(np.nanpercentile(main, y_quantile))
    if not np.isfinite(ymax) or ymax <= 0:
        report["applied"] = False
        return report
    ymin = float(y_floor) if y_floor is not None else float(np.nanpercentile(main, 2.0))
    if ymin < 0:
        pad_low = 0.08 * max(ymax - ymin, 1.0e-12)
        ymin -= pad_low
    ymax *= 1.0 + float(headroom)
    if ymax <= ymin:
        report["applied"] = False
        return report

    ax.set_ylim(ymin, ymax)
    reason = str(report.get("reason") or "full range")
    inset_title = "early transient" if reason == "early transient" else "full range"
    _draw_inset(
        ax,
        arrays,
        title=inset_title,
        x_cut=x_cut,
        color_for_label=color_for_label,
        categorical_xticklabels=categorical_xticklabels,
    )
    if x_cut is not None and np.isfinite(x_cut) and reason == "early transient":
        ax.axvline(float(x_cut), color="#9CA3AF", lw=0.45, ls=":", zorder=1)
    report["applied"] = True
    report["main_ylim_upper"] = ymax
    return report
