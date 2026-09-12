#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared Nature-style plotting helpers for the PINN evaluation suite.

The module centralizes the figure contract used by the Results/ evaluators:
compact journal typography, editable vector text, restrained method colors,
and consistent PNG/PDF/TIFF/SVG export.  It intentionally contains no metric
logic and can be imported by any plotting-only script.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt


METHOD_LABELS = tuple(f"M{i}" for i in range(8))

METHOD_PALETTE = {
    "M0": "#7A7F86",
    "M1": "#B7A99A",
    "M2": "#B45A4A",
    "M3": "#8B6A9E",
    "M4": "#C7B84A",
    "M5": "#5C8E88",
    "M6": "#4F7FB9",
    "M7": "#111827",
}

LOO_PALETTE = {
    "L0": "#111827",
    "L1": "#7A7F86",
    "L2": "#C7B84A",
    "L3": "#5C8E88",
    "L4": "#4F7FB9",
    "L5": "#2F6F63",
    "L6": "#8B6A9E",
    "LFVW": "#7F6A55",
    "LMSFF": "#B45A4A",
}

TRAINING_PALETTE = {
    "BASE": "#111827",
    "BETA": "#4F7FB9",
    "DIFF": "#5C8E88",
    "BETA_DIFF": "#B45A4A",
    "BETA-DIFF": "#B45A4A",
}

NEUTRAL_PALETTE = {
    "light": "#D0D5DD",
    "mid": "#7A7F86",
    "dark": "#344054",
    "black": "#111827",
    "grid": "#E5E7EB",
}

DEFAULT_COLOR_CYCLE = (
    "#111827", "#4F7FB9", "#5C8E88", "#B45A4A",
    "#8B6A9E", "#C7B84A", "#7A7F86", "#B7A99A",
)


def _canonical_label(label: Any) -> str:
    text = "" if label is None else str(label).strip()
    if not text:
        return ""
    upper = text.upper().replace(" ", "_")

    m = re.search(r"(?<![A-Z0-9])(M[0-7])(?![0-9])", upper)
    if m:
        return m.group(1)

    m = re.search(r"(?<![A-Z0-9])(L[0-6])(?![0-9])", upper)
    if m:
        return m.group(1)

    if "PLAIN_TWONET" in upper:
        return "L1"
    if "FRONT_PLUME" in upper:
        return "L2"
    if "PAIRGRAD" in upper:
        return "L3"
    if re.search(r"(?<![A-Z])RAR(?![A-Z])", upper):
        return "L4"
    if re.search(r"(?<![A-Z])FV(?![A-Z])|FINITE_VOLUME", upper):
        return "L5"
    if "COARSE_DETAIL" in upper:
        return "L6"
    if "WATER_FV" in upper:
        return "LFVW"
    if "MSFF" in upper:
        return "LMSFF"

    for key in sorted(TRAINING_PALETTE, key=len, reverse=True):
        if key in upper:
            return key
    return upper


def color_for_label(label: Any, fallback: str | None = None) -> str:
    """Return a stable, low-saturation color for a method/run label."""
    key = _canonical_label(label)
    if key in METHOD_PALETTE:
        return METHOD_PALETTE[key]
    if key in LOO_PALETTE:
        return LOO_PALETTE[key]
    if key in TRAINING_PALETTE:
        return TRAINING_PALETTE[key]
    if fallback is not None:
        return fallback
    digest = hashlib.md5(str(label).encode("utf-8")).hexdigest()
    idx = int(digest[:8], 16) % len(DEFAULT_COLOR_CYCLE)
    return DEFAULT_COLOR_CYCLE[idx]


def setup_nature_rcparams(font_size: float = 7.0, line_width: float = 0.8) -> None:
    """Apply compact Nature-style matplotlib settings before figure creation."""
    tick_width = max(0.35, min(0.75, float(line_width) * 0.55))
    mpl.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans", "sans-serif"],
        "font.size": float(font_size),
        "axes.titlesize": float(font_size) + 0.5,
        "axes.labelsize": float(font_size),
        "xtick.labelsize": max(5.5, float(font_size) - 0.6),
        "ytick.labelsize": max(5.5, float(font_size) - 0.6),
        "legend.fontsize": max(5.5, float(font_size) - 0.8),
        "axes.linewidth": float(line_width),
        "lines.linewidth": float(line_width),
        "patch.linewidth": max(0.35, float(line_width) * 0.55),
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.major.width": tick_width,
        "ytick.major.width": tick_width,
        "xtick.minor.width": tick_width * 0.8,
        "ytick.minor.width": tick_width * 0.8,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
    })


def apply_nature_axis(ax, grid: bool = True, grid_axis: str = "y"):
    """Apply shared axis styling to an existing axes."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(float(mpl.rcParams.get("axes.linewidth", 0.8)))
        ax.spines[side].set_color(NEUTRAL_PALETTE["dark"])
    ax.tick_params(axis="both", which="major", direction="out", length=2.5, pad=2.0)
    if grid:
        axis = "both" if str(grid_axis).lower() == "both" else str(grid_axis)
        ax.grid(True, axis=axis, linewidth=0.35, color=NEUTRAL_PALETTE["grid"], alpha=0.85)
        ax.set_axisbelow(True)
    else:
        ax.grid(False)
    leg = ax.get_legend()
    if leg is not None:
        leg.set_frame_on(False)
    return ax


def panel_label(
    ax,
    label: str,
    x: float = -0.13,
    y: float = 1.08,
    fontsize: float = 8.0,
    color: str = "black",
    fontweight: str = "bold",
) -> None:
    """Place a compact panel label in axes coordinates."""
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=fontsize,
        fontweight=fontweight,
        color=color,
    )


def save_pub_figure(
    fig,
    base_without_ext,
    dpi: int = 600,
    export_pdf: bool = True,
    export_tiff: bool = True,
    export_svg: bool = True,
    export_png: bool = True,
    pad_inches: float = 0.035,
) -> list[Path]:
    """Save a figure using the shared journal export contract."""
    base = Path(base_without_ext)
    base.parent.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    exports = (
        ("png", bool(export_png), {"dpi": int(dpi)}),
        ("pdf", bool(export_pdf), {}),
        ("tiff", bool(export_tiff), {"dpi": int(dpi)}),
        ("svg", bool(export_svg), {}),
    )
    for ext, enabled, kwargs in exports:
        if not enabled:
            continue
        path = base.with_suffix(f".{ext}")
        fig.savefig(path, bbox_inches="tight", pad_inches=pad_inches, facecolor="white", **kwargs)
        paths.append(path)
    return paths


def setup_nature_style() -> None:
    """Compatibility alias used by older plotting-only scripts."""
    setup_nature_rcparams(font_size=7.0, line_width=0.75)
