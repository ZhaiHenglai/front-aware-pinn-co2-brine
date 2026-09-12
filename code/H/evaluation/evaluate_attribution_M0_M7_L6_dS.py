#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Attribution evaluation for M0-M7 forward ablation and M7 BASE leave-one-out runs.

Reads the CSV outputs produced by:
  global/summary_global_accuracy.csv
  front/summary_front_resolution.csv
  front/pairgrad_metrics.csv
  front/speckle_metrics.csv
  plume/summary_plume_support.csv
  physics/summary_physics_consistency.csv
  stability/summary_training_stability.csv

Writes:
  attribution/module_attribution_M0_M7.csv
  attribution/leave_one_out_attribution_L0_L6.csv
  attribution/attribution_metric_matrix.csv
  attribution/figures/*.png|*.pdf|*.tiff

Notes on Sco2 initial/residual saturation
-----------------------------------------
This script is attribution-only; it does not recompute saturation, PDE residuals,
or mass balances. S_IC_CO2=0.0 and Snr=0.0 are therefore stored only as metadata by default.
The actual handling of these quantities is done in the global/front/plume/physics
scripts that generate the input CSV files. Attribution compares those metrics.
"""
from __future__ import annotations

import os
import json
import math
import argparse
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd
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


DEFAULT_EVAL_ROOT = "./eval_M0_M7"
DEFAULT_OUT_DIR = "./eval_M0_M7/attribution"
DEFAULT_EXPERIMENTS = ",".join([f"M{i}" for i in range(8)])
DEFAULT_LEAVE_OUTS = "NONE,PLAIN_TWONET,FRONT_PLUME,PAIRGRAD,RAR,FV,COARSE_DETAIL"
DEFAULT_INCLUDE_L6 = True
DEFAULT_FIGURE_DPI = 600
DEFAULT_EXPORT_PDF = True
DEFAULT_EXPORT_TIFF = True
DEFAULT_EXPORT_SVG = True
EXPORT_SVG = DEFAULT_EXPORT_SVG
DEFAULT_S_IC_CO2 = 0.0
DEFAULT_SNR = 0.0


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

LOO_MODULES = {
    "NONE": "Final",
    "PLAIN_TWONET": "plain TwoNet architecture",
    "FRONT_PLUME": "front/plume loss",
    "PAIRGRAD": "pairwise front-gradient",
    "RAR": "residual RAR",
    "FV": "FV mass loss",
    "COARSE_DETAIL": "coarse/detail branch",
    "WATER_FV": "weak-water FV",
    "MSFF": "MS Fourier",
}


@dataclass(frozen=True)
class MetricSpec:
    key: str
    label: str
    direction: str  # lower or higher
    family: str
    aliases: tuple[str, ...]

    @property
    def lower_is_better(self) -> bool:
        return self.direction.lower().startswith("lower")


METRICS: list[MetricSpec] = [
    MetricSpec("S_RMSE", "S RMSE", "lower", "global", ("S_rmse", "S_RMSE")),
    MetricSpec("p_RMSE_MPa", "p RMSE (MPa)", "lower", "global", ("p_phys_RMSE_MPa", "p_RMSE_MPa")),
    MetricSpec("front_band_RMSE", "Front-band RMSE", "lower", "front", (
        "front_band_wide_005_030_rmse_mean", "front_band_wide_005_030_rmse",
        "front_band_wide_020_060_rmse_mean", "front_band_wide_020_060_rmse", "S_RMSE_front_band",
    )),
    MetricSpec("contour_chamfer_front", "Contour Chamfer front", "lower", "front", (
        "contour_chamfer_S0175_mean", "contour_chamfer_S0175",
        "contour_chamfer_rawS0175_mean", "contour_chamfer_front_mean",
        "contour_chamfer_S040_mean", "contour_chamfer_S040", "contour_chamfer_rawS040_mean",
    )),
    MetricSpec("contour_chamfer_S040", "Contour Chamfer S=0.40", "lower", "front", (
        "contour_chamfer_S040_mean", "contour_chamfer_S040", "contour_chamfer_rawS040_mean",
    )),
    MetricSpec("plume_IoU_dS005", "Plume IoU dS>0.05", "higher", "plume", (
        "plume_IoU_dS005_mean", "plume_IoU_dS005", "grid_plume_IoU_dS005_mean", "grid_plume_IoU_dS005",
        "plume_IoU_S005_mean", "plume_IoU_S005",
    )),
    MetricSpec("FP_area_ratio_dS005", "FP area ratio dS>0.05", "lower", "plume", (
        "FP_area_ratio_dS005_mean", "FP_area_ratio_dS005", "grid_FP_area_ratio_dS005_mean", "grid_FP_area_ratio_dS005",
        "FP_area_ratio_S005_mean", "FP_area_ratio_S005",
    )),
    MetricSpec("FN_area_ratio_dS005", "FN area ratio dS>0.05", "lower", "plume", (
        "FN_area_ratio_dS005_mean", "FN_area_ratio_dS005", "grid_FN_area_ratio_dS005_mean", "grid_FN_area_ratio_dS005",
        "FN_area_ratio_S005_mean", "FN_area_ratio_S005",
    )),
    MetricSpec("speckle_area_ratio_dS002", "Speckle area dS", "lower", "front", (
        "speckle_area_ratio_dS002_mean", "speckle_area_ratio_dS002",
        "speckle_area_ratio_002_mean", "speckle_area_ratio_002",
    )),
    MetricSpec("pairgrad_RMSE", "Pair-gradient RMSE", "lower", "front", (
        "pairgrad_RMSE_mean", "pairgrad_RMSE", "pairgrad_rmse_mean", "pairgrad_rmse",
    )),
    MetricSpec("PDE_total_p95", "PDE total p95", "lower", "physics", (
        "PDE_total_p95", "PDE_total_p95_mean",
    )),
    MetricSpec("PDE_total_RMSE", "PDE total RMSE", "lower", "physics", (
        "PDE_total_RMSE", "PDE_total_RMSE_mean",
    )),
    MetricSpec("local_FV_CO2_RMSE", "Local FV CO₂ RMSE", "lower", "physics", (
        "local_FV_CO2_RMSE", "local_FV_CO2_RMSE_mean",
    )),
    MetricSpec("local_FV_total_RMSE", "Local FV total RMSE", "lower", "physics", (
        "local_FV_total_RMSE", "local_FV_total_RMSE_mean",
    )),
    MetricSpec("CO2_mass_rel_error", "CO₂ mass rel. error", "lower", "physics", (
        "CO2_mass_rel_error_mean", "CO2_mass_rel_error", "CO2_mass_rel_error_mean_mean",
    )),
    MetricSpec("IC_S_RMSE", "IC S RMSE", "lower", "physics", (
        "IC_S_RMSE", "IC_S_RMSE_vs_S_IC_CO2",
    )),
    MetricSpec("wall_time_hours", "Wall time (h)", "lower", "stability", (
        "wall_time_hours", "wall_time_hours_mean",
    )),
    MetricSpec("max_gpu_alloc_GB", "Max GPU alloc. (GB)", "lower", "stability", (
        "max_gpu_alloc_gb", "max_gpu_max_alloc_gb", "max_gpu_alloc_GB",
    )),
    MetricSpec("tail_oscillation", "Tail loss oscillation", "lower", "stability", (
        "oscillation_std_log10_tail", "oscillation_mad_diff_log10_tail",
    )),
]

FORWARD_STEPS = [
    ("M0", "M1", "TwoNet", "representation", "S_RMSE"),
    ("M1", "M2", "MSFF + p/S decoupling", "representation", "front_band_RMSE"),
    ("M2", "M3", "coarse/detail", "representation", "contour_chamfer_front"),
    ("M3", "M4", "front/plume loss", "front/plume supervision", "plume_IoU_dS005"),
    ("M4", "M5", "pairwise front-gradient", "front refinement", "pairgrad_RMSE"),
    ("M5", "M6", "residual RAR", "adaptive sampling", "PDE_total_p95"),
    ("M6", "M7", "FV mass loss", "mass conservation", "local_FV_CO2_RMSE"),
]

LOO_STEPS = [
    ("NONE", "Final", "reference", None),
    ("PLAIN_TWONET", "plain TwoNet architecture", "representation", "front_band_RMSE"),
    ("FRONT_PLUME", "front/plume loss", "front/plume supervision", "plume_IoU_dS005"),
    ("PAIRGRAD", "pairwise front-gradient", "front refinement", "pairgrad_RMSE"),
    ("RAR", "residual RAR", "adaptive sampling", "PDE_total_p95"),
    ("FV", "FV mass loss", "mass conservation", "local_FV_CO2_RMSE"),
    ("COARSE_DETAIL", "coarse/detail branch", "representation", "contour_chamfer_front"),
]


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def parse_list(s: str) -> list[str]:
    return [x.strip().upper() for x in str(s).split(",") if x.strip()]


def set_journal_style():
    setup_nature_rcparams(font_size=7.0, line_width=1.05)


def savefig(fig, base: str, dpi: int, export_pdf: bool, export_tiff: bool):
    ensure_dir(os.path.dirname(base))
    save_pub_figure(fig, base, dpi=dpi, export_pdf=export_pdf, export_tiff=export_tiff, export_svg=EXPORT_SVG, export_png=True, pad_inches=0.035)


def read_csv_optional(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        print(f"[WARN] missing CSV: {path}")
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        print(f"[WARN] failed reading {path}: {exc}")
        return pd.DataFrame()
    if df.empty:
        return pd.DataFrame()
    if "run_id" not in df.columns:
        print(f"[WARN] no run_id column in {path}; skipped")
        return pd.DataFrame()
    return df


def short_label_from_row(row: pd.Series | dict[str, Any], loo_labels: bool = True) -> str:
    get = row.get if isinstance(row, dict) else row.get
    exp = str(get("exp_name", "")).upper()
    leave = str(get("leave_out", "NONE")).upper()
    if leave != "NONE":
        return LOO_LABELS.get(leave, leave)
    if loo_labels and exp == "M7":
        return "L0"
    return exp


def ensure_ids(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    for col in ["exp_name", "training_strategy", "leave_out", "seed"]:
        if col not in df.columns:
            df[col] = ""
    # Recover IDs from run_id where possible.
    rid = df["run_id"].astype(str)
    if (df["exp_name"].astype(str).str.len() == 0).any():
        df.loc[df["exp_name"].astype(str).str.len() == 0, "exp_name"] = rid.str.extract(r"(M\d+)", expand=False)
    if (df["leave_out"].astype(str).str.len() == 0).any():
        ex = rid.str.extract(r"LOO-([A-Za-z0-9_]+)_seed", expand=False).fillna("NONE")
        df.loc[df["leave_out"].astype(str).str.len() == 0, "leave_out"] = ex
    if (df["training_strategy"].astype(str).str.len() == 0).any():
        ex = rid.str.extract(r"M\d+_([A-Z_]+?)_LOO", expand=False).fillna("")
        df.loc[df["training_strategy"].astype(str).str.len() == 0, "training_strategy"] = ex
    if (df["seed"].astype(str).str.len() == 0).any():
        ex = rid.str.extract(r"_seed(\d+)", expand=False).fillna("")
        df.loc[df["seed"].astype(str).str.len() == 0, "seed"] = ex
    df["exp_name"] = df["exp_name"].astype(str).str.upper()
    df["leave_out"] = df["leave_out"].fillna("NONE").astype(str).str.upper()
    df["training_strategy"] = df["training_strategy"].fillna("").astype(str).str.upper()
    df["label"] = [short_label_from_row(r, loo_labels=True) for _, r in df.iterrows()]
    return df


def aggregate_by_run(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = ensure_ids(df)
    id_cols = [c for c in ["run_id", "exp_name", "training_strategy", "leave_out", "seed", "label"] if c in df.columns]
    num_cols = []
    for c in df.columns:
        if c in id_cols or c in ("ckpt_path", "log_path", "out_path", "err_path"):
            continue
        vals = pd.to_numeric(df[c], errors="coerce")
        if vals.notna().any():
            df[c] = vals
            num_cols.append(c)
    rows = []
    for run_id, g in df.groupby("run_id", sort=False):
        first = g.iloc[0]
        row = {c: first.get(c, "") for c in id_cols}
        for c in num_cols:
            row[c] = float(pd.to_numeric(g[c], errors="coerce").mean(skipna=True))
        rows.append(row)
    return pd.DataFrame(rows)


def merge_two(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    """Merge two per-run summary tables without iterative column insertion.

    Semantics preserved from the original implementation:
    - run_id is the join key;
    - values already present in left take precedence;
    - NaN or empty-string cells in left are filled from right;
    - runs present only in right are appended;
    - duplicate run_id rows are reduced by keeping the first row, consistent with
      the previous `r.iloc[0]` behavior.
    """
    if left.empty:
        return right.copy()
    if right.empty:
        return left.copy()
    if "run_id" not in left.columns or "run_id" not in right.columns:
        return left.copy()

    left1 = left.drop_duplicates("run_id", keep="first").copy()
    right1 = right.drop_duplicates("run_id", keep="first").copy()

    # Preserve first-seen column order and allocate all columns in one step.
    all_cols = list(dict.fromkeys([*left1.columns, *right1.columns]))

    left_idx = left1.reindex(columns=all_cols).set_index("run_id", drop=False)
    right_idx = right1.reindex(columns=all_cols).set_index("run_id", drop=False)

    # Treat empty strings as missing only for the purpose of filling from right.
    left_missing = left_idx.mask(left_idx.eq(""))
    out = left_missing.combine_first(right_idx)

    # Keep deterministic row order: all left rows first, then right-only rows.
    row_order = list(left_idx.index) + [idx for idx in right_idx.index if idx not in left_idx.index]
    out = out.loc[row_order, all_cols].reset_index(drop=True)

    # Materialize a compact block layout; this removes pandas fragmentation.
    return out.copy()


def load_all_tables(root: str) -> tuple[pd.DataFrame, dict[str, str]]:
    paths = {
        "global": os.path.join(root, "global", "summary_global_accuracy.csv"),
        "front": os.path.join(root, "front", "summary_front_resolution.csv"),
        "pairgrad": os.path.join(root, "front", "pairgrad_metrics.csv"),
        "speckle": os.path.join(root, "front", "speckle_metrics.csv"),
        "plume": os.path.join(root, "plume", "summary_plume_support.csv"),
        "physics": os.path.join(root, "physics", "summary_physics_consistency.csv"),
        "stability": os.path.join(root, "stability", "summary_training_stability.csv"),
    }
    tables = []
    for name, path in paths.items():
        df = aggregate_by_run(read_csv_optional(path))
        if not df.empty:
            print(f"[load] {name}: {len(df)} runs from {path}")
            tables.append(df)
    merged = pd.DataFrame()
    for df in tables:
        merged = merge_two(merged, df)
    if not merged.empty:
        merged = ensure_ids(merged)
    return merged, paths


def resolve_metric_col(df: pd.DataFrame, spec: MetricSpec) -> str | None:
    if df.empty:
        return None
    for a in spec.aliases:
        if a in df.columns:
            return a
    # Case-insensitive fallback.
    lower_map = {c.lower(): c for c in df.columns}
    for a in spec.aliases:
        if a.lower() in lower_map:
            return lower_map[a.lower()]
    return None


def get_value(df: pd.DataFrame, selector: dict[str, str], metric: MetricSpec) -> float:
    if df.empty:
        return np.nan
    q = pd.Series(True, index=df.index)
    for k, v in selector.items():
        if k not in df.columns:
            return np.nan
        q &= df[k].astype(str).str.upper() == str(v).upper()
    sub = df[q]
    if sub.empty:
        return np.nan
    col = resolve_metric_col(df, metric)
    if col is None:
        return np.nan
    vals = pd.to_numeric(sub[col], errors="coerce")
    return float(vals.mean(skipna=True)) if vals.notna().any() else np.nan


def relative_change(new: float, old: float) -> float:
    if not (np.isfinite(new) and np.isfinite(old)):
        return np.nan
    den = abs(old) if abs(old) > 1e-300 else np.nan
    return float((new - old) / den) if np.isfinite(den) else np.nan


def improvement(child: float, parent: float, spec: MetricSpec) -> tuple[float, float]:
    if not (np.isfinite(child) and np.isfinite(parent)):
        return np.nan, np.nan
    if spec.lower_is_better:
        imp = parent - child
    else:
        imp = child - parent
    den = abs(parent) if abs(parent) > 1e-300 else np.nan
    return float(imp), float(imp / den) if np.isfinite(den) else np.nan


def deterioration(loo: float, final: float, spec: MetricSpec) -> tuple[float, float]:
    if not (np.isfinite(loo) and np.isfinite(final)):
        return np.nan, np.nan
    if spec.lower_is_better:
        det = loo - final
    else:
        det = final - loo
    den = abs(final) if abs(final) > 1e-300 else np.nan
    return float(det), float(det / den) if np.isfinite(den) else np.nan


def metric_by_key(key: str) -> MetricSpec:
    for m in METRICS:
        if m.key == key:
            return m
    raise KeyError(key)


def build_metric_matrix(df: pd.DataFrame, experiments: list[str], leave_outs: list[str]) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        exp = str(r.get("exp_name", "")).upper()
        lo = str(r.get("leave_out", "NONE")).upper()
        if exp not in experiments and lo not in leave_outs:
            continue
        row = {
            "run_id": r.get("run_id", ""),
            "label": short_label_from_row(r, loo_labels=True),
            "exp_name": exp,
            "training_strategy": r.get("training_strategy", ""),
            "leave_out": lo,
            "seed": r.get("seed", ""),
        }
        for m in METRICS:
            col = resolve_metric_col(df, m)
            row[m.key] = pd.to_numeric(pd.Series([r.get(col, np.nan)]), errors="coerce").iloc[0] if col else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def build_module_attribution(df: pd.DataFrame, experiments: list[str]) -> pd.DataFrame:
    rows = []
    for parent, child, module, family, primary in FORWARD_STEPS:
        if parent not in experiments or child not in experiments:
            continue
        row = {
            "parent": parent,
            "child": child,
            "step": f"{parent}->{child}",
            "module": module,
            "family": family,
            "primary_metric": primary,
        }
        for m in METRICS:
            pv = get_value(df, {"exp_name": parent, "leave_out": "NONE"}, m)
            cv = get_value(df, {"exp_name": child, "leave_out": "NONE"}, m)
            imp, imp_rel = improvement(cv, pv, m)
            row[f"{m.key}_parent"] = pv
            row[f"{m.key}_child"] = cv
            row[f"{m.key}_improvement_abs"] = imp
            row[f"{m.key}_improvement_rel"] = imp_rel
        pm = metric_by_key(primary)
        row["primary_parent"] = row.get(f"{primary}_parent", np.nan)
        row["primary_child"] = row.get(f"{primary}_child", np.nan)
        row["primary_improvement_abs"] = row.get(f"{primary}_improvement_abs", np.nan)
        row["primary_improvement_rel"] = row.get(f"{primary}_improvement_rel", np.nan)
        row["primary_direction"] = pm.direction
        row["conclusion"] = "improves primary metric" if np.isfinite(row["primary_improvement_abs"]) and row["primary_improvement_abs"] > 0 else "no positive primary attribution"
        rows.append(row)
    return pd.DataFrame(rows)


def build_loo_attribution(df: pd.DataFrame, leave_outs: list[str]) -> pd.DataFrame:
    rows = []
    final_sel = {"exp_name": "M7", "leave_out": "NONE"}
    for lo, module, family, primary in LOO_STEPS:
        if lo not in leave_outs:
            continue
        row = {
            "final_run": "M7 LOO-NONE",
            "loo_label": LOO_LABELS.get(lo, lo),
            "leave_out": lo,
            "removed_module": module,
            "family": family,
            "primary_metric": primary or "reference",
        }
        for m in METRICS:
            fv = get_value(df, final_sel, m)
            lv = get_value(df, {"exp_name": "M7", "leave_out": lo}, m)
            det, det_rel = deterioration(lv, fv, m)
            row[f"{m.key}_final"] = fv
            row[f"{m.key}_loo"] = lv
            row[f"{m.key}_deterioration_abs"] = det
            row[f"{m.key}_deterioration_rel"] = det_rel
        if primary:
            row["primary_final"] = row.get(f"{primary}_final", np.nan)
            row["primary_loo"] = row.get(f"{primary}_loo", np.nan)
            row["primary_deterioration_abs"] = row.get(f"{primary}_deterioration_abs", np.nan)
            row["primary_deterioration_rel"] = row.get(f"{primary}_deterioration_rel", np.nan)
            row["module_necessary"] = int(np.isfinite(row["primary_deterioration_abs"]) and row["primary_deterioration_abs"] > 0)
            row["conclusion"] = "removal worsens primary metric" if row["module_necessary"] else "no positive LOO evidence"
        else:
            row["primary_final"] = np.nan
            row["primary_loo"] = np.nan
            row["primary_deterioration_abs"] = np.nan
            row["primary_deterioration_rel"] = np.nan
            row["module_necessary"] = np.nan
            row["conclusion"] = "reference final model"
        rows.append(row)
    return pd.DataFrame(rows)


def write_csv(path: str, df: pd.DataFrame):
    ensure_dir(os.path.dirname(path))
    if df is None or df.empty:
        pd.DataFrame().to_csv(path, index=False)
    else:
        df.to_csv(path, index=False)


def write_json(path: str, obj: dict[str, Any]):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


def plot_primary_bars(mod_df: pd.DataFrame, loo_df: pd.DataFrame, fig_dir: str, dpi: int, export_pdf: bool, export_tiff: bool):
    set_journal_style()
    if not mod_df.empty:
        df = mod_df.copy()
        fig, ax = plt.subplots(figsize=(7.2, 2.65), constrained_layout=False)
        x = np.arange(len(df))
        y = pd.to_numeric(df["primary_improvement_rel"], errors="coerce") * 100.0
        ax.axhline(0.0, color="0.2", linewidth=0.6)
        ax.bar(x, y, width=0.72, color=[color_for_label(lab, "#6F8FAF") for lab in df["child"].astype(str)])
        ax.set_xticks(x)
        ax.set_xticklabels(df["child"].astype(str), rotation=0)
        ax.set_ylabel("Primary improvement (%)")
        ax.set_xlabel("Forward ablation step")
        ax.set_title("M0-M7 module attribution", pad=3)
        apply_nature_axis(ax, grid=True, grid_axis="y")
        # Module names below bars, compact.
        for xi, mod in zip(x, df["module"].astype(str)):
            ax.text(xi, ax.get_ylim()[0], mod, rotation=35, ha="right", va="top", fontsize=5.6)
        fig.subplots_adjust(left=0.08, right=0.99, top=0.88, bottom=0.42)
        savefig(fig, os.path.join(fig_dir, "module_primary_improvement"), dpi, export_pdf, export_tiff)
        plt.close(fig)
    if not loo_df.empty:
        df = loo_df[loo_df["leave_out"] != "NONE"].copy()
        fig, ax = plt.subplots(figsize=(5.6, 2.65), constrained_layout=False)
        x = np.arange(len(df))
        y = pd.to_numeric(df["primary_deterioration_rel"], errors="coerce") * 100.0
        ax.axhline(0.0, color="0.2", linewidth=0.6)
        ax.bar(x, y, width=0.72, color=[color_for_label(lab, "#C76D66") for lab in df["loo_label"].astype(str)])
        ax.set_xticks(x)
        ax.set_xticklabels(df["loo_label"].astype(str), rotation=0)
        ax.set_ylabel("Primary deterioration (%)")
        ax.set_xlabel("Leave-one-out run")
        ax.set_title("L0-L6 leave-one-out attribution", pad=3)
        apply_nature_axis(ax, grid=True, grid_axis="y")
        for xi, mod in zip(x, df["removed_module"].astype(str)):
            ax.text(xi, ax.get_ylim()[0], mod, rotation=35, ha="right", va="top", fontsize=5.6)
        fig.subplots_adjust(left=0.10, right=0.99, top=0.88, bottom=0.42)
        savefig(fig, os.path.join(fig_dir, "loo_primary_deterioration"), dpi, export_pdf, export_tiff)
        plt.close(fig)


def heat_values_from_module(mod_df: pd.DataFrame, keys: list[str]) -> tuple[np.ndarray, list[str], list[str]]:
    rows = []
    labels = []
    for _, r in mod_df.iterrows():
        labels.append(str(r.get("child", "")))
        rows.append([pd.to_numeric(pd.Series([r.get(f"{k}_improvement_rel", np.nan)]), errors="coerce").iloc[0] * 100.0 for k in keys])
    return np.asarray(rows, dtype=float), labels, keys


def heat_values_from_loo(loo_df: pd.DataFrame, keys: list[str]) -> tuple[np.ndarray, list[str], list[str]]:
    df = loo_df[loo_df["leave_out"] != "NONE"].copy()
    rows = []
    labels = []
    for _, r in df.iterrows():
        labels.append(str(r.get("loo_label", "")))
        rows.append([pd.to_numeric(pd.Series([r.get(f"{k}_deterioration_rel", np.nan)]), errors="coerce").iloc[0] * 100.0 for k in keys])
    return np.asarray(rows, dtype=float), labels, keys


def pretty_metric_label(key: str) -> str:
    for m in METRICS:
        if m.key == key:
            return m.label
    return key


def plot_heatmap(mat: np.ndarray, row_labels: list[str], metric_keys: list[str], title: str, base: str, dpi: int, export_pdf: bool, export_tiff: bool):
    if mat.size == 0:
        return
    set_journal_style()
    finite = mat[np.isfinite(mat)]
    vmax = np.nanpercentile(np.abs(finite), 90) if finite.size else 1.0
    vmax = max(vmax, 1e-12)
    fig_w = max(5.8, 0.55 * len(metric_keys) + 1.8)
    fig_h = max(2.6, 0.35 * len(row_labels) + 1.2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=False)
    im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_xticks(np.arange(len(metric_keys)))
    ax.set_xticklabels([pretty_metric_label(k) for k in metric_keys], rotation=35, ha="right")
    ax.set_title(title, pad=4)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    # Annotate cells when not too dense.
    if mat.shape[0] * mat.shape[1] <= 80:
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=5.4, color="black")
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.025)
    cbar.set_label("Relative attribution (%)")
    fig.subplots_adjust(left=0.16, right=0.94, top=0.88, bottom=0.31)
    savefig(fig, base, dpi, export_pdf, export_tiff)
    plt.close(fig)


def plot_metric_matrix(metric_df: pd.DataFrame, fig_dir: str, dpi: int, export_pdf: bool, export_tiff: bool, include_l6: bool):
    if metric_df.empty:
        return
    # Select compact main-table metrics.
    keys = ["S_RMSE", "front_band_RMSE", "contour_chamfer_front", "plume_IoU_dS005", "speckle_area_ratio_dS002", "pairgrad_RMSE", "PDE_total_p95", "local_FV_CO2_RMSE", "CO2_mass_rel_error"]
    keys = [k for k in keys if k in metric_df.columns]
    if not keys:
        return
    df = metric_df.copy()
    # M0-M7 raw values.
    mdf = df[(df["leave_out"].astype(str).str.upper() == "NONE") & (df["exp_name"].astype(str).str.match(r"M\d+"))].copy()
    mdf["mnum"] = mdf["exp_name"].str.extract(r"M(\d+)").astype(float)
    mdf = mdf.sort_values("mnum")
    # Normalize each metric to M0/M7-comparable minmax for display only.
    if not mdf.empty:
        mat = mdf[keys].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        # Convert higher-is-better metrics so smaller normalized value means better? For raw matrix, use z-normalized sign-adjusted score.
        score = np.full_like(mat, np.nan, dtype=float)
        for j, k in enumerate(keys):
            m = metric_by_key(k)
            vals = mat[:, j]
            finite = vals[np.isfinite(vals)]
            if finite.size:
                lo, hi = float(np.nanmin(finite)), float(np.nanmax(finite))
                den = hi - lo if hi > lo else 1.0
                norm = (vals - lo) / den
                if not m.lower_is_better:
                    norm = 1.0 - norm
                score[:, j] = norm
        plot_heatmap(score, mdf["exp_name"].tolist(), keys, "M0-M7 normalized metric map (lower is better)", os.path.join(fig_dir, "M_metric_matrix_normalized"), dpi, export_pdf, export_tiff)
    # L0-L6 raw normalized values.
    ldf = df[df["exp_name"].astype(str).str.upper() == "M7"].copy()
    if not include_l6:
        ldf = ldf[~(ldf["leave_out"].astype(str).str.upper() == "COARSE_DETAIL")]
    order = {"NONE":0,"PLAIN_TWONET":1,"FRONT_PLUME":2,"PAIRGRAD":3,"RAR":4,"FV":5,"COARSE_DETAIL":6}
    ldf["ord"] = ldf["leave_out"].astype(str).str.upper().map(order).fillna(99)
    ldf = ldf.sort_values("ord")
    if not ldf.empty:
        mat = ldf[keys].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        score = np.full_like(mat, np.nan, dtype=float)
        for j, k in enumerate(keys):
            m = metric_by_key(k)
            vals = mat[:, j]
            finite = vals[np.isfinite(vals)]
            if finite.size:
                lo, hi = float(np.nanmin(finite)), float(np.nanmax(finite))
                den = hi - lo if hi > lo else 1.0
                norm = (vals - lo) / den
                if not m.lower_is_better:
                    norm = 1.0 - norm
                score[:, j] = norm
        labels = [LOO_LABELS.get(str(x).upper(), str(x)) for x in ldf["leave_out"]]
        plot_heatmap(score, labels, keys, "L0-L6 normalized metric map (lower is better)", os.path.join(fig_dir, "L_metric_matrix_normalized"), dpi, export_pdf, export_tiff)


def main():
    ap = argparse.ArgumentParser(description="M0-M7 module attribution and L0-L6 leave-one-out attribution aggregator.")
    ap.add_argument("--eval-root", type=str, default=DEFAULT_EVAL_ROOT, help="Root directory containing global/front/plume/physics/stability outputs.")
    ap.add_argument("--out-dir", type=str, default=DEFAULT_OUT_DIR)
    ap.add_argument("--experiments", type=str, default=DEFAULT_EXPERIMENTS)
    ap.add_argument("--leave-outs", type=str, default=DEFAULT_LEAVE_OUTS)
    ap.add_argument("--include-l6", type=int, default=int(DEFAULT_INCLUDE_L6))
    ap.add_argument("--figure-dpi", type=int, default=DEFAULT_FIGURE_DPI)
    ap.add_argument("--export-pdf", type=int, default=int(DEFAULT_EXPORT_PDF))
    ap.add_argument("--export-tiff", type=int, default=int(DEFAULT_EXPORT_TIFF))
    ap.add_argument("--export-svg", type=int, default=int(DEFAULT_EXPORT_SVG),
                    help="Export editable SVG files in addition to PNG/PDF/TIFF.")
    ap.add_argument("--s-ic-co2", type=float, default=DEFAULT_S_IC_CO2)
    ap.add_argument("--Snr", type=float, default=DEFAULT_SNR)
    args = ap.parse_args()
    global EXPORT_SVG
    EXPORT_SVG = bool(args.export_svg)

    out_dir = args.out_dir
    fig_dir = os.path.join(out_dir, "figures")
    ensure_dir(out_dir); ensure_dir(fig_dir)

    experiments = parse_list(args.experiments)
    leave_outs = parse_list(args.leave_outs)
    if bool(args.include_l6) and "COARSE_DETAIL" not in leave_outs:
        leave_outs.append("COARSE_DETAIL")
    if "NONE" not in leave_outs:
        leave_outs.insert(0, "NONE")

    merged, paths = load_all_tables(args.eval_root)
    if merged.empty:
        raise RuntimeError(f"No input summary CSVs found under {args.eval_root}. Run global/front/plume/physics/stability first.")

    metric_df = build_metric_matrix(merged, experiments, leave_outs)
    mod_df = build_module_attribution(merged, experiments)
    loo_df = build_loo_attribution(merged, leave_outs)

    # Sort output tables.
    if not mod_df.empty:
        order = {f"M{i}": i for i in range(8)}
        mod_df["_ord"] = mod_df["child"].map(order).fillna(99)
        mod_df = mod_df.sort_values("_ord").drop(columns=["_ord"])
    if not loo_df.empty:
        order = {"NONE":0,"PLAIN_TWONET":1,"FRONT_PLUME":2,"PAIRGRAD":3,"RAR":4,"FV":5,"COARSE_DETAIL":6}
        loo_df["_ord"] = loo_df["leave_out"].map(order).fillna(99)
        loo_df = loo_df.sort_values("_ord").drop(columns=["_ord"])

    write_csv(os.path.join(out_dir, "attribution_metric_matrix.csv"), metric_df)
    write_csv(os.path.join(out_dir, "module_attribution_M0_M7.csv"), mod_df)
    write_csv(os.path.join(out_dir, "leave_one_out_attribution_L0_L6.csv"), loo_df)

    write_json(os.path.join(out_dir, "attribution_evaluation_config.json"), {
        "eval_root": args.eval_root,
        "input_paths": paths,
        "experiments": experiments,
        "leave_outs": leave_outs,
        "S_IC_CO2": args.s_ic_co2,
        "Snr": args.Snr,
        "note": "S_IC_CO2 and Snr are metadata here. Attribution uses metrics already computed by global/front/plume/physics/stability evaluators.",
        "metrics": [{"key": m.key, "label": m.label, "direction": m.direction, "family": m.family, "aliases": list(m.aliases)} for m in METRICS],
        "forward_steps": FORWARD_STEPS,
        "loo_steps": LOO_STEPS,
    })

    plot_primary_bars(mod_df, loo_df, fig_dir, args.figure_dpi, bool(args.export_pdf), bool(args.export_tiff))
    keys = ["S_RMSE", "front_band_RMSE", "contour_chamfer_front", "plume_IoU_dS005", "speckle_area_ratio_dS002", "pairgrad_RMSE", "PDE_total_p95", "local_FV_CO2_RMSE", "CO2_mass_rel_error"]
    keys = [k for k in keys if k in [m.key for m in METRICS]]
    if not mod_df.empty:
        mat, rows, cols = heat_values_from_module(mod_df, keys)
        plot_heatmap(mat, rows, cols, "M0-M7 relative module attribution", os.path.join(fig_dir, "module_attribution_heatmap"), args.figure_dpi, bool(args.export_pdf), bool(args.export_tiff))
    if not loo_df.empty:
        mat, rows, cols = heat_values_from_loo(loo_df, keys)
        plot_heatmap(mat, rows, cols, "L0-L6 leave-one-out deterioration", os.path.join(fig_dir, "leave_one_out_deterioration_heatmap"), args.figure_dpi, bool(args.export_pdf), bool(args.export_tiff))
    plot_metric_matrix(metric_df, fig_dir, args.figure_dpi, bool(args.export_pdf), bool(args.export_tiff), bool(args.include_l6))

    print(f"[done] attribution outputs written to {out_dir}")


if __name__ == "__main__":
    main()
