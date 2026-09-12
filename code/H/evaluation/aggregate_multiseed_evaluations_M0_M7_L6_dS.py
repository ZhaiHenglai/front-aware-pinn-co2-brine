#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aggregate per-seed M0-M7 evaluation CSV outputs.

Expected input layout:

  Results2/eval_M0_M7_split_seed0/M0_M7/...
  Results2/eval_M0_M7_split_seed1/M0_M7/...
  Results2/eval_M0_M7_split_seed2/M0_M7/...
  Results2/eval_M0_M7_split_seed42/M0_M7/...

The script does not run model inference. It only combines CSV outputs produced
by the existing evaluators and computes per-run mean/std/SEM/95% CI summaries
across seeds.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_TABLES = (
    "global/summary_global_accuracy.csv",
    "global/selected_time_global_accuracy.csv",
    "global/by_time_global_accuracy.csv",
    "front/summary_front_resolution.csv",
    "front/by_time_front_resolution.csv",
    "front/pairgrad_metrics.csv",
    "front/speckle_metrics.csv",
    "plume/summary_plume_support.csv",
    "plume/by_time_plume_support.csv",
    "physics/summary_physics_consistency.csv",
    "physics/pde_metrics.csv",
    "physics/bc_ic_metrics.csv",
    "physics/local_fv_metrics.csv",
    "physics/mass_metrics.csv",
    "stability/summary_training_stability.csv",
)

PATH_LIKE_COLS = {
    "ckpt_path",
    "checkpoint_path",
    "log_path",
    "out_path",
    "err_path",
}
RUN_INSTANCE_COLS = {
    "run_id",
    "raw_run_id",
    "seed",
    "stem",
    "job_id",
    "host",
    "start_text",
    "end_text",
    "source_eval_root",
    "source_csv",
    "source_seed",
}

PREFERRED_ID_COLS = (
    "suite",
    "exp_name",
    "training_strategy",
    "leave_out",
    "label",
    "plot_label",
    "short_label",
    "snapshot_label",
    "time_index",
    "time_s",
    "time_display",
    "time_label",
    "split",
    "kind",
    "mode",
)

PRIMARY_METRICS = {
    "global/summary_global_accuracy.csv": (
        "S_RMSE",
        "S_rel_l2",
        "p_tilde_RMSE",
        "front_band_S_RMSE",
        "plume_S_RMSE",
        "IC_S_RMSE_vs_S_IC_CO2",
    ),
    "front/summary_front_resolution.csv": (
        "front_band_wide_005_030_rmse_mean",
        "contour_chamfer_S0175_mean",
        "speckle_area_ratio_dS002_mean",
    ),
    "front/pairgrad_metrics.csv": (
        "pairgrad_RMSE",
        "pairgrad_MAE",
    ),
    "plume/summary_plume_support.csv": (
        "plume_IoU_dS005_mean",
        "FP_area_ratio_dS005_mean",
        "FN_area_ratio_dS005_mean",
    ),
    "physics/summary_physics_consistency.csv": (
        "PDE_total_RMSE",
        "PDE_total_p95",
        "local_FV_CO2_RMSE",
        "CO2_mass_rel_error_mean",
    ),
    "stability/summary_training_stability.csv": (
        "final_logged_loss",
        "best_logged_loss",
        "wall_time_hours",
        "max_gpu_max_alloc_gb",
    ),
}


def split_csv(text: str) -> list[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def infer_root_seed(path: Path) -> str:
    m = re.search(r"_seed(\d+)$", path.name)
    return m.group(1) if m else ""


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_table(csv_path: Path, root: Path, suite: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path).copy()
    root_seed = infer_root_seed(root)
    for col in ("suite", "source_eval_root", "source_csv", "source_seed"):
        if col in df.columns:
            df = df.drop(columns=col)
    if "seed" not in df.columns or df["seed"].isna().all():
        seed_col = pd.Series(root_seed, index=df.index, name="seed")
        df = pd.concat([seed_col, df.drop(columns=["seed"], errors="ignore")], axis=1)
    elif root_seed:
        df["seed"] = df["seed"].fillna(root_seed)

    meta_left = pd.DataFrame({"suite": suite}, index=df.index)
    meta_right = pd.DataFrame(
        {
            "source_eval_root": str(root),
            "source_csv": str(csv_path),
            "source_seed": root_seed if root_seed else df["seed"].astype(str),
        },
        index=df.index,
    )
    return pd.concat([meta_left, df, meta_right], axis=1).copy()


def discover_roots(patterns: Iterable[str]) -> list[Path]:
    roots: list[Path] = []
    for pat in patterns:
        matches = sorted(Path().glob(pat))
        if matches:
            roots.extend(p for p in matches if p.is_dir())
        else:
            p = Path(pat)
            if p.is_dir():
                roots.append(p)
    seen = set()
    out = []
    for root in roots:
        key = root.resolve()
        if key not in seen:
            seen.add(key)
            out.append(root)
    return out


def numeric_columns(df: pd.DataFrame, id_cols: list[str]) -> list[str]:
    out = []
    skip = set(id_cols) | PATH_LIKE_COLS | RUN_INSTANCE_COLS
    for col in df.columns:
        if col in skip:
            continue
        vals = pd.to_numeric(df[col], errors="coerce")
        if vals.notna().any():
            df[col] = vals
            out.append(col)
    return out


def grouping_columns(df: pd.DataFrame) -> list[str]:
    cols = [c for c in PREFERRED_ID_COLS if c in df.columns]
    if "exp_name" not in cols and "label" in df.columns:
        cols.append("label")
    return cols


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    work = df.copy()
    id_cols = grouping_columns(work)
    metric_cols = numeric_columns(work, id_cols)
    if not id_cols or not metric_cols:
        return pd.DataFrame()

    rows = []
    grouped = work.groupby(id_cols, dropna=False, sort=True)
    for key, group in grouped:
        if not isinstance(key, tuple):
            key = (key,)
        row = {col: val for col, val in zip(id_cols, key)}
        seeds = (
            group["seed"].astype(str).replace({"nan": ""}).dropna().unique().tolist()
            if "seed" in group.columns
            else []
        )
        seeds = sorted(s for s in seeds if s != "")
        row["n_seed_rows"] = int(len(group))
        row["n_unique_seeds"] = int(len(seeds))
        row["seeds"] = ",".join(seeds)
        for col in metric_cols:
            vals = pd.to_numeric(group[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            n = int(vals.shape[0])
            mean = float(vals.mean()) if n else np.nan
            std = float(vals.std(ddof=1)) if n > 1 else np.nan
            sem = float(std / math.sqrt(n)) if n > 1 and np.isfinite(std) else np.nan
            ci95 = float(1.96 * sem) if np.isfinite(sem) else np.nan
            row[f"{col}_n"] = n
            row[f"{col}_mean"] = mean
            row[f"{col}_std"] = std
            row[f"{col}_sem"] = sem
            row[f"{col}_ci95"] = ci95
        rows.append(row)
    return pd.DataFrame(rows)


def compact_primary(summary: pd.DataFrame, rel_table: str) -> pd.DataFrame:
    wanted = PRIMARY_METRICS.get(rel_table, ())
    if summary.empty or not wanted:
        return pd.DataFrame()
    id_cols = [c for c in grouping_columns(summary) if c in summary.columns]
    base_cols = id_cols + ["n_unique_seeds", "seeds"]
    rows = []
    for metric in wanted:
        mean_col = f"{metric}_mean"
        if mean_col not in summary.columns:
            continue
        cols = [c for c in base_cols if c in summary.columns]
        for suffix in ("mean", "std", "sem", "ci95", "n"):
            c = f"{metric}_{suffix}"
            if c in summary.columns:
                cols.append(c)
        part = summary[cols].copy()
        part.insert(len(id_cols), "metric", metric)
        rename = {
            f"{metric}_mean": "mean",
            f"{metric}_std": "std",
            f"{metric}_sem": "sem",
            f"{metric}_ci95": "ci95",
            f"{metric}_n": "n",
        }
        part = part.rename(columns=rename)
        rows.append(part)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def write_outputs(combined: pd.DataFrame, out_root: Path, suite: str, rel_table: str) -> dict[str, str]:
    table_path = Path(rel_table)
    out_dir = out_root / suite / table_path.parent
    ensure_dir(out_dir)
    stem = table_path.stem

    long_path = out_dir / f"{stem}_multiseed_long.csv"
    summary_path = out_dir / f"{stem}_multiseed_summary.csv"
    primary_path = out_dir / f"{stem}_primary_metrics.csv"

    combined.to_csv(long_path, index=False)
    summary = summarize(combined)
    if not summary.empty:
        summary.to_csv(summary_path, index=False)
    primary = compact_primary(summary, rel_table)
    if not primary.empty:
        primary.to_csv(primary_path, index=False)

    out = {"long": str(long_path)}
    if not summary.empty:
        out["summary"] = str(summary_path)
    if not primary.empty:
        out["primary"] = str(primary_path)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Aggregate M0-M7 evaluation CSV files across seed-specific evaluation roots.")
    ap.add_argument("--roots", default="Results2/eval_M0_M7_split_seed*", help="Comma-separated root globs or paths.")
    ap.add_argument("--suites", default="M0_M7,L0_L6", help="Comma-separated suite directories to scan.")
    ap.add_argument("--tables", default=",".join(DEFAULT_TABLES), help="Comma-separated table paths relative to each suite root.")
    ap.add_argument("--out-root", default="Results2/eval_M0_M7_multiseed", help="Output root for aggregated CSVs.")
    args = ap.parse_args()

    roots = discover_roots(split_csv(args.roots))
    suites = split_csv(args.suites)
    tables = split_csv(args.tables)
    out_root = Path(args.out_root)
    ensure_dir(out_root)

    manifest: dict[str, object] = {
        "roots": [str(p) for p in roots],
        "suites": suites,
        "tables": tables,
        "outputs": [],
    }

    if not roots:
        raise SystemExit(f"No evaluation roots matched --roots={args.roots!r}")

    for suite in suites:
        for rel_table in tables:
            parts = []
            for root in roots:
                csv_path = root / suite / rel_table
                if csv_path.exists():
                    parts.append(read_table(csv_path, root, suite))
            if not parts:
                continue
            combined = pd.concat(parts, ignore_index=True, sort=False)
            outputs = write_outputs(combined, out_root, suite, rel_table)
            manifest["outputs"].append({
                "suite": suite,
                "table": rel_table,
                "n_rows": int(combined.shape[0]),
                "n_source_roots": int(combined["source_eval_root"].nunique()),
                **outputs,
            })
            print(f"[aggregate] {suite}/{rel_table}: rows={combined.shape[0]} roots={combined['source_eval_root'].nunique()}")

    manifest_path = out_root / "multiseed_aggregation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[done] manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
