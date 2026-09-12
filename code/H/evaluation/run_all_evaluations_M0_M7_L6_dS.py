#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
All-evaluations dispatcher for the M0-M7/L0-L6 evaluation suite.

It prevents mixed M-series / LOO-series / training-strategy outputs by running
all eight evaluation scripts into three independent directory trees:

  <eval-root>/M0_M7/      -> forward ablation M0-M7, usually BASE training
  <eval-root>/L0_L6/      -> leave-one-out runs; L0 is the final M7 model
  <eval-root>/train/      -> training-strategy runs for final model, split by strategy

Each tree contains the same evaluator-output subdirectories:
  metadata, global, front, plume, physics, snapshots, stability, attribution, figures_for_paper

The dispatcher calls existing evaluator scripts.  It does not modify metrics.
It only standardizes run selection, output locations, and labels.
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Iterable

DEFAULT_EVAL_ROOT = "./eval_M0_M7_split"
DEFAULT_EXPERIMENTS = "M0,M1,M2,M3,M4,M5,M6,M7"
DEFAULT_LEAVE_OUTS = "NONE,PLAIN_TWONET,FRONT_PLUME,PAIRGRAD,RAR,FV,COARSE_DETAIL"
DEFAULT_TRAIN_STRATEGIES = "BASE,BETA,DIFF,BETA_DIFF"
DEFAULT_TIME_INDICES = "t0,early,middle,late,final"
DEFAULT_FIGURE_DPI = "600"

# Expected core script names. These may be changed from the command line if your
# local filenames differ.
CORE_DEFAULTS = {
    "metadata": "evaluate_metadata_M0_M7_L6_dS.py",
    "global": "evaluate_global_accuracy_M0_M7_L6_dS_shortlabels.py",
    "front_plume": "evaluate_front_plume_M0_M7_L6_dS.py",
    "physics": "evaluate_physics_M0_M7_L6_dS.py",
    "snapshots": "evaluate_snapshot_gridviz_M0_M7_publication_snapshots_L6_dS_fixedlegend_split3.py",
    "stability": "evaluate_training_stability_M0_M7_L6_dS.py",
    "attribution": "evaluate_attribution_M0_M7_L6_dS.py",
    "figures": "assemble_figures_for_paper_M0_M7_L6_dS.py",
}


def split_csv(s: str) -> list[str]:
    return [x.strip() for x in str(s).split(",") if x.strip()]


def q(cmd: Iterable[str]) -> str:
    return " ".join(shlex.quote(str(c)) for c in cmd)


def run(cmd: list[str], dry_run: bool = False, env: dict[str, str] | None = None) -> None:
    print("\n[CMD]", q(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, check=True, env=env)


def script_path(script_dir: Path, name: str) -> str:
    p = Path(name)
    if p.is_absolute():
        return str(p)
    p = script_dir / name
    if not p.exists():
        raise FileNotFoundError(f"Required evaluator script not found: {p}")
    return str(p)


def has_strategy_checkpoint(ckpt_dir: str, exp: str, strategy: str) -> bool:
    """Return True when at least one checkpoint for exp/strategy exists."""
    root = Path(ckpt_dir)
    if not root.exists():
        return False
    exp = str(exp).upper()
    strategy = str(strategy).upper()
    for p in root.glob("*.pt"):
        name = p.name.upper()
        if f"MODEL_{exp}_{strategy}_" in name:
            return True
    return False


def common_python(script: str) -> list[str]:
    return [sys.executable, script]


def run_metadata(args, suite_root: Path, suite: str, strategy: str | None = None):
    script = script_path(args.script_dir, args.metadata_script)
    cmd = common_python(script) + [
        "--eval-root", str(suite_root),
        "--ckpt-dir", args.ckpt_dir,
        "--log-dir", args.log_dir,
        "--out-dir", str(suite_root / "metadata"),
        "--seed", args.seed,
        "--experiments", args.experiments if suite == "M" else ("M7" if suite in ("L", "train") else args.experiments),
        "--training-strategy", strategy or args.training_strategy,
        "--leave-outs", args.leave_outs if suite == "L" else "",
        "--time-indices", args.time_indices,
        "--s-ic-co2", str(args.s_ic_co2),
        "--Snr", str(args.Snr),
        "--Sw-irr", str(args.Sw_irr),
    ]
    run(cmd + args.extra_args, args.dry_run)


def run_global(args, suite_root: Path, suite: str, strategy: str | None = None):
    script = script_path(args.script_dir, args.global_script)
    base = common_python(script) + [
        "--ckpt-dir", args.ckpt_dir,
        "--data", args.data,
        "--out-dir", str(suite_root / "global"),
        "--seed", args.seed,
        "--device", args.device,
        "--gpu-id", str(args.gpu_id),
        "--sample-per-time", str(args.sample_per_time),
        "--infer-batch-size", str(args.infer_batch_size),
        "--time-indices", args.time_indices,
        "--time-unit", args.time_unit,
        "--pressure-display-unit", args.pressure_display_unit,
        "--figure-dpi", args.figure_dpi,
        "--export-pdf", str(int(args.export_pdf)),
        "--export-tiff", str(int(args.export_tiff)),
        "--export-svg", str(int(args.export_svg)),
        "--s-ic-co2", str(args.s_ic_co2),
        "--Snr", str(args.Snr),
        "--Sw-irr", str(args.Sw_irr),
    ]
    if suite == "M":
        cmd = base + ["--experiments", args.experiments, "--training-strategy", args.training_strategy, "--include-loo", "0", "--plot-label-mode", "exp"]
    elif suite == "L":
        cmd = base + ["--experiments", "", "--training-strategy", args.training_strategy, "--include-loo", "1", "--loo-exp", "M7", "--leave-outs", args.leave_outs, "--plot-label-mode", "loo"]
    else:  # train: one strategy per suite_root/<strategy>
        cmd = base + ["--experiments", "M7", "--training-strategy", strategy or args.training_strategy, "--include-loo", "0", "--plot-label-mode", "exp"]
    run(cmd + args.extra_args, args.dry_run)


def run_front_plume(args, suite_root: Path, suite: str, strategy: str | None = None):
    script = script_path(args.script_dir, args.front_plume_script)
    base = common_python(script) + [
        "--ckpt-dir", args.ckpt_dir,
        "--data", args.data,
        "--out-root", str(suite_root),
        "--seed", args.seed,
        "--device", args.device,
        "--gpu-id", str(args.gpu_id),
        "--sample-per-time", str(args.sample_per_time),
        "--infer-batch-size", str(args.infer_batch_size),
        "--time-indices", args.time_indices,
        "--time-unit", args.time_unit,
        "--grid-nx", str(args.grid_nx_front),
        "--grid-ny", str(args.grid_ny_front),
        "--figure-dpi", args.figure_dpi,
        "--export-pdf", str(int(args.export_pdf)),
        "--export-tiff", str(int(args.export_tiff)),
        "--export-svg", str(int(args.export_svg)),
        "--s-ic-co2", str(args.s_ic_co2),
    ]
    if suite == "M":
        cmd = base + ["--experiments", args.experiments, "--training-strategy", args.training_strategy, "--include-loo", "0", "--loo-labels", "0"]
    elif suite == "L":
        cmd = base + ["--experiments", "", "--training-strategy", args.training_strategy, "--include-loo", "1", "--loo-exp", "M7", "--leave-outs", args.leave_outs, "--loo-labels", "1"]
    else:
        cmd = base + ["--experiments", "M7", "--training-strategy", strategy or args.training_strategy, "--include-loo", "0", "--loo-labels", "0"]
    run(cmd + args.extra_args, args.dry_run)


def run_physics(args, suite_root: Path, suite: str, strategy: str | None = None):
    script = script_path(args.script_dir, args.physics_script)
    base = common_python(script) + [
        "--ckpt-dir", args.ckpt_dir,
        "--data", args.data,
        "--out-root", str(suite_root),
        "--seed", args.seed,
        "--device", args.device,
        "--gpu-id", str(args.gpu_id),
        "--infer-batch-size", str(args.infer_batch_size),
        "--time-indices", args.time_indices,
        "--time-unit", args.time_unit,
        "--pde-points", str(args.pde_points),
        "--pde-batch-size", str(args.pde_batch_size),
        "--bc-points", str(args.bc_points),
        "--ic-points", str(args.ic_points),
        "--mass-sample-per-time", str(args.mass_sample_per_time),
        "--fv-n-cells-eval", str(args.fv_n_cells_eval),
        "--fv-batch-size", str(args.fv_batch_size),
        "--figure-dpi", args.figure_dpi,
        "--export-pdf", str(int(args.export_pdf)),
        "--export-tiff", str(int(args.export_tiff)),
        "--export-svg", str(int(args.export_svg)),
        "--s-ic-co2", str(args.s_ic_co2),
        "--Snr", str(args.Snr),
        "--Sw-irr", str(args.Sw_irr),
    ]
    if suite == "M":
        cmd = base + ["--experiments", args.experiments, "--training-strategy", args.training_strategy, "--include-loo", "0", "--loo-labels", "0"]
    elif suite == "L":
        cmd = base + ["--experiments", "", "--training-strategy", args.training_strategy, "--include-loo", "1", "--loo-exp", "M7", "--leave-outs", args.leave_outs, "--loo-labels", "1"]
    else:
        cmd = base + ["--experiments", "M7", "--training-strategy", strategy or args.training_strategy, "--include-loo", "0", "--loo-labels", "0"]
    run(cmd + args.extra_args, args.dry_run)


def run_snapshots(args, suite_root: Path, suite: str, strategy: str | None = None):
    script = script_path(args.script_dir, args.snapshots_script)
    base = common_python(script) + [
        "--ckpt-dir", args.ckpt_dir,
        "--data", args.data,
        "--out-root", str(suite_root / "snapshots"),
        "--seed", args.seed,
        "--device", args.device,
        "--gpu-id", str(args.gpu_id),
        "--time-indices", args.time_indices,
        "--time-unit", args.time_unit,
        "--grid-nx", str(args.grid_nx_snapshot),
        "--grid-ny", str(args.grid_ny_snapshot),
        "--infer-batch-size", str(args.infer_batch_size),
        "--pressure-display-unit", args.pressure_display_unit,
        "--figure-dpi", args.figure_dpi,
        "--export-pdf", str(int(args.export_pdf)),
        "--export-tiff", str(int(args.export_tiff)),
        "--export-svg", str(int(args.export_svg)),
        "--s-ic-co2", str(args.s_ic_co2),
        "--Snr", str(args.Snr),
        "--threshold-mode", "excess",
    ]
    if suite == "M":
        cmd = base + ["--experiments", args.experiments, "--training-strategy", args.training_strategy, "--include-loo", "0"]
    elif suite == "L":
        cmd = base + ["--experiments", "", "--training-strategy", args.training_strategy, "--include-loo", "1", "--loo-exp", "M7", "--leave-outs", args.leave_outs]
    else:
        cmd = base + ["--experiments", "M7", "--training-strategy", strategy or args.training_strategy, "--include-loo", "0"]
    run(cmd + args.extra_args, args.dry_run)


def run_stability(args, suite_root: Path, suite: str, strategy: str | None = None):
    script = script_path(args.script_dir, args.stability_script)
    base = common_python(script) + [
        "--log-dir", args.log_dir,
        "--out-root", str(suite_root / "stability"),
        "--seed", args.seed,
        "--recursive", "1",
        "--figure-dpi", args.figure_dpi,
        "--export-pdf", str(int(args.export_pdf)),
        "--export-tiff", str(int(args.export_tiff)),
        "--export-svg", str(int(args.export_svg)),
        "--time-indices", args.time_indices,
        "--s-ic-co2", str(args.s_ic_co2),
        "--Snr", str(args.Snr),
    ]
    if suite == "M":
        cmd = base + ["--experiments", args.experiments, "--training-strategy", args.training_strategy, "--include-loo", "0", "--loo-labels", "0"]
    elif suite == "L":
        cmd = base + ["--experiments", "M7", "--training-strategy", args.training_strategy, "--include-loo", "1", "--leave-outs", args.leave_outs, "--loo-labels", "1"]
    else:
        cmd = base + ["--experiments", "M7", "--training-strategy", strategy or args.training_strategy, "--include-loo", "0", "--loo-labels", "0"]
    run(cmd + args.extra_args, args.dry_run)


def run_attribution(args, suite_root: Path, suite: str, strategy: str | None = None):
    script = script_path(args.script_dir, args.attribution_script)
    base = common_python(script) + [
        "--eval-root", str(suite_root),
        "--out-dir", str(suite_root / "attribution"),
        "--figure-dpi", args.figure_dpi,
        "--export-pdf", str(int(args.export_pdf)),
        "--export-tiff", str(int(args.export_tiff)),
        "--export-svg", str(int(args.export_svg)),
        "--s-ic-co2", str(args.s_ic_co2),
        "--Snr", str(args.Snr),
    ]
    if suite == "M":
        cmd = base + ["--experiments", args.experiments, "--leave-outs", "", "--include-l6", "0"]
    elif suite == "L":
        cmd = base + ["--experiments", "M7", "--leave-outs", args.leave_outs, "--include-l6", "1"]
    else:
        cmd = base + ["--experiments", "M7", "--leave-outs", "", "--include-l6", "0"]
    run(cmd + args.extra_args, args.dry_run)


def run_figures(args, suite_root: Path, suite: str, strategy: str | None = None):
    script = script_path(args.script_dir, args.figures_script)
    base = common_python(script) + [
        "--eval-root", str(suite_root),
        "--out-dir", str(suite_root / "figures_for_paper"),
        "--time-indices", args.time_indices,
        "--figure-dpi", args.figure_dpi,
        "--export-pdf", str(int(args.export_pdf)),
        "--export-tiff", str(int(args.export_tiff)),
        "--export-svg", str(int(args.export_svg)),
        "--s-ic-co2", str(args.s_ic_co2),
        "--Snr", str(args.Snr),
        "--make-snapshot-panels", "1",
        "--snapshot-panel-time", "final",
    ]
    if suite == "M":
        cmd = base + ["--experiments", args.experiments, "--leave-outs", ""]
    elif suite == "L":
        cmd = base + ["--experiments", "M7", "--leave-outs", args.leave_outs]
    else:
        cmd = base + ["--experiments", "M7", "--leave-outs", ""]
    run(cmd + args.extra_args, args.dry_run)


def parse_args():
    ap = argparse.ArgumentParser(description="Run M0-M7 evaluation scripts into separated M/L/train output trees.")
    ap.add_argument("--eval-root", default=DEFAULT_EVAL_ROOT)
    ap.add_argument("--suite", default="all", choices=["all", "M", "L", "train"])
    ap.add_argument("--script-dir", default=".", type=Path)
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--log-dir", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--experiments", default=DEFAULT_EXPERIMENTS)
    ap.add_argument("--training-strategy", default="BASE")
    ap.add_argument("--train-strategies", default=DEFAULT_TRAIN_STRATEGIES)
    ap.add_argument("--leave-outs", default=DEFAULT_LEAVE_OUTS)
    ap.add_argument("--seed", default="ANY")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--gpu-id", type=int, default=0)
    ap.add_argument("--time-indices", default=DEFAULT_TIME_INDICES)
    ap.add_argument("--time-unit", default="days")
    ap.add_argument("--pressure-display-unit", default="MPa")
    ap.add_argument("--figure-dpi", default=DEFAULT_FIGURE_DPI)
    ap.add_argument("--export-pdf", type=int, default=1)
    ap.add_argument("--export-tiff", type=int, default=1)
    ap.add_argument("--export-svg", type=int, default=1,
                    help="Export editable SVG outputs from all Nature-style plotting scripts.")
    ap.add_argument("--sample-per-time", type=int, default=20000)
    ap.add_argument("--infer-batch-size", type=int, default=65536)
    ap.add_argument("--grid-nx-front", type=int, default=420)
    ap.add_argument("--grid-ny-front", type=int, default=420)
    ap.add_argument("--grid-nx-snapshot", type=int, default=500)
    ap.add_argument("--grid-ny-snapshot", type=int, default=500)
    ap.add_argument("--pde-points", type=int, default=4096)
    ap.add_argument("--pde-batch-size", type=int, default=512)
    ap.add_argument("--bc-points", type=int, default=2048)
    ap.add_argument("--ic-points", type=int, default=4096)
    ap.add_argument("--mass-sample-per-time", type=int, default=30000)
    ap.add_argument("--fv-n-cells-eval", type=int, default=2048)
    ap.add_argument("--fv-batch-size", type=int, default=256)
    ap.add_argument("--s-ic-co2", type=float, default=0.0)
    ap.add_argument("--Snr", type=float, default=0.0)
    ap.add_argument("--Sw-irr", type=float, default=0.20)
    ap.add_argument("--steps", default="metadata,global,front_plume,physics,snapshots,stability,attribution,figures",
                    help="Comma-separated steps to run. Valid: metadata,global,front_plume,physics,snapshots,stability,attribution,figures")
    ap.add_argument("--dry-run", action="store_true")
    # Core scripts.
    for key, default in CORE_DEFAULTS.items():
        ap.add_argument(f"--{key.replace('_','-')}-script", default=default)
    ap.add_argument("extra_args", nargs=argparse.REMAINDER, help="Extra args appended to each called evaluator after '--'.")
    args = ap.parse_args()
    if args.extra_args and args.extra_args[0] == "--":
        args.extra_args = args.extra_args[1:]
    args.script_dir = Path(args.script_dir).resolve()
    return args


def main():
    args = parse_args()
    root = Path(args.eval_root).resolve()
    steps = split_csv(args.steps)
    suite_order = [args.suite] if args.suite != "all" else ["M", "L", "train"]
    runners = {
        "metadata": run_metadata,
        "global": run_global,
        "front_plume": run_front_plume,
        "physics": run_physics,
        "snapshots": run_snapshots,
        "stability": run_stability,
        "attribution": run_attribution,
        "figures": run_figures,
    }
    for suite in suite_order:
        if suite == "M":
            suite_root = root / "M0_M7"
            print(f"\n===== M-series output: {suite_root} =====")
            for step in steps:
                runners[step](args, suite_root, suite)
        elif suite == "L":
            suite_root = root / "L0_L6"
            print(f"\n===== LOO-series output: {suite_root} =====")
            for step in steps:
                runners[step](args, suite_root, suite)
        elif suite == "train":
            train_root = root / "train"
            print(f"\n===== Training-strategy output: {train_root} =====")
            for strategy in split_csv(args.train_strategies):
                if not has_strategy_checkpoint(args.ckpt_dir, "M7", strategy):
                    print(f"\n--- train strategy {strategy}: skipped, no M7 {strategy} checkpoint found in {args.ckpt_dir} ---")
                    continue
                suite_root = train_root / strategy
                print(f"\n--- train strategy {strategy}: {suite_root} ---")
                for step in steps:
                    runners[step](args, suite_root, suite, strategy=strategy)
        else:
            raise ValueError(suite)
    print("\n[done] separated evaluation outputs written under", root)


if __name__ == "__main__":
    main()
