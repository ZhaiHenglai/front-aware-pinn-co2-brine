#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Metadata/index builder for the M0-M7 evaluation suite.

Purpose
-------
Creates a reproducible metadata layer for:
  - M0-M7 forward ablation runs;
  - M7 BASE leave-one-out runs L0-L6;
  - checkpoint/log/config traceability;
  - metric definitions and plotting conventions shared by global/front/plume/
    physics/snapshots/stability/attribution evaluators.

This script does not recompute predictions or physics residuals. S_IC_CO2=0.0
and Snr=0.0 are stored as evaluation metadata by default. The actual numerical handling
belongs to the global/front/plume/physics/snapshot evaluators.
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Optional

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None


DEFAULT_EVAL_ROOT = "./eval_M0_M7"
DEFAULT_CKPT_DIR = "./Results1"
DEFAULT_LOG_DIR = "./logs"
DEFAULT_OUT_DIR = "./eval_M0_M7/metadata"
DEFAULT_EXPERIMENTS = ",".join([f"M{i}" for i in range(8)])
DEFAULT_LEAVE_OUTS = "NONE,PLAIN_TWONET,FRONT_PLUME,PAIRGRAD,RAR,FV,COARSE_DETAIL"
DEFAULT_TRAINING_STRATEGY = "BASE"
DEFAULT_TIME_INDICES = "t0,early,middle,late,final"
DEFAULT_S_IC_CO2 = 0.0
DEFAULT_SNR = 0.0
DEFAULT_SW_IRR = 0.20

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
    "PLAIN_TWONET": "plain TwoNet",
    "FRONT_PLUME": "front/plume loss",
    "PAIRGRAD": "pairwise front-gradient",
    "RAR": "residual RAR",
    "FV": "FV mass loss",
    "COARSE_DETAIL": "coarse/detail branch",
    "WATER_FV": "weak-water FV",
    "MSFF": "MS Fourier",
}

# Supports examples:
#   model_M7_BASE_LOO-FV_seed0_v100_final.pt
#   model_M7_BASE_seed0_v100_final.pt
#   model_M0_seed0_v100_final.pt
RE_CKPT = re.compile(
    r"model_(?P<exp>M\d+)"
    r"(?:_(?P<strategy>BASE|BETA|DIFF|BETA_DIFF))?"
    r"(?:_LOO-(?P<leave>[A-Za-z0-9_\-]+))?"
    r"_seed(?P<seed>\d+)"
    r"_(?P<profile>[^_]+)"
    r"_(?P<suffix>[^/\\]+?)\.pt$"
)
RE_EXP = re.compile(r"\b(M\d+)\b")
RE_STRATEGY = re.compile(r"\b(BASE|BETA|DIFF|BETA_DIFF)\b")
RE_LEAVE = re.compile(r"LOO[-_](NONE|PLAIN_TWONET|FRONT_PLUME|PAIRGRAD|RAR|FV|COARSE_DETAIL|WATER_FV|MSFF)")
RE_SEED = re.compile(r"seed(\d+)")
RE_EXP_HEADER = re.compile(r"^\s*EXP_NAME:\s*(M\d+)\s*$", re.MULTILINE)
RE_STRATEGY_HEADER = re.compile(r"^\s*Strategy:\s*(BASE|BETA|DIFF|BETA_DIFF)\s*$", re.MULTILINE)
RE_LEAVE_HEADER = re.compile(r"^\s*Leave-out:\s*([A-Za-z0-9_\-]+)\s*$", re.MULTILINE)
RE_SEED_HEADER = re.compile(r"^\s*SEED:\s*(\d+)\s*$", re.MULTILINE)
RE_JOBID = re.compile(r"^\s*JobID:\s*(\S+)\s*$", re.MULTILINE)
RE_HOST = re.compile(r"^\s*Host:\s*(.+?)\s*$", re.MULTILINE)
RE_START = re.compile(r"^\s*Start:\s*(.+?)\s*$", re.MULTILINE)
RE_END = re.compile(r"^\s*End:\s*(.+?)\s*$", re.MULTILINE)
RE_CKPT_PATH = re.compile(r"(?:CKPT|checkpoint).*?(\/\S+model_M\d+\S+?\.pt)")


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def parse_list(s: str) -> list[str]:
    return [x.strip() for x in str(s).split(",") if x.strip()]


def sha256_file(path: Path, chunk: int = 2**20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def safe_read_text(path: Path, max_bytes: int = 5_000_000) -> str:
    try:
        with path.open("rb") as f:
            data = f.read(max_bytes)
        return data.decode("utf-8", errors="replace")
    except Exception:
        return ""


def safe_json_dump(obj: Any, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True, ensure_ascii=False, default=str)


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    seen = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                keys.append(k)
                seen.add(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def infer_from_name(name: str, default_strategy: str) -> dict[str, Any]:
    m = RE_CKPT.search(name)
    if m:
        d = m.groupdict()
        return {
            "exp_name": d.get("exp"),
            "training_strategy": d.get("strategy") or default_strategy,
            "leave_out": (d.get("leave") or "NONE").upper(),
            "seed": int(d["seed"]) if d.get("seed") is not None else None,
            "runtime_profile": d.get("profile"),
            "checkpoint_suffix": d.get("suffix"),
        }
    exp = RE_EXP.search(name)
    strat = RE_STRATEGY.search(name)
    leave = RE_LEAVE.search(name)
    seed = RE_SEED.search(name)
    return {
        "exp_name": exp.group(1) if exp else None,
        "training_strategy": strat.group(1) if strat else default_strategy,
        "leave_out": leave.group(1).upper() if leave else "NONE",
        "seed": int(seed.group(1)) if seed else None,
        "runtime_profile": None,
        "checkpoint_suffix": None,
    }


def short_label(exp: Optional[str], leave: Optional[str]) -> str:
    leave = (leave or "NONE").upper()
    if leave != "NONE":
        return LOO_LABELS.get(leave, f"LOO-{leave}")
    return exp or "unknown"


def canonical_run_id(exp: Optional[str], strategy: Optional[str], leave: Optional[str], seed: Optional[int]) -> str:
    exp_s = exp or "UNK"
    strat_s = strategy or DEFAULT_TRAINING_STRATEGY
    leave_s = (leave or "NONE").upper()
    seed_s = "NA" if seed is None else str(seed)
    return f"{exp_s}_{strat_s}_LOO-{leave_s}_seed{seed_s}"


def extract_ckpt_metadata(path: Path) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "checkpoint_load_ok": False,
        "checkpoint_has_config": False,
        "checkpoint_config_keys": "",
    }
    if torch is None:
        meta["checkpoint_load_error"] = "torch not available"
        return meta
    try:
        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
        meta["checkpoint_load_ok"] = True
        if isinstance(ckpt, dict):
            cfg = ckpt.get("config", None)
            if isinstance(cfg, dict):
                meta["checkpoint_has_config"] = True
                meta["checkpoint_config_keys"] = ";".join(sorted(map(str, cfg.keys())))
                # Extract common nested metadata without expanding every checkpoint value.
                arch = cfg.get("architecture", {}) if isinstance(cfg.get("architecture", {}), dict) else {}
                sat_ic = cfg.get("saturation_ic", {}) if isinstance(cfg.get("saturation_ic", {}), dict) else {}
                m4_front = cfg.get("m4_front_plume_loss", {}) if isinstance(cfg.get("m4_front_plume_loss", {}), dict) else {}
                m5_pair = cfg.get("m5_pairwise_front_gradient", {}) if isinstance(cfg.get("m5_pairwise_front_gradient", {}), dict) else {}
                m6_rar = cfg.get("m6_rar", {}) if isinstance(cfg.get("m6_rar", {}), dict) else {}
                m7_fv = cfg.get("m7_fv_mass", {}) if isinstance(cfg.get("m7_fv_mass", {}), dict) else {}
                sched = cfg.get("training_schedules", {}) if isinstance(cfg.get("training_schedules", {}), dict) else {}
                phys = cfg.get("physical_parameters", {}) if isinstance(cfg.get("physical_parameters", {}), dict) else {}
                flat = {
                    "ckpt_model_type": cfg.get("model_type"),
                    "ckpt_dtype": cfg.get("dtype"),
                    "ckpt_use_twonet": arch.get("use_twonet"),
                    "ckpt_use_msfourier": arch.get("use_msfourier"),
                    "ckpt_decouple_ps_input": arch.get("decouple_ps_input"),
                    "ckpt_use_coarse_detail": arch.get("use_coarse_detail_s_branch"),
                    "ckpt_s_ic_co2": sat_ic.get("s_ic_co2", phys.get("S_ic_co2", phys.get("S_IC_CO2"))),
                    "ckpt_s_inj_co2": sat_ic.get("s_inj_co2", phys.get("S_inj_co2")),
                    "ckpt_s_co2_output_max": sat_ic.get("s_co2_output_max", phys.get("S_CO2_MAX")),
                    "ckpt_use_front_loss": m4_front.get("front_loss"),
                    "ckpt_use_plume_loss": m4_front.get("plume_loss"),
                    "ckpt_front_s_min": m4_front.get("front_s_min"),
                    "ckpt_front_s_max": m4_front.get("front_s_max"),
                    "ckpt_plume_threshold": m4_front.get("plume_threshold"),
                    "ckpt_use_pairgrad": m5_pair.get("enabled"),
                    "ckpt_pairgrad_s_min": m5_pair.get("s_min"),
                    "ckpt_pairgrad_s_max": m5_pair.get("s_max"),
                    "ckpt_use_rar": m6_rar.get("enabled"),
                    "ckpt_use_fv": m7_fv.get("enabled"),
                    "ckpt_fv_w_water": m7_fv.get("fv_w_water"),
                    "ckpt_fv_dt_tilde": m7_fv.get("fv_dt_tilde"),
                    "ckpt_training_use_beta": sched.get("use_beta_curriculum"),
                    "ckpt_training_use_diffusion": sched.get("use_diffusion_decay"),
                    "ckpt_Snr": phys.get("Snr"),
                    "ckpt_Sw_irr": phys.get("Sw_irr"),
                    "ckpt_L_ref": phys.get("L_ref"),
                    "ckpt_T_ref": phys.get("T_ref"),
                }
                meta.update({k: v for k, v in flat.items() if v is not None})
    except Exception as e:  # pragma: no cover
        meta["checkpoint_load_error"] = repr(e)
    return meta


def scan_checkpoints(ckpt_dir: Path, default_strategy: str, compute_hash: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not ckpt_dir.exists():
        return rows
    for path in sorted(ckpt_dir.rglob("model_M*.pt")):
        info = infer_from_name(path.name, default_strategy)
        run_id = canonical_run_id(info.get("exp_name"), info.get("training_strategy"), info.get("leave_out"), info.get("seed"))
        st = path.stat()
        row = {
            "run_id": run_id,
            "short_label": short_label(info.get("exp_name"), info.get("leave_out")),
            "checkpoint_path": str(path),
            "checkpoint_name": path.name,
            "file_size_bytes": st.st_size,
            "modified_time_iso": _dt.datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
            **info,
        }
        if compute_hash:
            row["sha256"] = sha256_file(path)
        row.update(extract_ckpt_metadata(path))
        rows.append(row)
    return rows


def scan_logs(log_dir: Path, default_strategy: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not log_dir.exists():
        return rows
    suffixes = {".out", ".err", ".log", ".txt"}
    for path in sorted(p for p in log_dir.rglob("*") if p.is_file() and p.suffix.lower() in suffixes):
        txt = safe_read_text(path)
        info = infer_from_name(path.name, default_strategy)
        # If filename is not informative, inspect text.
        exp_header = RE_EXP_HEADER.search(txt)
        if exp_header:
            info["exp_name"] = exp_header.group(1)
        elif info.get("exp_name") is None:
            exp = RE_EXP.search(txt)
            if exp:
                info["exp_name"] = exp.group(1)
        strat_header = RE_STRATEGY_HEADER.search(txt)
        if strat_header:
            info["training_strategy"] = strat_header.group(1)
        elif not info.get("training_strategy"):
            strat = RE_STRATEGY.search(txt)
            if strat:
                info["training_strategy"] = strat.group(1)
        leave_header = RE_LEAVE_HEADER.search(txt)
        if leave_header:
            info["leave_out"] = leave_header.group(1).upper()
        else:
            leave = RE_LEAVE.search(txt)
            if leave:
                info["leave_out"] = leave.group(1).upper()
        seed_header = RE_SEED_HEADER.search(txt)
        if seed_header:
            info["seed"] = int(seed_header.group(1))
        else:
            seed = RE_SEED.search(txt)
            if seed and info.get("seed") is None:
                info["seed"] = int(seed.group(1))
        run_id = canonical_run_id(info.get("exp_name"), info.get("training_strategy"), info.get("leave_out"), info.get("seed"))
        ckpt_paths = RE_CKPT_PATH.findall(txt)
        row = {
            "run_id": run_id,
            "short_label": short_label(info.get("exp_name"), info.get("leave_out")),
            "log_path": str(path),
            "log_name": path.name,
            "file_size_bytes": path.stat().st_size,
            "job_id": (RE_JOBID.search(txt).group(1) if RE_JOBID.search(txt) else None),
            "host": (RE_HOST.search(txt).group(1) if RE_HOST.search(txt) else None),
            "start_text": (RE_START.search(txt).group(1) if RE_START.search(txt) else None),
            "end_text": (RE_END.search(txt).group(1) if RE_END.search(txt) else None),
            "contains_traceback": int("Traceback" in txt),
            "contains_nan": int(bool(re.search(r"\bnan\b", txt, flags=re.IGNORECASE))),
            "contains_inf": int(bool(re.search(r"\binf\b", txt, flags=re.IGNORECASE))),
            "ckpt_paths_mentioned": ";".join(ckpt_paths[:20]),
            **info,
        }
        rows.append(row)
    return rows


def make_expected_runs(experiments: list[str], leave_outs: list[str], strategy: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for exp in experiments:
        rows.append({
            "run_id": canonical_run_id(exp, strategy, "NONE", None),
            "short_label": exp,
            "exp_name": exp,
            "training_strategy": strategy,
            "leave_out": "NONE",
            "run_group": "M_forward",
            "expected": 1,
        })
    for leave in leave_outs:
        leave = leave.upper()
        rows.append({
            "run_id": canonical_run_id("M7", strategy, leave, None),
            "short_label": LOO_LABELS.get(leave, leave),
            "exp_name": "M7",
            "training_strategy": strategy,
            "leave_out": leave,
            "removed_module": LOO_MODULES.get(leave, ""),
            "run_group": "L_leave_one_out",
            "expected": 1,
        })
    return rows


def build_run_index(expected: list[dict[str, Any]], ckpts: list[dict[str, Any]], logs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # Aggregate by exp/strategy/leave/seed when possible. Expected rows have seed NA.
    ckpt_by_key: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for r in ckpts:
        key = (r.get("exp_name"), r.get("training_strategy"), r.get("leave_out"), r.get("seed"))
        ckpt_by_key.setdefault(key, []).append(r)
    log_by_key: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for r in logs:
        key = (r.get("exp_name"), r.get("training_strategy"), r.get("leave_out"), r.get("seed"))
        log_by_key.setdefault(key, []).append(r)

    rows: list[dict[str, Any]] = []
    keys = set(ckpt_by_key.keys()) | set(log_by_key.keys())
    for e in expected:
        # Include actual seeds if present; otherwise placeholder expected row.
        matching = [k for k in keys if k[0] == e.get("exp_name") and k[1] == e.get("training_strategy") and k[2] == e.get("leave_out")]
        if not matching:
            row = dict(e)
            row.update({"seed": None, "checkpoint_found": 0, "log_found": 0})
            rows.append(row)
        else:
            for k in sorted(matching, key=lambda x: (-1 if x[3] is None else x[3])):
                ck = ckpt_by_key.get(k, [])
                lg = log_by_key.get(k, [])
                row = dict(e)
                row.update({
                    "run_id": canonical_run_id(k[0], k[1], k[2], k[3]),
                    "short_label": short_label(k[0], k[2]),
                    "seed": k[3],
                    "checkpoint_found": int(bool(ck)),
                    "n_checkpoints": len(ck),
                    "checkpoint_paths": ";".join(r.get("checkpoint_path", "") for r in ck),
                    "latest_checkpoint_path": ck[-1].get("checkpoint_path") if ck else None,
                    "log_found": int(bool(lg)),
                    "n_logs": len(lg),
                    "log_paths": ";".join(r.get("log_path", "") for r in lg),
                    "has_traceback_in_logs": int(any(r.get("contains_traceback") for r in lg)),
                    "has_nan_in_logs": int(any(r.get("contains_nan") for r in lg)),
                    "has_inf_in_logs": int(any(r.get("contains_inf") for r in lg)),
                })
                # Pull a few checkpoint metadata values from the latest checkpoint.
                if ck:
                    latest = ck[-1]
                    for fld in [
                        "ckpt_use_twonet", "ckpt_use_msfourier", "ckpt_use_coarse_detail",
                        "ckpt_s_ic_co2", "ckpt_s_inj_co2", "ckpt_s_co2_output_max",
                        "ckpt_use_front_loss", "ckpt_use_plume_loss",
                        "ckpt_front_s_min", "ckpt_front_s_max", "ckpt_plume_threshold",
                        "ckpt_use_pairgrad", "ckpt_pairgrad_s_min", "ckpt_pairgrad_s_max",
                        "ckpt_use_rar", "ckpt_use_fv", "ckpt_fv_w_water", "ckpt_fv_dt_tilde",
                        "ckpt_Snr", "ckpt_Sw_irr",
                    ]:
                        row[fld] = latest.get(fld)
                rows.append(row)
    return rows


def metric_definitions(s_ic: float, snr: float) -> dict[str, Any]:
    return {
        "global": {
            "S_RMSE": "root-mean-square error between raw Sco2 prediction and raw Sco2 reference",
            "p_RMSE_MPa": "pressure RMSE in MPa after dimensional conversion",
            "front_band_region": "S_true in [0.05, 0.30] by default, aligned with the M4 front/plume and M5 pair-gradient band",
            "plume_region_raw_legacy": "S_true > 0.05 is retained only as a legacy raw-S diagnostic",
            "plume_region_dS_preferred": "preferred plume support uses dS = S - S_IC_CO2 > 0.05",
        },
        "front": {
            "front_band_RMSE": "saturation RMSE in selected transition band",
            "contour_chamfer_front": "preferred symmetric nearest-distance between predicted and true S=0.175 contours",
            "contour_chamfer_S040": "legacy symmetric nearest-distance between predicted and true S=0.40 contours",
            "pairgrad_RMSE": "RMSE of pairwise finite-difference saturation gradients on same-time neighbor pairs",
            "speckle_area_ratio_dS002": "preferred speckle area fraction where dS_pred > 0.02 and dS_true < 0.01",
            "speckle_area_ratio_002": "legacy raw-S speckle diagnostic",
        },
        "plume": {
            "plume_IoU_dS005": "preferred IoU for excess plume support dS > 0.05",
            "FP_area_ratio_dS005": "preferred false-positive excess plume area normalized by true excess plume area",
            "FN_area_ratio_dS005": "preferred false-negative excess plume area normalized by true excess plume area",
            "plume_IoU_S005": "legacy raw-S IoU diagnostic",
        },
        "physics": {
            "IC_S_RMSE": f"initial saturation RMSE relative to S_IC_CO2={s_ic}",
            "PDE_total_p95": "95th percentile of combined two-phase PDE residual magnitude",
            "local_FV_CO2_RMSE": "local finite-volume CO2 balance residual RMSE on evaluation control volumes",
            "CO2_mass_rel_error": "relative error in global CO2 mass compared with reference data",
        },
        "stability": {
            "tail_oscillation": "standard deviation or median absolute difference of log10 loss in the tail of training",
            "ramp_loss_jump": "ratio of post-ramp to pre-ramp loss around PDE/pairgrad/FV activation",
        },
        "saturation_convention": {
            "raw_Sco2": "model output and data target remain raw Sco2; do not remap to effective saturation for accuracy figures",
            "S_IC_CO2": s_ic,
            "Snr": snr,
            "Snr_usage": "used in relative permeability/PDE/FV physics, not for remapping raw Sco2 predictions",
        },
    }


def figure_style_contract() -> dict[str, Any]:
    """Document the shared Nature-style figure contract used by the suite."""
    return {
        "style_module": "_nature_plot_style_M0_M7.py",
        "font_family": "sans-serif, preferring Arial/Helvetica with DejaVu Sans fallback",
        "editable_vector_text": {"svg.fonttype": "none", "pdf.fonttype": 42},
        "primary_export": "SVG",
        "screening_export": "PNG",
        "submission_exports": ["PDF", "TIFF"],
        "method_labels": "compact M0-M7, L0-L6, and BASE/BETA/DIFF/BETA_DIFF labels only",
        "color_policy": "low-saturation method palette with M7/L0 as the visual anchor",
        "image_plate_policy": "field snapshots keep raster fields but preserve vector labels/colorbars where SVG/PDF is enabled",
    }


def directory_tree(root: Path, max_depth: int = 3) -> str:
    root = root.resolve()
    lines = [str(root)]
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        depth = len(rel.parts)
        if depth > max_depth:
            continue
        indent = "  " * depth
        suffix = "/" if path.is_dir() else ""
        lines.append(f"{indent}{path.name}{suffix}")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="Build metadata indices for M0-M7 evaluation outputs.")
    ap.add_argument("--eval-root", default=DEFAULT_EVAL_ROOT)
    ap.add_argument("--ckpt-dir", default=DEFAULT_CKPT_DIR)
    ap.add_argument("--log-dir", default=DEFAULT_LOG_DIR)
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--experiments", default=DEFAULT_EXPERIMENTS)
    ap.add_argument("--training-strategy", default=DEFAULT_TRAINING_STRATEGY)
    ap.add_argument("--seed", default="ANY", help="Training seed to index; ANY keeps all discovered seeds.")
    ap.add_argument("--leave-outs", default=DEFAULT_LEAVE_OUTS)
    ap.add_argument("--time-indices", default=DEFAULT_TIME_INDICES)
    ap.add_argument("--s-ic-co2", type=float, default=DEFAULT_S_IC_CO2)
    ap.add_argument("--Snr", type=float, default=DEFAULT_SNR)
    ap.add_argument("--Sw-irr", type=float, default=DEFAULT_SW_IRR)
    ap.add_argument("--compute-checkpoint-hash", type=int, default=0)
    ap.add_argument("--directory-tree-depth", type=int, default=3)
    args = ap.parse_args()

    eval_root = Path(args.eval_root)
    ckpt_dir = Path(args.ckpt_dir)
    log_dir = Path(args.log_dir)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    experiments = parse_list(args.experiments)
    leave_outs = [x.upper() for x in parse_list(args.leave_outs)]
    time_indices = parse_list(args.time_indices)

    ckpts = scan_checkpoints(ckpt_dir, args.training_strategy, bool(args.compute_checkpoint_hash))
    logs = scan_logs(log_dir, args.training_strategy)
    if str(args.seed).upper() != "ANY":
        seed = int(args.seed)
        ckpts = [r for r in ckpts if r.get("seed") == seed]
        logs = [r for r in logs if r.get("seed") == seed]
    expected = make_expected_runs(experiments, leave_outs, args.training_strategy)
    run_idx = build_run_index(expected, ckpts, logs)

    write_csv(ckpts, out_dir / "checkpoint_index.csv")
    write_csv(logs, out_dir / "log_index.csv")
    write_csv(expected, out_dir / "expected_runs.csv")
    write_csv(run_idx, out_dir / "run_index.csv")

    label_rows = []
    for e in experiments:
        label_rows.append({"exp_name": e, "leave_out": "NONE", "short_label": e, "display_group": "M_forward"})
    for lo in leave_outs:
        label_rows.append({"exp_name": "M7", "leave_out": lo, "short_label": LOO_LABELS.get(lo, lo), "removed_module": LOO_MODULES.get(lo, ""), "display_group": "L_leave_one_out"})
    write_csv(label_rows, out_dir / "run_label_map.csv")

    eval_cfg = {
        "created_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "python": sys.version,
        "eval_root": str(eval_root),
        "ckpt_dir": str(ckpt_dir),
        "log_dir": str(log_dir),
        "out_dir": str(out_dir),
        "experiments": experiments,
        "training_strategy": args.training_strategy,
        "leave_outs": leave_outs,
        "time_indices": time_indices,
        "S_IC_CO2": args.s_ic_co2,
        "Snr": args.Snr,
        "Sw_irr": args.Sw_irr,
        "saturation_note": "Raw Sco2 is not remapped for accuracy/snapshot evaluation; Snr enters only physics computations.",
        "figure_style_contract": figure_style_contract(),
    }
    safe_json_dump(eval_cfg, out_dir / "evaluation_config.json")
    safe_json_dump({"time_indices": time_indices, "semantics": {"t0": "first time layer", "early": "approximately 25% time index, avoiding t0 when possible", "middle": "approximately 50% time index", "late": "approximately 75% time index", "final": "last time layer"}}, out_dir / "time_selection.json")
    safe_json_dump({"S_vmin": 0.0, "S_vmax": 1.0 - float(args.Sw_irr), "S_error_symmetric": True, "S_error_recommended_abs": 0.25, "pressure_error_symmetric": True, "speckle_threshold_mode": "excess", "S_IC_CO2": args.s_ic_co2, "figure_style_contract": figure_style_contract()}, out_dir / "colormap_config.json")
    safe_json_dump(metric_definitions(args.s_ic_co2, args.Snr), out_dir / "metric_definitions.json")

    (out_dir / "directory_tree.txt").write_text(directory_tree(eval_root, max_depth=int(args.directory_tree_depth)), encoding="utf-8")
    readme = f"""# M0-M7 evaluation metadata\n\nCreated: {eval_cfg['created_at']}\n\nThis directory indexes checkpoints, logs, expected runs, labels, time-layer conventions, colour-map conventions, and metric definitions for the M0-M7 evaluation suite.\n\nSaturation convention: raw Sco2 is evaluated directly. S_IC_CO2={args.s_ic_co2}; Snr={args.Snr}. Snr is for relative permeability/PDE/FV physics, not for remapping raw Sco2 predictions.\n"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")

    print(f"[metadata] checkpoints: {len(ckpts)}")
    print(f"[metadata] logs:        {len(logs)}")
    print(f"[metadata] run index:   {len(run_idx)}")
    print(f"[metadata] output:      {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
