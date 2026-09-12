#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training stability/cost evaluator for M0-M7 and M7 leave-one-out PINN runs.

This script parses SLURM stdout/stderr/log files and exports:
  stability/summary_training_stability.csv
  stability/loss_points.csv
  stability/loss_components_summary.csv
  stability/ramp_stage_metrics.csv
  stability/loss_curves/*.png|pdf|tiff
  stability/memory_curves/*.png|pdf|tiff

It is designed for logs produced by the M-series training script, e.g.
  [M7|BASE] it=   200 [rnd] wPDE=0.00 wFV=0.00e+00 wFG=0.00e+00 beta= 1.00 eps=1.00e-05 loss=...
  ... | pde=... fv=... fvC=... fvW=... | dataP=... dataS=... Sg=... Sf=... Sp=... Sfg=... ...
  ... | injF=... injS=... outP=... outBF=... noflow=... icP=... icS=... | alloc=...GB ...

The script also supports older lines approximately matching [M0] it=... and [LBFGS-M0].

Notes on S_IC_CO2 and Snr:
  Training stability is log-based and does not recompute saturation physics. Therefore S_IC_CO2=0.0
  and Snr=0.0 do not enter the numerical stability metrics. They are still written to the config JSON
  so the stability output remains consistent with global/front/physics/snapshot evaluations.
"""
from __future__ import annotations

import os
import re
import csv
import json
import math
import argparse
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
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

try:
    from dateutil import parser as dtparser
except Exception:  # pragma: no cover
    dtparser = None

ALL_EXPS = tuple(f"M{i}" for i in range(8))
DEFAULT_FIGURE_DPI = 600
DEFAULT_EXPORT_SVG = True
EXPORT_SVG = DEFAULT_EXPORT_SVG
FIGURE_DPI_RUNTIME = DEFAULT_FIGURE_DPI
DEFAULT_LEAVE_OUTS = ("NONE", "PLAIN_TWONET", "FRONT_PLUME", "PAIRGRAD", "RAR", "FV", "COARSE_DETAIL")
LOO_LABEL = {
    "NONE": "L0",
    "PLAIN_TWONET": "L1",
    "FRONT_PLUME": "L2",
    "PAIRGRAD": "L3",
    "RAR": "L4",
    "FV": "L5",
    "COARSE_DETAIL": "L6",
    "MSFF": "L7",
    "WATER_FV": "LFVw",
}

# Main M-series Adam log pattern. This intentionally captures the rest of line for flexible key=value parsing.
RE_ADAM_M = re.compile(
    r"^\[(?P<exp>M\d+)(?:\|(?P<strategy>[A-Za-z0-9_\-]+))?\]\s+"
    r"it=\s*(?P<it>\d+)\s+\[(?P<mode>[^\]]+)\]\s+"
    r"(?P<rest>.*?)(?:\s*)$"
)
RE_LBFGS_M = re.compile(
    r"^\[LBFGS-(?P<exp>M\d+)(?:\|(?P<strategy>[A-Za-z0-9_\-]+))?\]\s+"
    r"call=\s*(?P<call>\d+)\s+loss=(?P<loss>[^\s]+)"
)
RE_KEYVAL = re.compile(r"(?P<key>[A-Za-z][A-Za-z0-9_]*)=(?P<val>[-+]?nan|[-+]?inf|[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")
RE_CKPT = re.compile(
    r"model_(?P<exp>M\d+)_(?P<strategy>[A-Za-z0-9_\-]+?)_LOO-(?P<loo>[A-Za-z0-9_\-]+)_seed(?P<seed>\d+)_(?P<profile>[^_\s/]+)_(?P<suffix>[^\s/]+)\.pt"
)
RE_RUNNING_EXP = re.compile(r"Running EXP_NAME\s*=\s*(M\d+)\b")
RE_EXP_LINE = re.compile(r"^\s*EXP(?:_NAME)?\s*[:=]\s*(M\d+)\s*$", re.MULTILINE)
RE_STRATEGY_LINE = re.compile(r"^\s*(?:TRAINING_STRATEGY|strategy)\s*[:=]\s*([A-Za-z0-9_\-]+)\s*$", re.IGNORECASE | re.MULTILINE)
RE_LEAVE_OUT_LINE = re.compile(r"^\s*(?:LEAVE_OUT|leave_out|Leave-out)\s*[:=]\s*([A-Za-z0-9_\-]+)\s*$", re.IGNORECASE | re.MULTILINE)
RE_SEED_LINE = re.compile(r"^\s*SEED\s*[:=]\s*(\d+)\s*$", re.IGNORECASE | re.MULTILINE)
RE_RUNTIME = re.compile(r"\[Runtime\].*?EXP=(M\d+).*?strategy=([^\s|]+).*?profile=([^\s|]+)")
RE_JOBID = re.compile(r"^\s*JobID:\s*(\S+)\s*$", re.MULTILINE)
RE_HOST = re.compile(r"^\s*Host:\s*(.+?)\s*$", re.MULTILINE)
RE_START = re.compile(r"^\s*Start:\s*(.+?)\s*$", re.MULTILINE)
RE_END = re.compile(r"^\s*End:\s*(.+?)\s*$", re.MULTILINE)
RE_CKPT_SAVED = re.compile(r"^\s*\[checkpoint\]\s+saved:\s+(.+?)\s*$", re.MULTILINE)
RE_GPU_ALLOC = re.compile(r"alloc=([0-9.+\-eE]+)GB")
RE_GPU_RESERV = re.compile(r"reserv=([0-9.+\-eE]+)GB")
RE_GPU_MAX_ALLOC = re.compile(r"max_alloc=([0-9.+\-eE]+)GB")
RE_GPU_MAX_RESERV = re.compile(r"max_reserv=([0-9.+\-eE]+)GB")
RE_TOTAL_PARAMS = re.compile(r"total_params=(\d+)")
RE_TRAINABLE_PARAMS = re.compile(r"trainable_params=(\d+)")

COMPONENT_KEYS = (
    "pde", "fv", "fvC", "fvW", "dataP", "dataS", "Sg", "Sf", "Sp", "Sfg",
    "injF", "injS", "outP", "outBF", "noflow", "icP", "icS",
)
SCHEDULE_KEYS = ("wPDE", "wFV", "wFG", "beta", "eps")
MEMORY_KEYS = ("alloc", "reserv", "max_alloc", "max_reserv")

@dataclass
class LossPoint:
    kind: str
    x: int
    loss: float
    mode: str = ""
    exp: Optional[str] = None
    strategy: Optional[str] = None
    values: dict[str, float] = field(default_factory=dict)
    raw_line: str = ""

@dataclass
class RunRecord:
    run_id: str
    stem: str
    out_path: Optional[str] = None
    err_path: Optional[str] = None
    exp: Optional[str] = None
    strategy: Optional[str] = None
    leave_out: Optional[str] = None
    short_label: Optional[str] = None
    seed: Optional[int] = None
    runtime_profile: Optional[str] = None
    job_id: Optional[str] = None
    host: Optional[str] = None
    start_text: Optional[str] = None
    end_text: Optional[str] = None
    wall_time_hours: Optional[float] = None
    total_params: Optional[int] = None
    trainable_params: Optional[int] = None
    ckpt_saved_paths: list[str] = field(default_factory=list)
    points: list[LossPoint] = field(default_factory=list)
    has_nan: bool = False
    has_inf: bool = False
    has_traceback: bool = False
    has_error: bool = False
    crashed: bool = False
    max_gpu_alloc_gb: Optional[float] = None
    max_gpu_reserv_gb: Optional[float] = None
    max_gpu_max_alloc_gb: Optional[float] = None
    max_gpu_max_reserv_gb: Optional[float] = None
    final_logged_loss: Optional[float] = None
    best_logged_loss: Optional[float] = None
    best_logged_x: Optional[int] = None
    final_adam_loss: Optional[float] = None
    final_adam_iter: Optional[int] = None
    best_adam_loss: Optional[float] = None
    best_adam_iter: Optional[int] = None
    final_lbfgs_loss: Optional[float] = None
    final_lbfgs_call: Optional[int] = None
    n_loss_points: int = 0
    n_adam_points: int = 0
    n_lbfgs_points: int = 0
    oscillation_std_log10_tail: Optional[float] = None
    oscillation_mad_diff_log10_tail: Optional[float] = None
    threshold_iters: dict[str, Optional[int]] = field(default_factory=dict)
    parse_notes: list[str] = field(default_factory=list)

    def to_summary_row(self, threshold_labels: list[str]) -> dict[str, Any]:
        row = dict(
            run_id=self.run_id,
            short_label=self.short_label,
            stem=self.stem,
            exp_name=self.exp,
            training_strategy=self.strategy,
            leave_out=self.leave_out,
            seed=self.seed,
            runtime_profile=self.runtime_profile,
            job_id=self.job_id,
            host=self.host,
            start_text=self.start_text,
            end_text=self.end_text,
            wall_time_hours=self.wall_time_hours,
            total_params=self.total_params,
            trainable_params=self.trainable_params,
            n_loss_points=self.n_loss_points,
            n_adam_points=self.n_adam_points,
            n_lbfgs_points=self.n_lbfgs_points,
            final_logged_loss=self.final_logged_loss,
            best_logged_loss=self.best_logged_loss,
            best_logged_x=self.best_logged_x,
            final_adam_loss=self.final_adam_loss,
            final_adam_iter=self.final_adam_iter,
            best_adam_loss=self.best_adam_loss,
            best_adam_iter=self.best_adam_iter,
            final_lbfgs_loss=self.final_lbfgs_loss,
            final_lbfgs_call=self.final_lbfgs_call,
            max_gpu_alloc_gb=self.max_gpu_alloc_gb,
            max_gpu_reserv_gb=self.max_gpu_reserv_gb,
            max_gpu_max_alloc_gb=self.max_gpu_max_alloc_gb,
            max_gpu_max_reserv_gb=self.max_gpu_max_reserv_gb,
            oscillation_std_log10_tail=self.oscillation_std_log10_tail,
            oscillation_mad_diff_log10_tail=self.oscillation_mad_diff_log10_tail,
            has_nan=int(self.has_nan),
            has_inf=int(self.has_inf),
            has_traceback=int(self.has_traceback),
            has_error=int(self.has_error),
            crashed=int(self.crashed),
            parse_notes=" | ".join(self.parse_notes),
            out_path=self.out_path,
            err_path=self.err_path,
        )
        for label in threshold_labels:
            row[f"iter_to_{label}"] = self.threshold_iters.get(label)
        return row


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def parse_float(x: str) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def finite_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except Exception:
        return None
    return v if math.isfinite(v) else None


def max_merge(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if b is None:
        return a
    if a is None:
        return b
    return max(a, b)


def parse_keyvals(rest: str) -> dict[str, float]:
    out = {}
    for m in RE_KEYVAL.finditer(rest):
        k = m.group("key")
        v = parse_float(m.group("val"))
        if v is not None:
            out[k] = v
    return out


def read_text(path: Optional[str]) -> str:
    if path is None or not os.path.isfile(path):
        return ""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


_MONTH_ABBR = {m: i for i, m in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], start=1)}
# SLURM `date` stamp, e.g. "Wed  3 Jun 03:27:11 BST 2026" (weekday + tz are ignored).
_RE_SLURM_DATE = re.compile(r"(\d{1,2})\s+([A-Za-z]{3})\s+(\d{1,2}):(\d{2}):(\d{2}).*?(\d{4})")


def parse_datetime_text(s: Optional[str]):
    if not s:
        return None
    if dtparser is not None:
        try:
            return dtparser.parse(s, fuzzy=True)
        except Exception:
            pass
    # dateutil-free fallback. The tz token (BST/GMT) defeats strptime and, when
    # python-dateutil is absent, dtparser is None — that is why wall-time came back
    # blank. Pull the numeric fields directly instead.
    m = _RE_SLURM_DATE.search(s)
    if not m:
        return None
    day, mon, hh, mm, ss, yr = m.groups()
    mon_i = _MONTH_ABBR.get(mon[:3].title())
    if not mon_i:
        return None
    import datetime as _dt
    try:
        return _dt.datetime(int(yr), mon_i, int(day), int(hh), int(mm), int(ss))
    except Exception:
        return None


def wall_hours(start_text: Optional[str], end_text: Optional[str]) -> Optional[float]:
    a = parse_datetime_text(start_text)
    b = parse_datetime_text(end_text)
    if a is None or b is None:
        return None
    try:
        h = (b - a).total_seconds() / 3600.0
        return float(h) if h >= 0 else None
    except Exception:
        return None


def threshold_label(v: float) -> str:
    return f"loss_le_{v:.1e}".replace(".", "p").replace("-", "m").replace("+", "")


def parse_thresholds(s: str) -> list[float]:
    vals = []
    for part in str(s).split(","):
        part = part.strip()
        if part:
            vals.append(float(part))
    return vals


def csv_write(path: str, rows: list[dict[str, Any]], fieldnames: Optional[list[str]] = None):
    ensure_dir(os.path.dirname(path))
    if fieldnames is None:
        keys = []
        seen = set()
        for row in rows:
            for k in row.keys():
                if k not in seen:
                    keys.append(k)
                    seen.add(k)
        fieldnames = keys
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def collect_log_groups(log_dir: str, recursive: bool = True) -> list[tuple[str, Optional[str], Optional[str]]]:
    files = []
    if recursive:
        for root, _, fns in os.walk(log_dir):
            for fn in fns:
                files.append(os.path.join(root, fn))
    else:
        files = [os.path.join(log_dir, fn) for fn in os.listdir(log_dir)]
    groups: dict[str, dict[str, str]] = {}
    for full in files:
        if not os.path.isfile(full):
            continue
        low = os.path.basename(full).lower()
        if low.endswith(".out"):
            stem = os.path.splitext(full)[0]
            groups.setdefault(stem, {})["out"] = full
        elif low.endswith(".err"):
            stem = os.path.splitext(full)[0]
            groups.setdefault(stem, {})["err"] = full
        elif low.endswith((".log", ".txt")):
            stem = os.path.splitext(full)[0]
            groups.setdefault(stem, {})["out"] = full
    out = []
    for stem, d in sorted(groups.items()):
        out.append((os.path.basename(stem), d.get("out"), d.get("err")))
    return out


def infer_metadata(text: str, stem: str) -> tuple[Optional[str], Optional[str], Optional[str], Optional[int], Optional[str], list[str]]:
    notes = []
    exp = strategy = loo = profile = None
    seed = None
    ckpts = list(RE_CKPT.finditer(text))
    if ckpts:
        m = ckpts[-1]
        exp = m.group("exp")
        strategy = m.group("strategy")
        loo = m.group("loo")
        seed = int(m.group("seed"))
        profile = m.group("profile")
        notes.append("metadata from checkpoint filename")
    m = RE_RUNTIME.search(text)
    if m:
        exp = exp or m.group(1)
        strategy = strategy or m.group(2)
        profile = profile or m.group(3)
        notes.append("metadata from [Runtime]")
    if exp is None:
        m = RE_RUNNING_EXP.search(text) or RE_EXP_LINE.search(text)
        if m:
            exp = m.group(1)
            notes.append("exp from log echo")
    if strategy is None:
        m = RE_STRATEGY_LINE.search(text)
        if m:
            strategy = m.group(1)
            notes.append("strategy from log echo")
    if loo is None:
        m = RE_LEAVE_OUT_LINE.search(text)
        if m:
            loo = m.group(1)
            notes.append("leave_out from log echo")
    if seed is None:
        m = RE_SEED_LINE.search(text)
        if m:
            seed = int(m.group(1))
            notes.append("seed from log echo")
    # filename fallback
    stem_text = os.path.basename(stem)
    m = RE_CKPT.search(stem_text)
    if m:
        exp = exp or m.group("exp")
        strategy = strategy or m.group("strategy")
        loo = loo or m.group("loo")
        seed = seed if seed is not None else int(m.group("seed"))
        profile = profile or m.group("profile")
        notes.append("metadata from log filename")
    return exp, strategy, loo, seed, profile, notes


def short_label(exp: Optional[str], leave_out: Optional[str], loo_labels: bool) -> str:
    if loo_labels and exp == "M7" and leave_out:
        return LOO_LABEL.get(leave_out.upper(), f"LOO-{leave_out}")
    return exp or "unknown"


def parse_run(stem: str, out_path: Optional[str], err_path: Optional[str], thresholds: list[float], tail_points: int, loo_labels: bool) -> RunRecord:
    text = read_text(out_path) + "\n" + read_text(err_path)
    exp, strategy, loo, seed, profile, notes = infer_metadata(text, stem)
    if loo is None:
        loo = "NONE" if (exp == "M7" and "LOO-" in text) else None
    run_id_parts = [exp or "unknown", strategy or "NA"]
    if loo is not None:
        run_id_parts.append(f"LOO-{loo}")
    if seed is not None:
        run_id_parts.append(f"seed{seed}")
    rec = RunRecord(run_id="_".join(run_id_parts), stem=stem, out_path=out_path, err_path=err_path)
    rec.exp, rec.strategy, rec.leave_out, rec.seed, rec.runtime_profile = exp, strategy, loo, seed, profile
    rec.short_label = short_label(exp, loo, loo_labels)
    rec.parse_notes.extend(notes)
    rec.job_id = (RE_JOBID.search(text).group(1) if RE_JOBID.search(text) else None)
    rec.host = (RE_HOST.search(text).group(1) if RE_HOST.search(text) else None)
    rec.start_text = (RE_START.search(text).group(1) if RE_START.search(text) else None)
    rec.end_text = (RE_END.search(text).group(1) if RE_END.search(text) else None)
    rec.wall_time_hours = wall_hours(rec.start_text, rec.end_text)
    _mtp = RE_TOTAL_PARAMS.search(text)
    rec.total_params = int(_mtp.group(1)) if _mtp else None
    _mtr = RE_TRAINABLE_PARAMS.search(text)
    rec.trainable_params = int(_mtr.group(1)) if _mtr else None
    rec.ckpt_saved_paths = [m.group(1).strip() for m in RE_CKPT_SAVED.finditer(text)]
    # general failure flags
    low = text.lower()
    rec.has_nan = bool(re.search(r"\bnan\b", low))
    rec.has_inf = bool(re.search(r"\binf\b", low))
    rec.has_traceback = "traceback (most recent call last)" in low
    rec.has_error = bool(re.search(r"\berror\b|runtimeerror|nameerror|valueerror|cuda out of memory|oom", low))
    # If End exists and no traceback, do not mark crashed. Otherwise infer conservatively.
    rec.crashed = bool(rec.has_traceback or (rec.has_error and rec.end_text is None))

    for line in text.splitlines():
        m = RE_ADAM_M.match(line.strip())
        if m:
            vals = parse_keyvals(m.group("rest"))
            loss = vals.get("loss")
            if loss is None or not math.isfinite(loss):
                continue
            p = LossPoint(
                kind="adam",
                x=int(m.group("it")),
                loss=float(loss),
                mode=m.group("mode"),
                exp=m.group("exp"),
                strategy=m.group("strategy"),
                values=vals,
                raw_line=line,
            )
            rec.points.append(p)
            # memory
            rec.max_gpu_alloc_gb = max_merge(rec.max_gpu_alloc_gb, vals.get("alloc"))
            rec.max_gpu_reserv_gb = max_merge(rec.max_gpu_reserv_gb, vals.get("reserv"))
            rec.max_gpu_max_alloc_gb = max_merge(rec.max_gpu_max_alloc_gb, vals.get("max_alloc"))
            rec.max_gpu_max_reserv_gb = max_merge(rec.max_gpu_max_reserv_gb, vals.get("max_reserv"))
            continue
        m = RE_LBFGS_M.match(line.strip())
        if m:
            loss = parse_float(m.group("loss"))
            if loss is None or not math.isfinite(loss):
                continue
            rec.points.append(LossPoint(kind="lbfgs", x=int(m.group("call")), loss=float(loss), exp=m.group("exp"), strategy=m.group("strategy"), raw_line=line))

    rec.n_loss_points = len(rec.points)
    adam = [p for p in rec.points if p.kind == "adam"]
    lbfgs = [p for p in rec.points if p.kind == "lbfgs"]
    rec.n_adam_points = len(adam)
    rec.n_lbfgs_points = len(lbfgs)
    all_finite = [p for p in rec.points if math.isfinite(p.loss)]
    if all_finite:
        last = all_finite[-1]
        best = min(all_finite, key=lambda p: p.loss)
        rec.final_logged_loss = float(last.loss)
        rec.best_logged_loss = float(best.loss)
        rec.best_logged_x = int(best.x)
    if adam:
        last = adam[-1]
        best = min(adam, key=lambda p: p.loss)
        rec.final_adam_loss = float(last.loss)
        rec.final_adam_iter = int(last.x)
        rec.best_adam_loss = float(best.loss)
        rec.best_adam_iter = int(best.x)
    if lbfgs:
        last = lbfgs[-1]
        rec.final_lbfgs_loss = float(last.loss)
        rec.final_lbfgs_call = int(last.x)
    # oscillation tail on adam if available else all
    tail_src = adam if adam else all_finite
    if tail_src:
        vals = np.asarray([p.loss for p in tail_src[-max(3, tail_points):]], dtype=float)
        vals = vals[np.isfinite(vals) & (vals > 0)]
        if vals.size >= 3:
            logv = np.log10(vals)
            rec.oscillation_std_log10_tail = float(np.std(logv))
            rec.oscillation_mad_diff_log10_tail = float(np.median(np.abs(np.diff(logv)))) if logv.size >= 2 else None
    # threshold iteration, Adam only preferred
    thresh_src = adam if adam else all_finite
    for t in thresholds:
        lab = threshold_label(t)
        hit = None
        for p in thresh_src:
            if math.isfinite(p.loss) and p.loss <= t:
                hit = int(p.x)
                break
        rec.threshold_iters[lab] = hit
    return rec


def component_summary_rows(records: list[RunRecord]) -> list[dict[str, Any]]:
    rows = []
    for r in records:
        adam = [p for p in r.points if p.kind == "adam"]
        row = dict(run_id=r.run_id, short_label=r.short_label, exp_name=r.exp, training_strategy=r.strategy, leave_out=r.leave_out, seed=r.seed)
        if adam:
            last = adam[-1]
            for k in COMPONENT_KEYS + SCHEDULE_KEYS:
                row[f"final_{k}"] = last.values.get(k)
            # minima/means for key losses
            for k in ("pde", "fv", "fvC", "fvW", "dataS", "Sfg", "Sf", "Sp", "outBF", "icS"):
                arr = np.asarray([p.values.get(k, np.nan) for p in adam], dtype=float)
                arr = arr[np.isfinite(arr)]
                row[f"best_{k}"] = float(np.min(arr)) if arr.size else None
                row[f"tail_mean_{k}"] = float(np.mean(arr[-20:])) if arr.size else None
        rows.append(row)
    return rows


def ramp_rows(records: list[RunRecord], window: int = 3) -> list[dict[str, Any]]:
    rows = []
    for r in records:
        adam = [p for p in r.points if p.kind == "adam"]
        row = dict(run_id=r.run_id, short_label=r.short_label, exp_name=r.exp, training_strategy=r.strategy, leave_out=r.leave_out, seed=r.seed)
        for sched_key, name in (("wPDE", "PDE"), ("wFG", "PAIRGRAD"), ("wFV", "FV")):
            idx = None
            for i, p in enumerate(adam):
                if p.values.get(sched_key, 0.0) and p.values.get(sched_key, 0.0) > 0.0:
                    idx = i
                    break
            if idx is None:
                row[f"{name}_first_iter"] = None
                row[f"{name}_loss_jump_ratio"] = None
                row[f"{name}_loss_jump_log10"] = None
                continue
            before = [p.loss for p in adam[max(0, idx - window):idx] if p.loss > 0]
            after = [p.loss for p in adam[idx:min(len(adam), idx + window)] if p.loss > 0]
            ratio = None
            if before and after:
                ratio = float(np.mean(after) / max(np.mean(before), 1e-300))
            row[f"{name}_first_iter"] = int(adam[idx].x)
            row[f"{name}_loss_jump_ratio"] = ratio
            row[f"{name}_loss_jump_log10"] = float(np.log10(ratio)) if ratio and ratio > 0 else None
        rows.append(row)
    return rows


def loss_point_rows(records: list[RunRecord]) -> list[dict[str, Any]]:
    rows = []
    for r in records:
        for p in r.points:
            row = dict(run_id=r.run_id, short_label=r.short_label, exp_name=r.exp, training_strategy=r.strategy, leave_out=r.leave_out, seed=r.seed, kind=p.kind, x=p.x, mode=p.mode, loss=p.loss)
            for k in SCHEDULE_KEYS + COMPONENT_KEYS + MEMORY_KEYS + ("nFG", "dFG"):
                row[k] = p.values.get(k)
            rows.append(row)
    return rows


def aggregate_seed_rows(records: list[RunRecord]) -> list[dict[str, Any]]:
    groups: dict[tuple, list[RunRecord]] = {}
    for r in records:
        groups.setdefault((r.exp, r.strategy, r.leave_out, r.short_label), []).append(r)
    fields = ["final_logged_loss", "best_logged_loss", "wall_time_hours", "max_gpu_max_alloc_gb", "oscillation_std_log10_tail"]
    out = []
    for (exp, strat, loo, lab), rs in sorted(groups.items(), key=lambda kv: str(kv[0])):
        row = dict(exp_name=exp, training_strategy=strat, leave_out=loo, short_label=lab, n_runs=len(rs))
        for f in fields:
            arr = np.asarray([getattr(r, f) if getattr(r, f) is not None else np.nan for r in rs], dtype=float)
            row[f"{f}_mean"] = float(np.nanmean(arr)) if np.any(np.isfinite(arr)) else None
            row[f"{f}_std"] = float(np.nanstd(arr)) if np.any(np.isfinite(arr)) else None
        row["n_crashed"] = int(sum(r.crashed for r in rs))
        row["n_nan"] = int(sum(r.has_nan for r in rs))
        out.append(row)
    return out


def set_pub_style():
    setup_nature_rcparams(font_size=7.4, line_width=1.25)


def savefig(fig, path_base: str, export_pdf: bool, export_tiff: bool):
    ensure_dir(os.path.dirname(path_base))
    save_pub_figure(fig, path_base, dpi=FIGURE_DPI_RUNTIME, export_pdf=export_pdf, export_tiff=export_tiff, export_svg=EXPORT_SVG, export_png=True, pad_inches=0.035)
    plt.close(fig)


def sort_records(records: list[RunRecord]) -> list[RunRecord]:
    def key(r: RunRecord):
        lab = r.short_label or ""
        if re.match(r"M\d+", lab):
            return (0, int(lab[1:]), r.seed if r.seed is not None else -1)
        if re.match(r"L\d+", lab):
            return (1, int(lab[1:]), r.seed if r.seed is not None else -1)
        return (2, lab, r.seed if r.seed is not None else -1)
    return sorted(records, key=key)


def _job_id_int(r: RunRecord) -> int:
    try:
        return int(str(r.job_id))
    except Exception:
        return -1


def deduplicate_latest_records(records: list[RunRecord]) -> list[RunRecord]:
    """Keep one canonical log per run_id.

    Multiple Slurm attempts can write the same checkpoint filename, especially
    for repeated seed0 runs.  The paper-facing stability table should align with
    one checkpoint per run_id, so prefer the latest completed non-crashed log.
    """
    groups: dict[str, list[RunRecord]] = {}
    for r in records:
        groups.setdefault(r.run_id, []).append(r)

    out: list[RunRecord] = []
    for _, group in groups.items():
        best = sorted(
            group,
            key=lambda r: (
                int(not r.crashed),
                int(bool(r.end_text)),
                int(bool(r.ckpt_saved_paths)),
                int(r.n_loss_points),
                _job_id_int(r),
            ),
            reverse=True,
        )[0]
        if len(group) > 1:
            best.parse_notes.append(
                "deduplicated repeated logs for same run_id; kept latest completed checkpointed attempt"
            )
        out.append(best)
    return out


def plot_loss_curves(records: list[RunRecord], out_dir: str, export_pdf: bool, export_tiff: bool):
    set_pub_style()
    ensure_dir(out_dir)
    records = sort_records(records)
    # Individual curves
    for r in records:
        adam = [p for p in r.points if p.kind == "adam"]
        if not adam:
            continue
        x = np.asarray([p.x for p in adam])
        y = np.asarray([p.loss for p in adam])
        fig, ax = plt.subplots(figsize=(3.6, 2.4))
        ax.plot(x, y, label=r.short_label or r.run_id, color=color_for_label(r.short_label or r.run_id, None))
        ax.set_yscale("log")
        ax.set_xlabel("Adam iteration")
        ax.set_ylabel("Total training loss")
        ax.set_title(r.short_label or r.run_id)
        apply_nature_axis(ax, grid=True, grid_axis="both")
        ax.legend(frameon=False, loc="best")
        fig.subplots_adjust(left=0.16, right=0.98, bottom=0.20, top=0.88)
        savefig(fig, os.path.join(out_dir, f"total_loss_curve_{r.short_label or r.run_id}"), export_pdf, export_tiff)
        # Components if present
        comp_keys = [k for k in ("pde", "dataS", "Sfg", "fv", "outBF", "icS") if any(k in p.values for p in adam)]
        if comp_keys:
            fig, ax = plt.subplots(figsize=(4.1, 2.7))
            for k in comp_keys:
                yy = np.asarray([p.values.get(k, np.nan) for p in adam], dtype=float)
                if np.any(np.isfinite(yy)):
                    ax.plot(x, yy, label=k)
            ax.set_yscale("log")
            ax.set_xlabel("Adam iteration")
            ax.set_ylabel("Loss component")
            ax.set_title(r.short_label or r.run_id)
            apply_nature_axis(ax, grid=True, grid_axis="both")
            ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.24), handlelength=1.6, columnspacing=1.1)
            fig.subplots_adjust(left=0.16, right=0.98, bottom=0.34, top=0.88)
            savefig(fig, os.path.join(out_dir, f"loss_components_{r.short_label or r.run_id}"), export_pdf, export_tiff)
    # Combined M and L overviews separated
    for prefix, title in (("M", "M0-M7 training loss"), ("L", "L0-L6 leave-one-out training loss")):
        subset = [r for r in records if (r.short_label or "").startswith(prefix)]
        if not subset:
            continue
        fig, ax = plt.subplots(figsize=(4.4, 2.8))
        for r in subset:
            adam = [p for p in r.points if p.kind == "adam"]
            if not adam:
                continue
            ax.plot([p.x for p in adam], [p.loss for p in adam], label=r.short_label, color=color_for_label(r.short_label, None))
        ax.set_yscale("log")
        ax.set_xlabel("Adam iteration")
        ax.set_ylabel("Total training loss")
        ax.set_title(title)
        apply_nature_axis(ax, grid=True, grid_axis="both")
        ax.legend(frameon=False, ncol=min(5, max(1, len(subset))), loc="upper center", bbox_to_anchor=(0.5, -0.24), handlelength=1.6, columnspacing=1.0)
        fig.subplots_adjust(left=0.14, right=0.98, bottom=0.35, top=0.88)
        savefig(fig, os.path.join(out_dir, f"combined_{prefix}_total_loss"), export_pdf, export_tiff)


def plot_memory_curves(records: list[RunRecord], out_dir: str, export_pdf: bool, export_tiff: bool):
    set_pub_style()
    ensure_dir(out_dir)
    for prefix, title in (("M", "M0-M7 GPU memory"), ("L", "L0-L6 GPU memory")):
        subset = [r for r in sort_records(records) if (r.short_label or "").startswith(prefix)]
        if not subset:
            continue
        fig, ax = plt.subplots(figsize=(4.4, 2.8))
        any_curve = False
        for r in subset:
            adam = [p for p in r.points if p.kind == "adam"]
            x = [p.x for p in adam if p.values.get("max_alloc") is not None]
            y = [p.values.get("max_alloc") for p in adam if p.values.get("max_alloc") is not None]
            if x and y:
                any_curve = True
                ax.plot(x, y, label=r.short_label, color=color_for_label(r.short_label, None))
        if not any_curve:
            plt.close(fig)
            continue
        ax.set_xlabel("Adam iteration")
        ax.set_ylabel("Max allocated GPU memory (GB)")
        ax.set_title(title)
        apply_nature_axis(ax, grid=True, grid_axis="both")
        ax.legend(frameon=False, ncol=min(5, max(1, len(subset))), loc="upper center", bbox_to_anchor=(0.5, -0.24), handlelength=1.6, columnspacing=1.0)
        fig.subplots_adjust(left=0.16, right=0.98, bottom=0.35, top=0.88)
        savefig(fig, os.path.join(out_dir, f"combined_{prefix}_gpu_memory"), export_pdf, export_tiff)


def plot_summary_panels(records: list[RunRecord], out_dir: str, export_pdf: bool, export_tiff: bool):
    set_pub_style()
    ensure_dir(out_dir)
    for prefix, title in (("M", "M0-M7 training stability"), ("L", "L0-L6 leave-one-out stability")):
        subset = [r for r in sort_records(records) if (r.short_label or "").startswith(prefix)]
        if not subset:
            continue
        labels = [r.short_label for r in subset]
        final_loss = np.asarray([r.final_logged_loss if r.final_logged_loss is not None else np.nan for r in subset], dtype=float)
        wall = np.asarray([r.wall_time_hours if r.wall_time_hours is not None else np.nan for r in subset], dtype=float)
        osc = np.asarray([r.oscillation_std_log10_tail if r.oscillation_std_log10_tail is not None else np.nan for r in subset], dtype=float)
        mem = np.asarray([r.max_gpu_max_alloc_gb if r.max_gpu_max_alloc_gb is not None else np.nan for r in subset], dtype=float)
        fig, axes = plt.subplots(2, 2, figsize=(6.0, 4.2))
        specs = [
            (axes[0,0], final_loss, "Final logged loss", "log"),
            (axes[0,1], wall, "Wall time (h)", "linear"),
            (axes[1,0], osc, "Tail oscillation, std(log10 loss)", "linear"),
            (axes[1,1], mem, "Max GPU allocation (GB)", "linear"),
        ]
        for ax, vals, ylabel, scale in specs:
            ax.bar(np.arange(len(labels)), vals, width=0.72, color=[color_for_label(lab, "#7F7F7F") for lab in labels])
            ax.set_xticks(np.arange(len(labels)))
            ax.set_xticklabels(labels, rotation=0)
            ax.set_ylabel(ylabel)
            if scale == "log":
                ax.set_yscale("log")
            apply_nature_axis(ax, grid=True, grid_axis="y")
        fig.suptitle(title, y=0.985)
        fig.subplots_adjust(left=0.12, right=0.98, bottom=0.10, top=0.90, wspace=0.34, hspace=0.42)
        savefig(fig, os.path.join(out_dir, f"{prefix}_stability_summary_panel"), export_pdf, export_tiff)


def filter_records(records: list[RunRecord], experiments: set[str], strategy: str, include_loo: bool, leave_outs: set[str], seed: str = "ANY") -> list[RunRecord]:
    out = []
    seed = str(seed).upper()
    for r in records:
        if seed != "ANY" and str(r.seed) != seed:
            continue
        if r.exp not in experiments and not (include_loo and r.exp == "M7"):
            continue
        if r.strategy and strategy and r.strategy.upper() != strategy.upper():
            continue
        leave = (r.leave_out or "").upper()
        is_main_run = leave in ("", "NONE", "NA")
        if is_main_run:
            if r.exp in experiments:
                out.append(r)
        else:
            if include_loo and r.exp == "M7" and leave in leave_outs:
                out.append(r)
    return sort_records(deduplicate_latest_records(out))


def main():
    ap = argparse.ArgumentParser(description="Parse M0-M7 training logs for stability/cost metrics.")
    ap.add_argument("--log-dir", required=True, help="Directory containing .out/.err/.log training logs.")
    ap.add_argument("--out-root", required=True, help="Output directory, usually eval_M0_M7/stability.")
    ap.add_argument("--experiments", default="M0,M1,M2,M3,M4,M5,M6,M7")
    ap.add_argument("--training-strategy", default="BASE")
    ap.add_argument("--seed", default="ANY", help="Training seed to include; ANY keeps all parsed seeds.")
    ap.add_argument("--include-loo", type=int, default=1)
    ap.add_argument("--leave-outs", default="NONE,PLAIN_TWONET,FRONT_PLUME,PAIRGRAD,RAR,FV,COARSE_DETAIL")
    ap.add_argument("--recursive", type=int, default=1)
    ap.add_argument("--loss-thresholds", default="1e-2,1e-3,1e-4")
    ap.add_argument("--tail-points", type=int, default=20)
    ap.add_argument("--ramp-window", type=int, default=3)
    ap.add_argument("--figure-dpi", type=int, default=DEFAULT_FIGURE_DPI)
    ap.add_argument("--export-pdf", type=int, default=1)
    ap.add_argument("--export-tiff", type=int, default=1)
    ap.add_argument("--export-svg", type=int, default=int(DEFAULT_EXPORT_SVG),
                    help="Export editable SVG files in addition to PNG/PDF/TIFF.")
    ap.add_argument("--loo-labels", type=int, default=1)
    # Consistency-only metadata arguments. Stability metrics do not use time slices or saturation physics.
    ap.add_argument("--time-indices", default="t0,early,middle,late,final")
    ap.add_argument("--s-ic-co2", type=float, default=0.0)
    ap.add_argument("--Snr", type=float, default=0.0)
    args = ap.parse_args()

    global EXPORT_SVG, FIGURE_DPI_RUNTIME
    EXPORT_SVG = bool(args.export_svg)
    FIGURE_DPI_RUNTIME = int(args.figure_dpi)
    plt.rcParams["savefig.dpi"] = FIGURE_DPI_RUNTIME
    out_root = args.out_root
    ensure_dir(out_root)
    cfg = vars(args).copy()
    with open(os.path.join(out_root, "stability_evaluation_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    thresholds = parse_thresholds(args.loss_thresholds)
    threshold_labels = [threshold_label(v) for v in thresholds]
    groups = collect_log_groups(args.log_dir, recursive=bool(args.recursive))
    records = [parse_run(stem, outp, errp, thresholds, args.tail_points, bool(args.loo_labels)) for stem, outp, errp in groups]

    experiments = {x.strip() for x in args.experiments.split(",") if x.strip()}
    leave_outs = {x.strip().upper() for x in args.leave_outs.split(",") if x.strip()}
    selected = filter_records(records, experiments, args.training_strategy, bool(args.include_loo), leave_outs, args.seed)

    # If filtering found nothing, write all parsed rows for debugging.
    if not selected:
        selected = sort_records(records)
        for r in selected:
            r.parse_notes.append("No records matched filter; included for debugging")

    summary_rows = [r.to_summary_row(threshold_labels) for r in selected]
    csv_write(os.path.join(out_root, "summary_training_stability.csv"), summary_rows)
    csv_write(os.path.join(out_root, "loss_points.csv"), loss_point_rows(selected))
    csv_write(os.path.join(out_root, "loss_components_summary.csv"), component_summary_rows(selected))
    csv_write(os.path.join(out_root, "ramp_stage_metrics.csv"), ramp_rows(selected, window=args.ramp_window))
    csv_write(os.path.join(out_root, "seed_aggregate_training_stability.csv"), aggregate_seed_rows(selected))

    plot_loss_curves(selected, os.path.join(out_root, "loss_curves"), bool(args.export_pdf), bool(args.export_tiff))
    plot_memory_curves(selected, os.path.join(out_root, "memory_curves"), bool(args.export_pdf), bool(args.export_tiff))
    plot_summary_panels(selected, os.path.join(out_root, "figures"), bool(args.export_pdf), bool(args.export_tiff))

    print(f"[done] parsed logs={len(records)} selected={len(selected)}")
    print(f"[done] output: {out_root}")


if __name__ == "__main__":
    main()
