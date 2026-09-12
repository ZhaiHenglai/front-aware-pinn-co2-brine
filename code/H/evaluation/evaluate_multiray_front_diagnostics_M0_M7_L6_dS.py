#!/usr/bin/env python3
"""
Multi-ray front diagnostics for the M0-M7 PINN suite.

Purpose
-------
This script generalizes the single injector-to-producer diagonal front diagnostic
to multiple rays. It separates four physically distinct quantities:

  1. W_front_ratio_1090
     10-90% transition-band width ratio, pred / true.

  2. front_inverse_slope_width_ratio
     Legacy inverse-slope width ratio:
       (dS_jump / max|dS/ds|)_pred / (dS_jump / max|dS/ds|)_true.
     Values > 1 indicate a smoother predicted front core.

  3. front_sharpness_ratio
     Maximum-gradient ratio:
       max|dS/ds|_pred / max|dS/ds|_true.
     Values closer to 1 indicate sharper local front gradients.

  4. front_position_abs_error_mid
     Absolute front-position error at the 50% level of the local saturation jump.

Outputs
-------
  multiray_front_diagnostics.csv
      One row per run, time, and ray.

  multiray_front_profiles.csv
      Profile samples used to compute each diagnostic.

  multiray_front_summary.csv
      Run-level mean/std over valid time-ray front profiles.

  multiray_front_seed_aggregate.csv
      Mean/std across training seeds after run-level summarization.

  multiray_front_by_time.csv
      Run-time-level mean/std over valid rays.

  multiray_front_by_ray.csv
      Run-ray-level mean/std over valid times.

  multiray_front_config.json
      Full run configuration and ray geometry.

Definition of valid_front
-------------------------
A ray/time profile is included in summary statistics only when the reference
profile contains a resolvable front:

  - true dS_jump >= --min-ds-jump
  - true thickness_1090 is finite and positive
  - true front_position_mid is finite

Invalid rows are still written to multiray_front_diagnostics.csv for auditability.

Geometry note
-------------
When the injector is located at a physical corner, the sampled data cloud may
start at small positive coordinates rather than exactly zero. In that case, the
near-side rectangle intersection is only the ray entry point and is not a valid
front profile endpoint. This evaluator therefore uses the far-side intersection
along each ray and performs a ray-length quality-control check before inference.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from typing import Any

import numpy as np
import torch

import evaluate_global_accuracy_M0_M7_L6_dS_shortlabels as GA
import evaluate_heldout_and_front_timing as HT
import heldout_split as HS


def _pair(s: str) -> tuple[float, float]:
    parts = [float(v) for v in str(s).split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Expected 'x,y'")
    return parts[0], parts[1]


def _domain(s: str) -> tuple[float, float, float, float] | None:
    if str(s).strip().lower() in ("auto", "", "none"):
        return None
    parts = [float(v) for v in str(s).split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("Expected 'xmin,xmax,ymin,ymax' or 'auto'")
    xmin, xmax, ymin, ymax = parts
    if not (xmax > xmin and ymax > ymin):
        raise argparse.ArgumentTypeError("Invalid domain bounds")
    return xmin, xmax, ymin, ymax


def _float_list(s: str) -> list[float]:
    vals = [float(v) for v in str(s).split(",") if str(v).strip()]
    if not vals:
        raise argparse.ArgumentTypeError("Expected one or more comma-separated floats")
    return vals


def _finite_mean(vals: list[float]) -> float:
    a = [float(v) for v in vals if math.isfinite(float(v))]
    return float(np.mean(a)) if a else float("nan")


def _finite_std(vals: list[float]) -> float:
    a = [float(v) for v in vals if math.isfinite(float(v))]
    return float(np.std(a, ddof=0)) if a else float("nan")


def resolve_domain(domain_arg, x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    if domain_arg is not None:
        return domain_arg
    return float(np.min(x)), float(np.max(x)), float(np.min(y)), float(np.max(y))


def boundary_endpoint(
    origin: tuple[float, float],
    angle_deg: float,
    domain: tuple[float, float, float, float],
) -> tuple[float, float, float]:
    """Return the far-side endpoint where the ray exits the rectangular domain."""
    ox, oy = origin
    xmin, xmax, ymin, ymax = domain
    theta = math.radians(float(angle_deg))
    dx = math.cos(theta)
    dy = math.sin(theta)
    eps = 1.0e-12
    candidates: list[tuple[float, float, float]] = []

    if abs(dx) > eps:
        for xb in (xmin, xmax):
            lam = (xb - ox) / dx
            if lam > eps:
                yb = oy + lam * dy
                if ymin - 1e-9 <= yb <= ymax + 1e-9:
                    candidates.append((lam, xb, yb))
    if abs(dy) > eps:
        for yb in (ymin, ymax):
            lam = (yb - oy) / dy
            if lam > eps:
                xb = ox + lam * dx
                if xmin - 1e-9 <= xb <= xmax + 1e-9:
                    candidates.append((lam, xb, yb))

    if not candidates:
        raise ValueError(f"Ray angle {angle_deg} deg from {origin} does not intersect domain {domain}")
    lam, ex, ey = max(candidates, key=lambda item: item[0])
    return float(ex), float(ey), float(lam)


def build_rays(args, domain: tuple[float, float, float, float]) -> list[dict[str, Any]]:
    if args.angles_deg.strip():
        angles = _float_list(args.angles_deg)
    else:
        if int(args.ray_count) < 1:
            raise ValueError("--ray-count must be >= 1")
        angles = list(np.linspace(float(args.angle_min_deg), float(args.angle_max_deg), int(args.ray_count)))

    if not (0.0 <= float(args.ray_start_frac) < float(args.ray_end_frac) <= 1.0):
        raise ValueError("Require 0 <= --ray-start-frac < --ray-end-frac <= 1")

    ix, iy = args.inj_center
    rays = []
    for i, angle in enumerate(angles):
        ex, ey, length = boundary_endpoint((ix, iy), float(angle), domain)
        u = np.linspace(float(args.ray_start_frac), float(args.ray_end_frac), int(args.profile_points))
        lx = ix + u * (ex - ix)
        ly = iy + u * (ey - iy)
        arc = np.hypot(lx - ix, ly - iy)
        rays.append(dict(
            ray_id=f"ray{i:02d}",
            ray_index=i,
            angle_deg=float(angle),
            x_end=float(ex),
            y_end=float(ey),
            ray_length=float(length),
            x=lx.astype(np.float64),
            y=ly.astype(np.float64),
            arc=arc.astype(np.float64),
        ))
    return rays


def validate_ray_geometry(
    rays: list[dict[str, Any]],
    domain: tuple[float, float, float, float],
    min_domain_diag_frac: float,
) -> dict[str, float]:
    lengths = np.asarray([float(ray["ray_length"]) for ray in rays], dtype=np.float64)
    domain_diag = float(math.hypot(domain[1] - domain[0], domain[3] - domain[2]))
    if lengths.size == 0 or not math.isfinite(domain_diag) or domain_diag <= 0.0:
        raise RuntimeError("Invalid multi-ray geometry: no rays or non-positive domain diagonal.")

    ray_length_min = float(np.min(lengths))
    ray_length_median = float(np.median(lengths))
    ray_length_max = float(np.max(lengths))
    threshold = float(min_domain_diag_frac) * domain_diag
    if float(min_domain_diag_frac) > 0.0 and ray_length_median < threshold:
        raise RuntimeError(
            "Invalid multi-ray geometry: median ray length "
            f"{ray_length_median:.6g} is below {float(min_domain_diag_frac):.3g} "
            f"of the domain diagonal ({threshold:.6g}). Check --domain and --inj-center."
        )
    return dict(
        domain_diag=domain_diag,
        ray_length_min=ray_length_min,
        ray_length_median=ray_length_median,
        ray_length_max=ray_length_max,
        min_domain_diag_frac=float(min_domain_diag_frac),
    )


def time_indices_from_fracs(time_unique: np.ndarray, frac_spec: str) -> list[int]:
    fracs = _float_list(frac_spec)
    n = int(len(time_unique))
    idx = sorted(set(int(round(float(f) * (n - 1))) for f in fracs))
    return [min(max(i, 0), n - 1) for i in idx]


def _regular_grid_interpolate(
    xm: np.ndarray,
    ym: np.ndarray,
    vals: np.ndarray,
    xq: np.ndarray,
    yq: np.ndarray,
) -> np.ndarray:
    """Bilinear interpolation fallback for complete rectangular grid slices."""
    xs = np.unique(xm)
    ys = np.unique(ym)
    if xs.size < 2 or ys.size < 2 or xs.size * ys.size != vals.size:
        raise ValueError("slice is not a complete 2-D rectangular grid")

    order = np.lexsort((xm, ym))
    xm_s = xm[order]
    ym_s = ym[order]
    if not (np.allclose(xm_s.reshape(ys.size, xs.size), xs[None, :])
            and np.allclose(ym_s.reshape(ys.size, xs.size), ys[:, None])):
        raise ValueError("slice points are not a complete 2-D rectangular grid")
    grid = vals[order].reshape(ys.size, xs.size)

    xqc = np.clip(xq, xs[0], xs[-1])
    yqc = np.clip(yq, ys[0], ys[-1])
    ix0 = np.searchsorted(xs, xqc, side="right") - 1
    iy0 = np.searchsorted(ys, yqc, side="right") - 1
    ix0 = np.clip(ix0, 0, xs.size - 2)
    iy0 = np.clip(iy0, 0, ys.size - 2)
    ix1 = ix0 + 1
    iy1 = iy0 + 1

    x0 = xs[ix0]
    x1 = xs[ix1]
    y0 = ys[iy0]
    y1 = ys[iy1]
    wx = np.divide(xqc - x0, x1 - x0, out=np.zeros_like(xqc), where=(x1 != x0))
    wy = np.divide(yqc - y0, y1 - y0, out=np.zeros_like(yqc), where=(y1 != y0))

    v00 = grid[iy0, ix0]
    v10 = grid[iy0, ix1]
    v01 = grid[iy1, ix0]
    v11 = grid[iy1, ix1]
    return ((1.0 - wx) * (1.0 - wy) * v00
            + wx * (1.0 - wy) * v10
            + (1.0 - wx) * wy * v01
            + wx * wy * v11)


def _nearest_interpolate(
    xm: np.ndarray,
    ym: np.ndarray,
    vals: np.ndarray,
    xq: np.ndarray,
    yq: np.ndarray,
) -> np.ndarray:
    """Last-resort nearest-neighbor interpolation for non-grid slices."""
    out = np.empty_like(xq, dtype=np.float64)
    for i, (qx, qy) in enumerate(zip(xq, yq)):
        j = int(np.argmin((xm - qx) ** 2 + (ym - qy) ** 2))
        out[i] = vals[j]
    return out


def interpolate_true_profile(
    x: np.ndarray,
    y: np.ndarray,
    s: np.ndarray,
    time_id: np.ndarray,
    time_index: int,
    ray: dict[str, Any],
) -> np.ndarray:
    mask = time_id == int(time_index)
    if int(mask.sum()) < 8:
        return np.full_like(ray["x"], np.nan, dtype=np.float64)
    xm = x[mask]
    ym = y[mask]
    vals = s[mask]
    try:
        from scipy.interpolate import griddata
        pts = np.column_stack([xm, ym])
        out = griddata(pts, vals, (ray["x"], ray["y"]), method="linear")
        bad = ~np.isfinite(out)
        if np.any(bad):
            out[bad] = griddata(pts, vals, (ray["x"][bad], ray["y"][bad]), method="nearest")
    except Exception:
        try:
            out = _regular_grid_interpolate(xm, ym, vals, ray["x"], ray["y"])
        except Exception:
            out = _nearest_interpolate(xm, ym, vals, ray["x"], ray["y"])
    return np.asarray(out, dtype=np.float64)


def front_row_metrics(
    w_true: dict[str, float],
    w_pred: dict[str, float],
    arc: np.ndarray,
    min_ds_jump: float,
) -> dict[str, float | int]:
    width_ratio = (
        w_pred["thickness_1090"] / w_true["thickness_1090"]
        if (math.isfinite(w_true["thickness_1090"]) and w_true["thickness_1090"] > 0)
        else float("nan")
    )
    width_relerr = width_ratio - 1.0 if math.isfinite(width_ratio) else float("nan")
    inverse_slope_ratio = (
        w_pred["thickness_grad"] / w_true["thickness_grad"]
        if (math.isfinite(w_true["thickness_grad"]) and w_true["thickness_grad"] > 0)
        else float("nan")
    )
    sharpness_ratio = (
        w_pred["max_abs_grad"] / w_true["max_abs_grad"]
        if (math.isfinite(w_true["max_abs_grad"]) and w_true["max_abs_grad"] > 0)
        else float("nan")
    )
    pos_signed = (
        w_pred["front_position_mid"] - w_true["front_position_mid"]
        if (math.isfinite(w_true["front_position_mid"]) and math.isfinite(w_pred["front_position_mid"]))
        else float("nan")
    )
    pos_abs = abs(pos_signed) if math.isfinite(pos_signed) else float("nan")
    arc_span = float(np.max(arc) - np.min(arc)) if arc.size else float("nan")
    pos_rel = pos_abs / arc_span if (math.isfinite(pos_abs) and arc_span > 0) else float("nan")
    pos_maxgrad_signed = (
        w_pred["front_position_maxgrad"] - w_true["front_position_maxgrad"]
        if (math.isfinite(w_true["front_position_maxgrad"]) and math.isfinite(w_pred["front_position_maxgrad"]))
        else float("nan")
    )
    pos_maxgrad_abs = abs(pos_maxgrad_signed) if math.isfinite(pos_maxgrad_signed) else float("nan")
    valid_front = int(
        math.isfinite(w_true["dS_jump"])
        and w_true["dS_jump"] >= float(min_ds_jump)
        and math.isfinite(w_true["thickness_1090"])
        and w_true["thickness_1090"] > 0
        and math.isfinite(w_true["front_position_mid"])
    )
    return dict(
        valid_front=valid_front,
        W_front_true=w_true["thickness_1090"],
        W_front_pred=w_pred["thickness_1090"],
        W_front_ratio=width_ratio,
        W_front_true_1090=w_true["thickness_1090"],
        W_front_pred_1090=w_pred["thickness_1090"],
        W_front_ratio_1090=width_ratio,
        front_width_ratio_1090=width_ratio,
        W_front_rel_error_1090=width_relerr,
        W_front_abs_rel_error_1090=abs(width_relerr) if math.isfinite(width_relerr) else float("nan"),
        front_inverse_slope_width_true=w_true["thickness_grad"],
        front_inverse_slope_width_pred=w_pred["thickness_grad"],
        front_inverse_slope_width_ratio=inverse_slope_ratio,
        W_front_ratio_legacy_grad=inverse_slope_ratio,
        W_front_ratio_grad=inverse_slope_ratio,
        front_sharpness_true=w_true["max_abs_grad"],
        front_sharpness_pred=w_pred["max_abs_grad"],
        front_sharpness_ratio=sharpness_ratio,
        front_position_true_mid=w_true["front_position_mid"],
        front_position_pred_mid=w_pred["front_position_mid"],
        front_position_signed_error_mid=pos_signed,
        front_position_abs_error_mid=pos_abs,
        front_position_rel_error_mid=pos_rel,
        front_position_true_maxgrad=w_true["front_position_maxgrad"],
        front_position_pred_maxgrad=w_pred["front_position_maxgrad"],
        front_position_signed_error_maxgrad=pos_maxgrad_signed,
        front_position_abs_error_maxgrad=pos_maxgrad_abs,
        dS_jump_true=w_true["dS_jump"],
        dS_jump_pred=w_pred["dS_jump"],
    )


AGG_METRICS = (
    "W_front_ratio_1090",
    "W_front_abs_rel_error_1090",
    "front_inverse_slope_width_ratio",
    "front_sharpness_ratio",
    "front_position_signed_error_mid",
    "front_position_abs_error_mid",
    "front_position_rel_error_mid",
    "front_position_abs_error_maxgrad",
)


def summarize(rows: list[dict[str, Any]], group_fields: tuple[str, ...]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row.get(field) for field in group_fields)
        groups.setdefault(key, []).append(row)

    out = []
    for key, group in sorted(groups.items(), key=lambda kv: tuple(str(v) for v in kv[0])):
        valid = [r for r in group if int(r.get("valid_front", 0)) == 1]
        rec = {field: value for field, value in zip(group_fields, key)}
        rec["n_time_ray_total"] = int(len(group))
        rec["n_time_ray_valid"] = int(len(valid))
        for metric in AGG_METRICS:
            vals = [float(r.get(metric, float("nan"))) for r in valid]
            rec[f"{metric}_mean"] = _finite_mean(vals)
            rec[f"{metric}_std"] = _finite_std(vals)
        out.append(rec)
    return out


def filter_runs(specs: list[GA.RunSpec], args) -> list[GA.RunSpec]:
    out = specs
    if args.primary_only:
        out = [s for s in out if s.training_strategy == "BASE" and s.leave_out == "NONE"]
    if args.exp_filter.strip():
        keep = set(v.strip() for v in args.exp_filter.split(",") if v.strip())
        out = [s for s in out if s.exp_name in keep]
    if args.training_strategy_filter.strip():
        keep = set(v.strip().upper() for v in args.training_strategy_filter.split(",") if v.strip())
        out = [s for s in out if str(s.training_strategy).upper() in keep]
    if args.leave_out_filter.strip():
        keep = set(v.strip() for v in args.leave_out_filter.split(",") if v.strip())
        out = [s for s in out if s.leave_out in keep]
    if args.run_id_regex.strip():
        pat = re.compile(args.run_id_regex)
        out = [s for s in out if pat.search(s.run_id)]
    if int(args.max_runs) > 0:
        out = out[: int(args.max_runs)]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Multi-ray front diagnostics for M0-M7 checkpoints.")
    ap.add_argument("--ckpt-dir", type=str, default="./Results1")
    ap.add_argument("--data", type=str, default="./tables_cache_0_723_step1.pt")
    ap.add_argument("--out-dir", type=str, default="./eval_M0_M7/multiray_front")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--infer-batch-size", type=int, default=65536)
    ap.add_argument("--primary-only", action="store_true", help="Evaluate only BASE + LOO-NONE runs.")
    ap.add_argument("--exp-filter", type=str, default="", help="Comma-separated exp names, e.g. M0,M4,M7.")
    ap.add_argument("--training-strategy-filter", type=str, default="", help="Comma-separated training strategies, e.g. BASE,BETA.")
    ap.add_argument("--leave-out-filter", type=str, default="", help="Comma-separated leave-out names.")
    ap.add_argument("--run-id-regex", type=str, default="", help="Regex filter applied to run_id.")
    ap.add_argument("--max-runs", type=int, default=0, help="Debug limit; <=0 means no limit.")

    ap.add_argument("--inj-center", type=_pair, default=(0.0, 0.0))
    ap.add_argument("--domain", type=_domain, default=None, help="'auto' or xmin,xmax,ymin,ymax.")
    ap.add_argument("--ray-count", type=int, default=9)
    ap.add_argument("--angle-min-deg", type=float, default=15.0)
    ap.add_argument("--angle-max-deg", type=float, default=75.0)
    ap.add_argument("--angles-deg", type=str, default="", help="Overrides min/max/count, e.g. 20,30,45,60,70.")
    ap.add_argument("--profile-points", type=int, default=201)
    ap.add_argument("--ray-start-frac", type=float, default=0.0)
    ap.add_argument("--ray-end-frac", type=float, default=1.0)
    ap.add_argument("--ray-min-domain-diag-frac", type=float, default=0.25,
                    help="Fail if median full ray length is below this fraction of the domain diagonal; <=0 disables.")
    ap.add_argument("--time-fracs", type=str, default="0.25,0.5,0.75")
    ap.add_argument("--min-ds-jump", type=float, default=0.05)

    ap.add_argument("--s-ic-co2", type=float, default=0.0)
    ap.add_argument("--Snr", type=float, default=0.0)
    ap.add_argument("--Sw-irr", type=float, default=0.2)
    args = ap.parse_args()

    GA.ensure_dir(args.out_dir)
    device = torch.device("cuda" if (args.device.lower().startswith("cuda") and torch.cuda.is_available()) else "cpu")

    print(f"[data] {args.data}")
    arrays, time_unique = GA.load_dataset_pack(args.data)
    x = GA.to_np_1d(arrays["x"]).astype(np.float64)
    y = GA.to_np_1d(arrays["y"]).astype(np.float64)
    t = GA.to_np_1d(arrays["t"]).astype(np.float64)
    s = np.clip(GA.to_np_1d(arrays["Sco2"]).astype(np.float64), 0.0, 1.0)
    time_id, tu = HS.assign_time_id(t, time_unique)

    domain = resolve_domain(args.domain, x, y)
    rays = build_rays(args, domain)
    ray_geometry = validate_ray_geometry(rays, domain, args.ray_min_domain_diag_frac)
    t_idx = time_indices_from_fracs(tu, args.time_fracs)
    print(f"[domain] {domain}")
    print(f"[geometry] {ray_geometry}")
    print(f"[times] {t_idx}")
    ray_log = [(r["ray_id"], round(r["angle_deg"], 3), float(format(r["ray_length"], ".6g"))) for r in rays]
    print(f"[rays] {ray_log}")

    print("[true] precomputing ray profiles")
    true_cache: dict[tuple[int, str], dict[str, Any]] = {}
    for k in t_idx:
        for ray in rays:
            s_true_line = interpolate_true_profile(x, y, s, time_id, k, ray)
            w_true = HS.front_thickness_from_profile(ray["arc"], s_true_line)
            true_cache[(k, ray["ray_id"])] = dict(s_true=s_true_line, w_true=w_true)

    specs = filter_runs(HT.discover_runs(args.ckpt_dir), args)
    if not specs:
        raise RuntimeError(f"No runs selected from checkpoint dir: {args.ckpt_dir}")
    print(f"[runs] {len(specs)} selected")

    diag_rows: list[dict[str, Any]] = []
    profile_rows: list[dict[str, Any]] = []

    for spec in specs:
        print(f"[eval] {spec.run_id}")
        model, cfg = HT.load_model_for_run(spec, device, args.s_ic_co2, args.Snr, args.Sw_irr)
        base = dict(run_id=spec.run_id, exp_name=spec.exp_name,
                    training_strategy=spec.training_strategy, leave_out=spec.leave_out, seed=spec.seed)

        for k in t_idx:
            tv = float(tu[k])
            all_x = np.concatenate([ray["x"] for ray in rays])
            all_y = np.concatenate([ray["y"] for ray in rays])
            all_t = np.full_like(all_x, tv, dtype=np.float64)
            _, all_s_pred = GA.predict_model(model, all_x, all_y, all_t, cfg, device, int(args.infer_batch_size))

            offset = 0
            for ray in rays:
                n = int(ray["x"].shape[0])
                s_pred_line = np.asarray(all_s_pred[offset: offset + n], dtype=np.float64)
                offset += n

                cached = true_cache[(k, ray["ray_id"])]
                s_true_line = cached["s_true"]
                w_true = cached["w_true"]
                w_pred = HS.front_thickness_from_profile(ray["arc"], s_pred_line)
                metrics = front_row_metrics(w_true, w_pred, ray["arc"], args.min_ds_jump)

                diag_rows.append(dict(
                    **base,
                    time_index=int(k),
                    time_s=tv,
                    ray_id=ray["ray_id"],
                    ray_index=int(ray["ray_index"]),
                    ray_angle_deg=float(ray["angle_deg"]),
                    ray_x_end=float(ray["x_end"]),
                    ray_y_end=float(ray["y_end"]),
                    ray_length=float(ray["ray_length"]),
                    **metrics,
                ))

                for j in range(n):
                    profile_rows.append(dict(
                        **base,
                        time_index=int(k),
                        time_s=tv,
                        ray_id=ray["ray_id"],
                        ray_index=int(ray["ray_index"]),
                        ray_angle_deg=float(ray["angle_deg"]),
                        arclength=float(ray["arc"][j]),
                        x=float(ray["x"][j]),
                        y=float(ray["y"][j]),
                        s_true=float(s_true_line[j]),
                        s_pred=float(s_pred_line[j]),
                    ))

    summary_rows = summarize(diag_rows, ("run_id", "exp_name", "training_strategy", "leave_out", "seed"))
    seed_aggregate_rows = HT.aggregate_summary_by_seed(summary_rows)
    by_time_rows = summarize(diag_rows, ("run_id", "exp_name", "training_strategy", "leave_out", "seed", "time_index", "time_s"))
    by_ray_rows = summarize(diag_rows, ("run_id", "exp_name", "training_strategy", "leave_out", "seed", "ray_id", "ray_angle_deg"))

    GA.write_csv(os.path.join(args.out_dir, "multiray_front_diagnostics.csv"), diag_rows)
    GA.write_csv(os.path.join(args.out_dir, "multiray_front_profiles.csv"), profile_rows)
    GA.write_csv(os.path.join(args.out_dir, "multiray_front_summary.csv"), summary_rows)
    GA.write_csv(os.path.join(args.out_dir, "multiray_front_seed_aggregate.csv"), seed_aggregate_rows)
    GA.write_csv(os.path.join(args.out_dir, "multiray_front_by_time.csv"), by_time_rows)
    GA.write_csv(os.path.join(args.out_dir, "multiray_front_by_ray.csv"), by_ray_rows)

    config = dict(
        data=args.data,
        ckpt_dir=args.ckpt_dir,
        out_dir=args.out_dir,
        device=str(device),
        primary_only=bool(args.primary_only),
        exp_filter=args.exp_filter,
        training_strategy_filter=args.training_strategy_filter,
        leave_out_filter=args.leave_out_filter,
        run_id_regex=args.run_id_regex,
        domain=dict(xmin=domain[0], xmax=domain[1], ymin=domain[2], ymax=domain[3]),
        inj_center=dict(x=args.inj_center[0], y=args.inj_center[1]),
        ray_count=len(rays),
        angles_deg=[float(r["angle_deg"]) for r in rays],
        profile_points=int(args.profile_points),
        ray_start_frac=float(args.ray_start_frac),
        ray_end_frac=float(args.ray_end_frac),
        ray_min_domain_diag_frac=float(args.ray_min_domain_diag_frac),
        ray_geometry=ray_geometry,
        time_indices=[int(i) for i in t_idx],
        time_seconds=[float(tu[i]) for i in t_idx],
        min_ds_jump=float(args.min_ds_jump),
        outputs=dict(
            diagnostics="multiray_front_diagnostics.csv",
            profiles="multiray_front_profiles.csv",
            summary="multiray_front_summary.csv",
            seed_aggregate="multiray_front_seed_aggregate.csv",
            by_time="multiray_front_by_time.csv",
            by_ray="multiray_front_by_ray.csv",
        ),
    )
    with open(os.path.join(args.out_dir, "multiray_front_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(f"[done] wrote outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
