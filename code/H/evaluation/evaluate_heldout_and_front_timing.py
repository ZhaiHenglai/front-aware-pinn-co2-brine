"""
Held-out accuracy + front-timing evaluator for the M0-M7 PINN suite.

Adds three things the canonical suite does not currently provide:

  1. HELD-OUT accuracy.  Computes RelL2 of saturation, pressure, and the
     front-band saturation on a held-out split, AND on the in-sample (train)
     split, so you can SEE the in-sample-vs-held-out gap.  See the validity
     note below.
  2. BREAKTHROUGH-TIME error (E_bt).  CO2 arrival time at a producer probe ring:
     |t_breakthrough_pred - t_breakthrough_true|.  This is the canonical
     Fuks-Tchelepi symptom and a single headline scalar.
  3. FRONT DIAGNOSTICS along the injector->producer diagonal:
     - W_front_ratio_1090: resolved transition-band width ratio.
     - front_sharpness_ratio: maximum-gradient ratio.
     - front_position_abs_error_mid: 50%-jump front-location error.
     - front_inverse_slope_width_ratio: legacy gradient-width diagnostic.

It REUSES the verified model loader / predictor / metrics from
`evaluate_global_accuracy_M0_M7_L6_dS_shortlabels.py`, and the torch-free split
and diagnostics from `heldout_split.py`, so nothing is reimplemented.

--------------------------------------------------------------------------------
HELD-OUT VALIDITY (read this):
A held-out number is only publishable if the SAME points were excluded from the
supervised data loss during TRAINING.  The checkpoints in Results1 were trained
on all data, so with `--split-mode != none` the "held-out" rows were still seen
in training -> the numbers remain in-sample until you retrain with the matching
split.  Use heldout_split.training_keep_indices(...) with the identical spec in
the training script, then re-run this evaluator.  Breakthrough and thickness are
field-fidelity diagnostics and do not require a split (the same in-sample caveat
that applies to all accuracy-vs-truth metrics still holds).
--------------------------------------------------------------------------------

Usage (CPU):
  python3 evaluate_heldout_and_front_timing.py \
      --ckpt-dir ./Results1 --data ./tables_cache_0_723_step1.pt \
      --out-dir ./eval_M0_M7/heldout_front_timing \
      --split-mode time_holdout --hold-every 5 --hold-start 2

Outputs (CSV + JSON) in --out-dir:
  heldout_accuracy.csv, breakthrough_time.csv, breakthrough_curves.csv,
  front_thickness.csv, front_profiles.csv, summary_heldout_front_timing.csv,
  heldout_front_timing_config.json

Front-diagnostic columns:
  W_front_ratio_1090                    10-90% transition-width ratio, pred/true
  front_sharpness_ratio                 max|dS/ds|_pred / max|dS/ds|_true
  front_position_abs_error_mid          absolute 50%-jump crossing error
  front_inverse_slope_width_ratio       legacy (dS/max|dS/ds|)_pred / true
  W_front_ratio                         compatibility alias for W_front_ratio_1090
  W_front_ratio_legacy_grad             compatibility alias for front_inverse_slope_width_ratio
"""

from __future__ import annotations

import os
import sys
import json
import argparse
from typing import Any

import numpy as np

# Reuse the verified evaluator (model build, predict, metrics, IO).
import evaluate_global_accuracy_M0_M7_L6_dS_shortlabels as GA
import heldout_split as HS

import torch


# -----------------------------------------------------------------------------
# Run discovery and model loading (mirrors GA.evaluate_run model build)
# -----------------------------------------------------------------------------
def discover_runs(ckpt_dir: str) -> list[GA.RunSpec]:
    import glob
    specs: list[GA.RunSpec] = []
    for p in sorted(glob.glob(os.path.join(ckpt_dir, "*.pt"))):
        exp, strat, loo, seed = GA.parse_run_from_filename(p)
        if exp is None:
            continue
        run_id = f"{exp}_{strat}_LOO-{loo}_seed{seed}"
        specs.append(GA.RunSpec(run_id, exp, strat, loo, seed, p))
    return specs


def load_model_for_run(spec: GA.RunSpec, device, s_ic: float, Snr: float, Sw_irr: float):
    state, ckpt_cfg = GA.load_checkpoint(spec.ckpt_path)
    cfg = GA.default_cfg()
    cfg["s_ic_co2"] = float(s_ic)
    cfg["Snr"] = float(Snr)
    cfg["Sw_irr"] = float(Sw_irr)
    GA.apply_exp_preset(cfg, spec.exp_name)
    GA.cfg_update_from_ckpt(cfg, ckpt_cfg)
    ovr = cfg.get("P_ref_override", None)
    cfg["P_ref"] = float(ovr) if ovr is not None else (
        float(cfg["mu_ref"]) * float(cfg["U_ref"]) * float(cfg["L_ref"]) / float(cfg["k_ref"])
    )
    model = GA.build_model(cfg)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        raise RuntimeError(f"{spec.run_id}: state_dict missing keys: {missing}")
    model.to(device)
    model.eval()
    model.set_beta(float(cfg.get("beta_eval", 1.0)))
    return model, cfg


# -----------------------------------------------------------------------------
# 1. Held-out accuracy
# -----------------------------------------------------------------------------
def _subsample(idx: np.ndarray, max_n: int, seed: int) -> np.ndarray:
    if max_n <= 0 or idx.size <= max_n:
        return idx
    rng = np.random.default_rng(seed)
    sel = rng.choice(idx, size=int(max_n), replace=False)
    sel.sort()
    return sel


def accuracy_on_indices(model, cfg, fields, idx, device, batch):
    x, y, t, p_pa, s = fields
    if idx.size == 0:
        return None
    p_tilde_pred, s_pred = GA.predict_model(model, x[idx], y[idx], t[idx], cfg, device, batch)
    P_ref = float(cfg["P_ref"]); p0 = float(cfg["p0"])
    p_true_tilde = (p_pa[idx] - p0) / P_ref
    s_true = s[idx]
    met_s = GA.regression_basic(s_pred, s_true)
    met_p = GA.regression_basic(p_tilde_pred, p_true_tilde)
    fl = float(cfg.get("front_low", 0.05)); fh = float(cfg.get("front_high", 0.30))
    fb = (s_true >= fl) & (s_true <= fh)
    met_fb = GA.regression_basic(s_pred[fb], s_true[fb]) if np.any(fb) else None
    return dict(
        n=int(idx.size),
        relL2_S=met_s["rel_l2"], rmse_S=met_s["rmse"],
        relL2_p_tilde=met_p["rel_l2"], rmse_p_tilde=met_p["rmse"],
        relL2_S_front=(met_fb["rel_l2"] if met_fb else float("nan")),
        rmse_S_front=(met_fb["rmse"] if met_fb else float("nan")),
        n_front=int(fb.sum()),
    )


# -----------------------------------------------------------------------------
# 2. Breakthrough time at a producer probe ring
# -----------------------------------------------------------------------------
def breakthrough_for_run(model, cfg, fields, time_id, time_unique, device, batch, args):
    x, y, t, p_pa, s = fields
    ox, oy = args.outlet_center if args.outlet_center else (float(cfg["L_ref"]), float(cfg["L_ref"]))
    d_out = np.hypot(x - ox, y - oy)
    probe = (d_out >= float(args.well_radius)) & (d_out <= float(args.probe_radius))
    idx = np.nonzero(probe)[0]
    if idx.size == 0:
        return None, []
    p_tilde_pred, s_pred = GA.predict_model(model, x[idx], y[idx], t[idx], cfg, device, batch)
    tid = time_id[idx]
    Tn = int(time_unique.shape[0])
    s_true_curve = np.full(Tn, np.nan)
    s_pred_curve = np.full(Tn, np.nan)
    for k in range(Tn):
        m = tid == k
        if np.any(m):
            s_true_curve[k] = float(np.mean(s[idx][m]))
            s_pred_curve[k] = float(np.mean(s_pred[m]))
    good = np.isfinite(s_true_curve) & np.isfinite(s_pred_curve)
    times = time_unique[good]
    bt = HS.breakthrough_metrics(times, s_true_curve[good], s_pred_curve[good], float(args.bt_threshold))
    bt["n_probe_points_per_time_mean"] = float(idx.size / max(1, Tn))
    curves = [dict(time_s=float(time_unique[k]), s_true=float(s_true_curve[k]), s_pred=float(s_pred_curve[k]))
              for k in range(Tn) if good[k]]
    return bt, curves


# -----------------------------------------------------------------------------
# 3. Front thickness along the injector->producer diagonal
# -----------------------------------------------------------------------------
def thickness_for_run(model, cfg, fields, time_id, time_unique, device, batch, args):
    from scipy.interpolate import griddata
    x, y, t, p_pa, s = fields
    ix, iy = args.inj_center if args.inj_center else (0.0, 0.0)
    ox, oy = args.outlet_center if args.outlet_center else (float(cfg["L_ref"]), float(cfg["L_ref"]))
    M = int(args.profile_points)
    u = np.linspace(0.0, 1.0, M)
    lx = ix + u * (ox - ix)
    ly = iy + u * (oy - iy)
    arc = np.hypot(lx - ix, ly - iy)  # physical arc-length from injector

    Tn = int(time_unique.shape[0])
    # choose time indices at requested fractions of the time range
    fracs = [float(f) for f in str(args.thickness_fracs).split(",") if f.strip()]
    t_idx = sorted(set(int(round(f * (Tn - 1))) for f in fracs))

    rows = []
    profiles = []
    for k in t_idx:
        tv = float(time_unique[k])
        tline = np.full(M, tv)
        _, s_pred = GA.predict_model(model, lx, ly, tline, cfg, device, batch)
        # interpolate true field from this time-slice's scattered points
        m = time_id == k
        if m.sum() < 8:
            continue
        pts = np.column_stack([x[m], y[m]])
        s_true_line = griddata(pts, s[m], (lx, ly), method="linear")
        nan = ~np.isfinite(s_true_line)
        if np.any(nan):
            s_true_line[nan] = griddata(pts, s[m], (lx[nan], ly[nan]), method="nearest")
        w_true = HS.front_thickness_from_profile(arc, s_true_line)
        w_pred = HS.front_thickness_from_profile(arc, s_pred)
        inverse_slope_ratio = (
            w_pred["thickness_grad"] / w_true["thickness_grad"]
            if (np.isfinite(w_true["thickness_grad"]) and w_true["thickness_grad"] > 0)
            else float("nan")
        )
        width_ratio_1090 = (
            w_pred["thickness_1090"] / w_true["thickness_1090"]
            if (np.isfinite(w_true["thickness_1090"]) and w_true["thickness_1090"] > 0)
            else float("nan")
        )
        width_relerr_1090 = (width_ratio_1090 - 1.0) if np.isfinite(width_ratio_1090) else float("nan")
        sharpness_ratio = (
            w_pred["max_abs_grad"] / w_true["max_abs_grad"]
            if (np.isfinite(w_true["max_abs_grad"]) and w_true["max_abs_grad"] > 0)
            else float("nan")
        )
        pos_true_mid = w_true["front_position_mid"]
        pos_pred_mid = w_pred["front_position_mid"]
        pos_signed_mid = (
            pos_pred_mid - pos_true_mid
            if (np.isfinite(pos_true_mid) and np.isfinite(pos_pred_mid))
            else float("nan")
        )
        pos_abs_mid = abs(pos_signed_mid) if np.isfinite(pos_signed_mid) else float("nan")
        arc_span = float(np.max(arc) - np.min(arc)) if arc.size else float("nan")
        pos_rel_mid = (pos_abs_mid / arc_span) if (np.isfinite(pos_abs_mid) and arc_span > 0) else float("nan")
        pos_true_maxgrad = w_true["front_position_maxgrad"]
        pos_pred_maxgrad = w_pred["front_position_maxgrad"]
        pos_signed_maxgrad = (
            pos_pred_maxgrad - pos_true_maxgrad
            if (np.isfinite(pos_true_maxgrad) and np.isfinite(pos_pred_maxgrad))
            else float("nan")
        )
        pos_abs_maxgrad = abs(pos_signed_maxgrad) if np.isfinite(pos_signed_maxgrad) else float("nan")
        rows.append(dict(
            time_index=k, time_s=tv,
            W_front_true=w_true["thickness_1090"], W_front_pred=w_pred["thickness_1090"],
            W_front_ratio=width_ratio_1090,
            W_front_true_1090=w_true["thickness_1090"], W_front_pred_1090=w_pred["thickness_1090"],
            W_front_ratio_1090=width_ratio_1090,
            front_width_ratio_1090=width_ratio_1090,
            W_front_rel_error_1090=width_relerr_1090,
            W_front_abs_rel_error_1090=abs(width_relerr_1090) if np.isfinite(width_relerr_1090) else float("nan"),
            front_inverse_slope_width_true=w_true["thickness_grad"],
            front_inverse_slope_width_pred=w_pred["thickness_grad"],
            front_inverse_slope_width_ratio=inverse_slope_ratio,
            W_front_true_legacy_grad=w_true["thickness_grad"],
            W_front_pred_legacy_grad=w_pred["thickness_grad"],
            W_front_ratio_legacy_grad=inverse_slope_ratio,
            W_front_ratio_grad=inverse_slope_ratio,
            front_sharpness_true=w_true["max_abs_grad"],
            front_sharpness_pred=w_pred["max_abs_grad"],
            front_sharpness_ratio=sharpness_ratio,
            front_position_true_mid=pos_true_mid,
            front_position_pred_mid=pos_pred_mid,
            front_position_signed_error_mid=pos_signed_mid,
            front_position_abs_error_mid=pos_abs_mid,
            front_position_rel_error_mid=pos_rel_mid,
            front_position_true_maxgrad=pos_true_maxgrad,
            front_position_pred_maxgrad=pos_pred_maxgrad,
            front_position_signed_error_maxgrad=pos_signed_maxgrad,
            front_position_abs_error_maxgrad=pos_abs_maxgrad,
            dS_jump_true=w_true["dS_jump"], dS_jump_pred=w_pred["dS_jump"],
        ))
        for j in range(M):
            profiles.append(dict(time_s=tv, arclength=float(arc[j]),
                                 s_true=float(s_true_line[j]), s_pred=float(s_pred[j])))
    return rows, profiles


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def _pair(s):
    parts = [float(v) for v in str(s).split(",")]
    return (parts[0], parts[1])


def _holdout_spec_match(ckpt_holdout, eval_spec: dict[str, Any]) -> tuple[int, str]:
    mode = str(eval_spec.get("mode", "none")).lower()
    if mode in ("none", "off", ""):
        return (1 if not ckpt_holdout else 0), ("none" if not ckpt_holdout else "checkpoint has holdout but evaluator uses none")
    if not isinstance(ckpt_holdout, dict):
        return 0, "checkpoint has no data_holdout spec"
    ck_mode = str(ckpt_holdout.get("mode", "")).lower()
    if ck_mode != mode:
        return 0, f"mode mismatch checkpoint={ck_mode} evaluator={mode}"
    if mode == "time_holdout":
        keys = ("hold_every", "hold_start")
        for key in keys:
            if int(ckpt_holdout.get(key, -999999)) != int(eval_spec.get(key, -999998)):
                return 0, f"{key} mismatch checkpoint={ckpt_holdout.get(key)} evaluator={eval_spec.get(key)}"
        return 1, "matched"
    if mode == "random_fraction":
        frac_ok = abs(float(ckpt_holdout.get("fraction", float("nan"))) - float(eval_spec.get("fraction", float("nan")))) < 1.0e-12
        seed_ok = int(ckpt_holdout.get("seed", -999999)) == int(eval_spec.get("seed", -999998))
        if not frac_ok:
            return 0, f"fraction mismatch checkpoint={ckpt_holdout.get('fraction')} evaluator={eval_spec.get('fraction')}"
        if not seed_ok:
            return 0, f"seed mismatch checkpoint={ckpt_holdout.get('seed')} evaluator={eval_spec.get('seed')}"
        return 1, "matched"
    return 1, "mode matched; detailed comparison not implemented"


def _to_float_or_nan(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def aggregate_summary_by_seed(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate per-run holdout summaries across training seeds."""
    if not rows:
        return []
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            str(row.get("exp_name", "")),
            str(row.get("training_strategy", "")),
            str(row.get("leave_out", "")),
        )
        groups.setdefault(key, []).append(row)

    id_cols = {"run_id", "exp_name", "training_strategy", "leave_out", "seed", "holdout_spec_note"}
    numeric_cols: list[str] = []
    seen = set()
    for row in rows:
        for key, value in row.items():
            if key in id_cols or key in seen:
                continue
            if np.isfinite(_to_float_or_nan(value)):
                numeric_cols.append(key)
                seen.add(key)

    out: list[dict[str, Any]] = []
    for (exp, strategy, leave), group in sorted(groups.items()):
        seeds = sorted({str(g.get("seed", "")) for g in group if str(g.get("seed", ""))})
        rec: dict[str, Any] = {
            "exp_name": exp,
            "training_strategy": strategy,
            "leave_out": leave,
            "n_runs": len(group),
            "n_unique_seeds": len(seeds),
            "seeds": ",".join(seeds),
        }
        for col in numeric_cols:
            vals = np.asarray([_to_float_or_nan(g.get(col)) for g in group], dtype=float)
            vals = vals[np.isfinite(vals)]
            rec[f"{col}_mean"] = float(np.mean(vals)) if vals.size else float("nan")
            rec[f"{col}_std"] = float(np.std(vals, ddof=1)) if vals.size >= 2 else (0.0 if vals.size == 1 else float("nan"))
            rec[f"{col}_n"] = int(vals.size)
        out.append(rec)
    return out


def main():
    ap = argparse.ArgumentParser(description="Held-out accuracy + breakthrough + front-thickness evaluator.")
    ap.add_argument("--ckpt-dir", type=str, default="./Results1")
    ap.add_argument("--data", type=str, default="./tables_cache_0_723_step1.pt")
    ap.add_argument("--out-dir", type=str, default="./eval_M0_M7/heldout_front_timing")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--infer-batch-size", type=int, default=65536)
    # split
    ap.add_argument("--split-mode", type=str, default="time_holdout",
                    choices=("none", "random_fraction", "time_holdout", "space_radial", "space_quadrant"))
    ap.add_argument("--fraction", type=float, default=0.2)
    ap.add_argument("--hold-every", type=int, default=5)
    ap.add_argument("--hold-start", type=int, default=2)
    ap.add_argument("--r-lo", type=float, default=1.0)
    ap.add_argument("--r-hi", type=float, default=2.5)
    ap.add_argument("--split-x", type=float, default=2.5)
    ap.add_argument("--split-y", type=float, default=2.5)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--seed-filter", type=str, default="", help="Comma-separated training seeds to evaluate, e.g. 0,1,2.")
    ap.add_argument("--max-acc-points", type=int, default=200000)
    # geometry / physics for probes
    ap.add_argument("--inj-center", type=_pair, default=(0.0, 0.0))
    ap.add_argument("--outlet-center", type=_pair, default=None, help="default = (L_ref, L_ref)")
    ap.add_argument("--well-radius", type=float, default=0.5)
    ap.add_argument("--probe-radius", type=float, default=1.0)
    ap.add_argument("--bt-threshold", type=float, default=0.05)
    ap.add_argument("--profile-points", type=int, default=201)
    ap.add_argument("--thickness-fracs", type=str, default="0.25,0.5,0.75")
    # physical fallbacks (only used for legacy ckpts without config)
    ap.add_argument("--s-ic-co2", type=float, default=0.0)
    ap.add_argument("--Snr", type=float, default=0.0)
    ap.add_argument("--Sw-irr", type=float, default=0.2)
    args = ap.parse_args()

    device = torch.device("cuda" if (args.device.lower().startswith("cuda") and torch.cuda.is_available()) else "cpu")
    GA.ensure_dir(args.out_dir)

    print(f"[data] {args.data}")
    arrays, time_unique = GA.load_dataset_pack(args.data)
    x = GA.to_np_1d(arrays["x"]).astype(np.float64)
    y = GA.to_np_1d(arrays["y"]).astype(np.float64)
    t = GA.to_np_1d(arrays["t"]).astype(np.float64)
    p_pa = GA.to_np_1d(arrays["p"]).astype(np.float64)
    s = np.clip(GA.to_np_1d(arrays["Sco2"]).astype(np.float64), 0.0, 1.0)
    fields = (x, y, t, p_pa, s)
    time_id, tu = HS.assign_time_id(t, time_unique)

    # Build the split (single source of truth, shared with training).
    spec = dict(mode=args.split_mode, seed=args.split_seed, fraction=args.fraction,
                hold_every=args.hold_every, hold_start=args.hold_start,
                inj_center=args.inj_center, r_lo=args.r_lo, r_hi=args.r_hi,
                split_x=args.split_x, split_y=args.split_y)
    masks = HS.make_split_masks(x, y, t, time_unique, spec)
    print(f"[split] mode={masks['mode']} n_train={masks['n_train']} n_test={masks['n_test']}")
    if masks["mode"] != "none":
        print("[WARN] Held-out numbers are only valid if TRAINING excluded the SAME split.\n"
              "       Check summary_heldout_front_timing.csv columns checkpoint_data_holdout\n"
              "       and holdout_spec_match; require both to be 1 for true out-of-sample accuracy.")

    idx_test = _subsample(np.nonzero(masks["test_mask"])[0], args.max_acc_points, args.split_seed)
    idx_train = _subsample(np.nonzero(masks["train_mask"])[0], args.max_acc_points, args.split_seed + 1)

    runs = discover_runs(args.ckpt_dir)
    if args.seed_filter.strip():
        keep_seeds = {int(v.strip()) for v in args.seed_filter.split(",") if v.strip()}
        runs = [r for r in runs if int(r.seed) in keep_seeds]
    if not runs:
        raise RuntimeError(f"No checkpoints found in {args.ckpt_dir}")
    print(f"[runs] {[r.run_id for r in runs]}")

    acc_rows, bt_rows, bt_curves, thick_rows, thick_profiles, summary_rows = [], [], [], [], [], []

    for spec_run in runs:
        print(f"[eval] {spec_run.run_id}")
        model, cfg = load_model_for_run(spec_run, device, args.s_ic_co2, args.Snr, args.Sw_irr)
        ckpt_holdout = cfg.get("checkpoint_config", {}).get("data_holdout", None)
        holdout_match, holdout_note = _holdout_spec_match(ckpt_holdout, spec)
        if masks["mode"] != "none" and not holdout_match:
            print(f"[WARN] {spec_run.run_id}: evaluator split does not match checkpoint holdout spec: {holdout_note}")
        batch = int(args.infer_batch_size)

        held = accuracy_on_indices(model, cfg, fields, idx_test, device, batch)
        insample = accuracy_on_indices(model, cfg, fields, idx_train, device, batch)
        base = dict(run_id=spec_run.run_id, exp_name=spec_run.exp_name,
                    training_strategy=spec_run.training_strategy,
                    leave_out=spec_run.leave_out, seed=spec_run.seed)
        if held:
            acc_rows.append({**base, "split": "heldout", **held})
        if insample:
            acc_rows.append({**base, "split": "insample", **insample})

        bt, curves = breakthrough_for_run(model, cfg, fields, time_id, tu, device, batch, args)
        if bt:
            bt_rows.append({**base, **bt})
            for c in curves:
                bt_curves.append({**base, **c})

        trows, tprof = thickness_for_run(model, cfg, fields, time_id, tu, device, batch, args)
        for r in trows:
            thick_rows.append({**base, **r})
        for pr in tprof:
            thick_profiles.append({**base, **pr})

        # one-line summary per run
        inverse_slope_vals = [
            r["front_inverse_slope_width_ratio"]
            for r in trows if np.isfinite(r.get("front_inverse_slope_width_ratio", np.nan))
        ]
        ratio_1090_vals = [r["W_front_ratio_1090"] for r in trows if np.isfinite(r.get("W_front_ratio_1090", np.nan))]
        absrel_1090_vals = [r["W_front_abs_rel_error_1090"] for r in trows if np.isfinite(r.get("W_front_abs_rel_error_1090", np.nan))]
        sharpness_vals = [r["front_sharpness_ratio"] for r in trows if np.isfinite(r.get("front_sharpness_ratio", np.nan))]
        pos_signed_mid_vals = [
            r["front_position_signed_error_mid"]
            for r in trows if np.isfinite(r.get("front_position_signed_error_mid", np.nan))
        ]
        pos_abs_mid_vals = [
            r["front_position_abs_error_mid"]
            for r in trows if np.isfinite(r.get("front_position_abs_error_mid", np.nan))
        ]
        pos_rel_mid_vals = [
            r["front_position_rel_error_mid"]
            for r in trows if np.isfinite(r.get("front_position_rel_error_mid", np.nan))
        ]
        pos_abs_maxgrad_vals = [
            r["front_position_abs_error_maxgrad"]
            for r in trows if np.isfinite(r.get("front_position_abs_error_maxgrad", np.nan))
        ]
        summary_rows.append(dict(
            **base,
            heldout_relL2_S=(held["relL2_S"] if held else float("nan")),
            heldout_relL2_S_front=(held["relL2_S_front"] if held else float("nan")),
            heldout_relL2_p_tilde=(held["relL2_p_tilde"] if held else float("nan")),
            insample_relL2_S=(insample["relL2_S"] if insample else float("nan")),
            insample_to_heldout_gap_S=((held["relL2_S"] - insample["relL2_S"])
                                       if (held and insample) else float("nan")),
            bt_abs_error=(bt["bt_abs_error"] if bt else float("nan")),
            bt_rel_error=(bt["bt_rel_error"] if bt else float("nan")),
            checkpoint_data_holdout=int(bool(ckpt_holdout)),
            holdout_spec_match=int(holdout_match),
            holdout_spec_note=holdout_note,
            W_front_ratio_mean=(float(np.mean(ratio_1090_vals)) if ratio_1090_vals else float("nan")),
            W_front_ratio_1090_mean=(float(np.mean(ratio_1090_vals)) if ratio_1090_vals else float("nan")),
            front_width_ratio_1090_mean=(float(np.mean(ratio_1090_vals)) if ratio_1090_vals else float("nan")),
            W_front_ratio_recommended_mean=(float(np.mean(ratio_1090_vals)) if ratio_1090_vals else float("nan")),
            W_front_abs_rel_error_1090_mean=(float(np.mean(absrel_1090_vals)) if absrel_1090_vals else float("nan")),
            front_inverse_slope_width_ratio_mean=(float(np.mean(inverse_slope_vals)) if inverse_slope_vals else float("nan")),
            W_front_ratio_legacy_grad_mean=(float(np.mean(inverse_slope_vals)) if inverse_slope_vals else float("nan")),
            W_front_ratio_grad_mean=(float(np.mean(inverse_slope_vals)) if inverse_slope_vals else float("nan")),
            front_sharpness_ratio_mean=(float(np.mean(sharpness_vals)) if sharpness_vals else float("nan")),
            front_position_signed_error_mid_mean=(float(np.mean(pos_signed_mid_vals)) if pos_signed_mid_vals else float("nan")),
            front_position_abs_error_mid_mean=(float(np.mean(pos_abs_mid_vals)) if pos_abs_mid_vals else float("nan")),
            front_position_rel_error_mid_mean=(float(np.mean(pos_rel_mid_vals)) if pos_rel_mid_vals else float("nan")),
            front_position_abs_error_maxgrad_mean=(float(np.mean(pos_abs_maxgrad_vals)) if pos_abs_maxgrad_vals else float("nan")),
        ))

    GA.write_csv(os.path.join(args.out_dir, "heldout_accuracy.csv"), acc_rows)
    GA.write_csv(os.path.join(args.out_dir, "breakthrough_time.csv"), bt_rows)
    GA.write_csv(os.path.join(args.out_dir, "breakthrough_curves.csv"), bt_curves)
    GA.write_csv(os.path.join(args.out_dir, "front_thickness.csv"), thick_rows)
    GA.write_csv(os.path.join(args.out_dir, "front_profiles.csv"), thick_profiles)
    GA.write_csv(os.path.join(args.out_dir, "summary_heldout_front_timing.csv"), summary_rows)
    GA.write_csv(os.path.join(args.out_dir, "summary_heldout_front_timing_seed_aggregate.csv"),
                 aggregate_summary_by_seed(summary_rows))

    with open(os.path.join(args.out_dir, "heldout_front_timing_config.json"), "w") as f:
        json.dump(dict(
            data=args.data, ckpt_dir=args.ckpt_dir, device=str(device),
            split=dict(masks_spec=spec, n_train=masks["n_train"], n_test=masks["n_test"],
                       held_time_indices=masks.get("held_time_indices")),
            breakthrough=dict(outlet_center=args.outlet_center, well_radius=args.well_radius,
                              probe_radius=args.probe_radius, threshold=args.bt_threshold),
            thickness=dict(inj_center=args.inj_center, profile_points=args.profile_points,
                           thickness_fracs=args.thickness_fracs),
            holdout_validity_note=("Held-out accuracy is only publishable if training excluded "
                                   "the same split via heldout_split.training_keep_indices with "
                                   "this exact spec."),
        ), f, indent=2)

    print(f"[done] wrote outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
