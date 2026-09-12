"""
Shared, torch-free utilities for held-out evaluation and front-timing diagnostics.

This module is the SINGLE SOURCE OF TRUTH for the train/test split, so the
training run and the evaluation run can use the *identical* held-out set.

Why this matters
----------------
The current PINN is densely supervised on the simulator field (p, Sco2).  If the
model is trained on all data points and then "evaluated" on the same points, the
accuracy numbers are in-sample and optimistically biased.  A held-out split is
only meaningful if the SAME points are excluded from the supervised data loss
during training.  Use `make_split_masks(...)` in BOTH places:

  * Training: keep only rows where `train_mask` is True for the supervised data
    loss (collocation / PDE / BC points are unaffected).
  * Evaluation: report accuracy on `test_mask` rows (held out from training).

All functions here are pure NumPy so they can be unit-tested without torch.
"""

from __future__ import annotations

from typing import Any
import numpy as np


# -----------------------------------------------------------------------------
# Time-bucket assignment (matches evaluate_global_accuracy.build_eval_plan)
# -----------------------------------------------------------------------------
def assign_time_id(t: np.ndarray, time_unique: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Assign each point to its nearest unique-time index.

    Returns (time_id, sorted_time_unique).
    """
    t = np.asarray(t, dtype=np.float64).reshape(-1)
    tu = np.asarray(time_unique, dtype=np.float64).reshape(-1).copy()
    tu.sort()
    Tn = int(tu.shape[0])
    if Tn == 0:
        raise ValueError("time_unique is empty")
    pos = np.searchsorted(tu, t)
    pos = np.clip(pos, 0, Tn - 1)
    prev = np.clip(pos - 1, 0, Tn - 1)
    choose_prev = np.abs(t - tu[prev]) < np.abs(t - tu[pos])
    time_id = np.where(choose_prev, prev, pos).astype(np.int64)
    return time_id, tu


# -----------------------------------------------------------------------------
# Train/test split
# -----------------------------------------------------------------------------
def make_split_masks(
    x: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    time_unique: np.ndarray,
    spec: dict[str, Any],
) -> dict[str, Any]:
    """Build deterministic train/test boolean masks over the N data points.

    spec["mode"] is one of:
      - "none"            : test_mask all False (no holdout; in-sample).
      - "random_fraction" : hold out a random fraction of points.
            spec: fraction (default 0.2), seed (default 0)
      - "time_holdout"    : hold out whole time levels (temporal interpolation test).
            spec: hold_time_indices (list, supports negatives) OR
                  hold_every (default 5) + hold_start (default 2)
      - "space_radial"    : hold out an annulus around the injector (front-region test).
            spec: inj_center (default (0,0)), r_lo (default 1.0), r_hi (default 2.5)
      - "space_quadrant"  : hold out a spatial region toward the producer (extrapolation).
            spec: split_x (default 2.5), split_y (default 2.5)

    Returns dict(train_mask, test_mask, n_train, n_test, mode, seed, spec, [held_time_indices]).
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    t = np.asarray(t, dtype=np.float64).reshape(-1)
    N = int(x.shape[0])
    if not (y.shape[0] == N and t.shape[0] == N):
        raise ValueError(f"x,y,t length mismatch: {x.shape[0]},{y.shape[0]},{t.shape[0]}")

    mode = str(spec.get("mode", "none")).lower()
    seed = int(spec.get("seed", 0))
    test = np.zeros(N, dtype=bool)
    extra: dict[str, Any] = {}

    if mode in ("none", "off", ""):
        pass

    elif mode == "random_fraction":
        frac = float(spec.get("fraction", 0.2))
        if not (0.0 <= frac < 1.0):
            raise ValueError(f"fraction must be in [0,1); got {frac}")
        rng = np.random.default_rng(seed)
        test = rng.random(N) < frac

    elif mode == "time_holdout":
        time_id, tu = assign_time_id(t, time_unique)
        Tn = int(tu.shape[0])
        held = spec.get("hold_time_indices", None)
        if held is None:
            every = int(spec.get("hold_every", 5))
            start = int(spec.get("hold_start", 2))
            held = list(range(start, Tn, max(1, every)))
        else:
            held = [int(i if i >= 0 else Tn + i) for i in held]
        held = sorted(set(int(i) for i in held if 0 <= int(i) < Tn))
        test = np.isin(time_id, np.asarray(held, dtype=np.int64))
        extra["held_time_indices"] = held
        extra["held_time_seconds"] = [float(tu[i]) for i in held]

    elif mode == "space_radial":
        cx, cy = spec.get("inj_center", (0.0, 0.0))
        r = np.hypot(x - float(cx), y - float(cy))
        r_lo = float(spec.get("r_lo", 1.0))
        r_hi = float(spec.get("r_hi", 2.5))
        test = (r >= r_lo) & (r <= r_hi)
        extra["r_lo"] = r_lo
        extra["r_hi"] = r_hi

    elif mode == "space_quadrant":
        sx = float(spec.get("split_x", 2.5))
        sy = float(spec.get("split_y", 2.5))
        test = (x > sx) & (y > sy)
        extra["split_x"] = sx
        extra["split_y"] = sy

    else:
        raise ValueError(f"Unknown split mode: {mode!r}")

    train = ~test
    out = dict(
        train_mask=train,
        test_mask=test,
        n_train=int(train.sum()),
        n_test=int(test.sum()),
        mode=mode,
        seed=seed,
        spec=dict(spec),
    )
    out.update(extra)
    return out


def training_keep_indices(x, y, t, time_unique, spec) -> np.ndarray:
    """Convenience for the TRAINING script: integer indices of rows to KEEP
    in the supervised data loss (i.e. the train side of the same split).

    Example (in the training script, after the dataset arrays are built):

        import heldout_split as HS
        spec = {"mode": "time_holdout", "hold_every": 5, "hold_start": 2}
        keep = HS.training_keep_indices(x_all, y_all, t_all, time_unique, spec)
        # then restrict supervised data tensors to `keep` rows only.
    """
    masks = make_split_masks(x, y, t, time_unique, spec)
    return np.nonzero(masks["train_mask"])[0]


# -----------------------------------------------------------------------------
# Breakthrough-time diagnostic (E_bt)  -- torch-free
# -----------------------------------------------------------------------------
def first_crossing_time(times: np.ndarray, values: np.ndarray, threshold: float) -> float:
    """First time at which `values` crosses `threshold` from below, with linear
    interpolation between bracketing samples.  Returns np.nan if never crossed.

    `times` must be sorted ascending.
    """
    times = np.asarray(times, dtype=np.float64).reshape(-1)
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if times.shape[0] != values.shape[0] or times.shape[0] < 2:
        return float("nan")
    if values[0] >= threshold:
        return float(times[0])
    for i in range(1, times.shape[0]):
        v0, v1 = values[i - 1], values[i]
        if (v0 < threshold) and (v1 >= threshold):
            if v1 == v0:
                return float(times[i])
            frac = (threshold - v0) / (v1 - v0)
            return float(times[i - 1] + frac * (times[i] - times[i - 1]))
    return float("nan")


def breakthrough_metrics(
    times: np.ndarray,
    s_true_curve: np.ndarray,
    s_pred_curve: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """Breakthrough-time error from monitor curves S(t) at a probe location."""
    bt_true = first_crossing_time(times, s_true_curve, threshold)
    bt_pred = first_crossing_time(times, s_pred_curve, threshold)
    err = abs(bt_pred - bt_true) if (np.isfinite(bt_true) and np.isfinite(bt_pred)) else float("nan")
    t_span = float(np.max(times) - np.min(times)) if np.asarray(times).size else float("nan")
    rel = (err / t_span) if (np.isfinite(err) and t_span > 0) else float("nan")
    return dict(
        bt_true=bt_true,
        bt_pred=bt_pred,
        bt_abs_error=err,
        bt_rel_error=rel,
        bt_threshold=float(threshold),
    )


# -----------------------------------------------------------------------------
# Front diagnostics (W_front, sharpness, position)  -- torch-free
# -----------------------------------------------------------------------------
def front_thickness_from_profile(arclength: np.ndarray, s_profile: np.ndarray) -> dict[str, float]:
    """Front diagnostics along a 1-D saturation profile vs arc-length.

    Width estimates (arc-length units):
      - thickness_grad  = dS_jump / max|dS/ds|        (legacy slope-width diagnostic)
      - thickness_1090  = arc-length span over which S crosses the 10% and 90%
                          levels of the local jump around the steepest point
                          (preferred for paper tables).

    Position estimate:
      - front_position_mid = crossing position at the 50% level of the local jump,
                             searched near the steepest point.

    Sharpness estimate:
      - max_abs_grad = max|dS/ds| along the profile.

    NOTE: this is a *jump-relative* width; do NOT use absolute 0.1->0.9 because
    the CO2 saturation here only spans ~[0, 0.8] and the resolved front band is
    Sco2 in [0.05, 0.30].
    """
    s_arc = np.asarray(arclength, dtype=np.float64).reshape(-1)
    s = np.asarray(s_profile, dtype=np.float64).reshape(-1)
    if s_arc.shape[0] != s.shape[0] or s.shape[0] < 3:
        return dict(thickness_grad=float("nan"), thickness_1090=float("nan"),
                    front_position_mid=float("nan"), front_position_maxgrad=float("nan"),
                    dS_jump=float("nan"), max_abs_grad=float("nan"),
                    level_10=float("nan"), level_50=float("nan"), level_90=float("nan"))

    order = np.argsort(s_arc)
    s_arc = s_arc[order]
    s = s[order]

    dS_jump = float(np.max(s) - np.min(s))
    grad = np.gradient(s, s_arc)
    max_abs_grad = float(np.max(np.abs(grad))) if grad.size else float("nan")
    thickness_grad = (dS_jump / max_abs_grad) if (max_abs_grad and max_abs_grad > 0) else float("nan")
    k = int(np.argmax(np.abs(grad))) if grad.size else -1
    front_position_maxgrad = float(s_arc[k]) if k >= 0 else float("nan")

    # 10-90 width measured locally around the steepest descent/ascent point.
    thickness_1090 = float("nan")
    front_position_mid = float("nan")
    lo = mid = hi = float("nan")
    if dS_jump > 0:
        smin = float(np.min(s))
        lo = smin + 0.10 * dS_jump
        mid = smin + 0.50 * dS_jump
        hi = smin + 0.90 * dS_jump
        # walk outward from the steepest point to find where it passes lo and hi
        a_hi = _interp_cross_near(s_arc, s, hi, k)
        a_lo = _interp_cross_near(s_arc, s, lo, k)
        front_position_mid = _interp_cross_near(s_arc, s, mid, k)
        if np.isfinite(a_hi) and np.isfinite(a_lo):
            thickness_1090 = abs(a_lo - a_hi)

    return dict(
        thickness_grad=thickness_grad,
        thickness_1090=thickness_1090,
        front_position_mid=front_position_mid,
        front_position_maxgrad=front_position_maxgrad,
        dS_jump=dS_jump,
        max_abs_grad=max_abs_grad,
        level_10=lo,
        level_50=mid,
        level_90=hi,
    )


def _interp_cross_near(arc: np.ndarray, s: np.ndarray, level: float, k: int) -> float:
    """Arc-length where the profile crosses `level`, searching outward from index k."""
    n = s.shape[0]
    # search forward
    fwd = float("nan")
    for i in range(k, n - 1):
        if (s[i] - level) * (s[i + 1] - level) <= 0 and s[i] != s[i + 1]:
            frac = (level - s[i]) / (s[i + 1] - s[i])
            fwd = arc[i] + frac * (arc[i + 1] - arc[i])
            break
    # search backward
    bwd = float("nan")
    for i in range(k, 0, -1):
        if (s[i] - level) * (s[i - 1] - level) <= 0 and s[i] != s[i - 1]:
            frac = (level - s[i]) / (s[i - 1] - s[i])
            bwd = arc[i] + frac * (arc[i - 1] - arc[i])
            break
    # prefer the nearer crossing to the steepest point
    cands = [c for c in (fwd, bwd) if np.isfinite(c)]
    if not cands:
        return float("nan")
    return float(min(cands, key=lambda a: abs(a - arc[k])))


# -----------------------------------------------------------------------------
# Self-test (torch-free):  python3 heldout_split.py
# -----------------------------------------------------------------------------
def _self_test() -> None:
    rng = np.random.default_rng(0)
    # synthetic field on a 40x40 grid over 6 time levels
    g = np.linspace(0.0, 5.0, 40)
    X, Y = np.meshgrid(g, g)
    x = X.ravel(); y = Y.ravel()
    tu = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    xs = np.concatenate([x] * tu.size)
    ys = np.concatenate([y] * tu.size)
    ts = np.concatenate([np.full_like(x, tv) for tv in tu])

    # --- split masks ---
    for spec in (
        {"mode": "none"},
        {"mode": "random_fraction", "fraction": 0.2, "seed": 1},
        {"mode": "time_holdout", "hold_every": 2, "hold_start": 1},
        {"mode": "space_radial", "r_lo": 1.0, "r_hi": 2.5},
        {"mode": "space_quadrant", "split_x": 2.5, "split_y": 2.5},
    ):
        m = make_split_masks(xs, ys, ts, tu, spec)
        assert m["n_train"] + m["n_test"] == xs.shape[0]
        assert m["train_mask"].sum() == m["n_train"]
        print(f"[split] {spec['mode']:16s} n_test={m['n_test']:6d} / {xs.shape[0]}")
    # determinism
    a = make_split_masks(xs, ys, ts, tu, {"mode": "random_fraction", "fraction": 0.3, "seed": 7})
    b = make_split_masks(xs, ys, ts, tu, {"mode": "random_fraction", "fraction": 0.3, "seed": 7})
    assert np.array_equal(a["test_mask"], b["test_mask"]), "split not deterministic"
    # time_holdout keep-indices are disjoint from held times
    keep = training_keep_indices(xs, ys, ts, tu, {"mode": "time_holdout", "hold_every": 2, "hold_start": 1})
    tid, _ = assign_time_id(ts, tu)
    assert set(tid[keep]).isdisjoint({1, 3, 5}), "train rows leak held-out times"

    # --- breakthrough ---
    times = np.linspace(0, 10, 11)
    true_curve = np.clip((times - 4.0) * 0.1, 0, 1)   # crosses 0.05 at t=4.5
    pred_curve = np.clip((times - 5.0) * 0.1, 0, 1)   # crosses 0.05 at t=5.5 (delayed)
    bt = breakthrough_metrics(times, true_curve, pred_curve, threshold=0.05)
    assert abs(bt["bt_true"] - 4.5) < 1e-6, bt
    assert abs(bt["bt_pred"] - 5.5) < 1e-6, bt
    assert abs(bt["bt_abs_error"] - 1.0) < 1e-6, bt
    print(f"[breakthrough] true={bt['bt_true']:.3f} pred={bt['bt_pred']:.3f} err={bt['bt_abs_error']:.3f}")

    # --- front thickness: sharp true front vs smeared pred front ---
    arc = np.linspace(0, 5, 501)
    sharp = 0.4 * (1.0 - 1.0 / (1.0 + np.exp(-(arc - 2.5) / 0.05)))   # steep drop
    smear = 0.4 * (1.0 - 1.0 / (1.0 + np.exp(-(arc - 2.5) / 0.40)))   # gentle drop
    w_true = front_thickness_from_profile(arc, sharp)
    w_pred = front_thickness_from_profile(arc, smear)
    assert w_pred["thickness_grad"] > w_true["thickness_grad"] > 0, (w_true, w_pred)
    assert abs(w_true["front_position_mid"] - 2.5) < 0.02, w_true
    assert abs(w_pred["front_position_mid"] - 2.5) < 0.02, w_pred
    ratio = w_pred["thickness_grad"] / w_true["thickness_grad"]
    print(f"[thickness] true={w_true['thickness_grad']:.4f} pred={w_pred['thickness_grad']:.4f} ratio={ratio:.2f}")
    print("ALL SELF-TESTS PASSED")


if __name__ == "__main__":
    _self_test()
