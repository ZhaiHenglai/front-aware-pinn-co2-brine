"""Supplemental profile and producer-probe diagnostics; these are in-sample."""
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from _nature_plot_style_M0_M7 import save_pub_figure

def write_csv(path, rows):
    if not rows:
        raise ValueError(f"No diagnostic rows for {path}")
    with Path(path).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

def first_arrival(times, values, threshold=0.05):
    """First sampled threshold crossing; no arrival is right-censored, not zero."""
    ids = np.flatnonzero(np.asarray(values) >= threshold)
    return float(times[ids[0]]) if ids.size else None

def export_diagnostics(out_base, arrays, times, model, cfg, device, args):
    from evaluate_global_accuracy_M0_M7_L6_dS_shortlabels import predict_model, to_np_1d
    out = Path(out_base)
    profiles, metrics = [], []
    for gridfile in sorted(out.glob("*/snapshot_grid.npz")):
        with np.load(gridfile) as z:
            X, Y = z["X"], z["Y"]
            L = float(cfg["L_ref"])
            fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
            for angle in np.linspace(10, 80, 9):
                theta = np.deg2rad(angle)
                direction = np.array([np.cos(theta), np.sin(theta)])
                origin = np.asarray(cfg["inj_center"], dtype=float)
                end = np.min((L - origin) / direction) * 0.97
                distance = np.linspace(float(cfg["r_well"]) * 1.1, end, 512)
                xy = origin + distance[:, None] * direction
                values = {}
                for key in ("S_true", "S_pred"):
                    interp = RegularGridInterpolator((Y[:, 0], X[0, :]), z[key],
                                                     bounds_error=False, fill_value=np.nan)
                    values[key] = interp(xy[:, ::-1])
                true, pred = values["S_true"], values["S_pred"]
                good = np.isfinite(true) & np.isfinite(pred)
                if good.sum() < 2:
                    raise ValueError("Ray has insufficient reference/prediction support")
                d, true, pred = distance[good], true[good], pred[good]
                # Farthest crossing of fixed S=0.175. Flag disconnected support.
                front = {}
                for key, v in (("true", true), ("pred", pred)):
                    inside = np.flatnonzero(v >= 0.175)
                    front[key] = float(d[inside[-1]]) if inside.size else np.nan
                metrics.append(dict(time_tag=gridfile.parent.name,
                    time_s=float(z["time_seconds"]), angle_deg=float(angle),
                    S_rmse=float(np.sqrt(np.mean((pred-true)**2))),
                    front_true_m=front["true"], front_pred_m=front["pred"],
                    front_abs_error_m=abs(front["pred"]-front["true"]),
                    front_true_present=bool(np.any(true >= 0.175)),
                    front_pred_present=bool(np.any(pred >= 0.175)),
                    true_segments=int(np.sum(np.diff(np.r_[False, true >= 0.175].astype(int)) == 1)),
                    pred_segments=int(np.sum(np.diff(np.r_[False, pred >= 0.175].astype(int)) == 1))))
                for di, tv, pv in zip(d, true, pred):
                    profiles.append(dict(time_tag=gridfile.parent.name, angle_deg=float(angle),
                                         distance_m=float(di), S_true=float(tv), S_pred=float(pv)))
                color = plt.get_cmap("viridis")((angle-10)/70)
                ax.plot(d, true, color=color, label=f"{angle:g} deg reference")
                ax.plot(d, pred, "--", color=color)
            ax.set(xlabel="Distance from injector (m)", ylabel="CO2 saturation",
                   title=f"{gridfile.parent.name}: solid reference, dashed PINN")
            ax.legend(fontsize=6, ncol=3)
            save_pub_figure(fig, out / f"multiray_profiles_{gridfile.parent.name}",
                            dpi=args.figure_dpi, export_tiff=False)
            plt.close(fig)
    write_csv(out / "multiray_front_by_ray.csv", metrics)
    write_csv(out / "multiray_front_profiles.csv", profiles)
    # Full time series, evaluated on the same native producer-ring points.
    ts = to_np_1d(arrays["t"])
    if np.any(ts[1:] < ts[:-1]):
        raise ValueError("Producer diagnostic requires time-grouped rows.")
    x, y = to_np_1d(arrays["x"]), to_np_1d(arrays["y"])
    s = to_np_1d(arrays["Sco2"])
    ox, oy = cfg["out_center"]
    radius = float(cfg["r_well"])
    curves = []
    for t in times:
        lo, hi = np.searchsorted(ts, t, side="left"), np.searchsorted(ts, t, side="right")
        r2 = (x[lo:hi]-ox)**2 + (y[lo:hi]-oy)**2
        indices = lo + np.flatnonzero((r2 > radius**2) & (r2 <= (1.5*radius)**2))
        if not indices.size:
            raise ValueError(f"No native points in producer probe annulus at t={t}")
        _, pred = predict_model(model, x[indices], y[indices], ts[indices], cfg,
                                device, args.infer_batch_size)
        if not np.isfinite(pred).all():
            raise ValueError("Nonfinite producer prediction")
        curves.append(dict(time_s=float(t), S_true=float(np.mean(s[indices])),
                           S_pred=float(np.mean(pred)), n_points=int(indices.size)))
    write_csv(out / "breakthrough_curves.csv", curves)
    true_arrival = first_arrival(times, [r["S_true"] for r in curves])
    pred_arrival = first_arrival(times, [r["S_pred"] for r in curves])
    write_csv(out / "breakthrough_time.csv", [dict(
        threshold=0.05, reference_first_sample_s=true_arrival, predicted_first_sample_s=pred_arrival,
        reference_censored=true_arrival is None, predicted_censored=pred_arrival is None,
        absolute_error_s=abs(pred_arrival-true_arrival) if None not in (true_arrival, pred_arrival) else None,
        probe_inner_radius_m=radius, probe_outer_radius_m=1.5*radius,
        weighting="native_point_mean", evaluation_scope="in_sample")])
    fig, ax = plt.subplots(constrained_layout=True)
    for key, label in (("S_true", "Reference"), ("S_pred", "PINN")):
        ax.plot(np.asarray(times)/86400, [r[key] for r in curves], label=label)
    ax.axhline(0.05, color="grey", linestyle=":")
    ax.set(xlabel="Time (days)", ylabel="Producer-probe mean CO2 saturation")
    ax.legend()
    save_pub_figure(fig, out / "breakthrough_curves", dpi=args.figure_dpi, export_tiff=False)
    plt.close(fig)
