#!/usr/bin/env python3
"""Extract reference-only H/K fields at K's common physical snapshot times."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import torch
from scipy.interpolate import NearestNDInterpolator


def as_numpy(value):
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def unique_average(x, y, values):
    order = np.lexsort((y, x))
    xs, ys, vs = x[order], y[order], values[order]
    starts = np.r_[0, np.flatnonzero((np.diff(xs) != 0) | (np.diff(ys) != 0)) + 1]
    counts = np.diff(np.r_[starts, len(xs)])
    return xs[starts], ys[starts], np.add.reduceat(vs, starts) / counts


def interpolate(x, y, values, X, Y, mask):
    good = np.isfinite(x) & np.isfinite(y) & np.isfinite(values)
    if not np.all(good):
        raise ValueError("reference snapshot contains nonfinite input values")
    xu, yu, vu = unique_average(x, y, values)
    if len(xu) < 3:
        raise ValueError("fewer than three unique spatial points")
    z = np.ma.asarray(mtri.LinearTriInterpolator(mtri.Triangulation(xu, yu), vu)(X, Y)).filled(np.nan)
    if np.isnan(z).any():
        near = NearestNDInterpolator(np.column_stack((xu, yu)), vu)(X, Y)
        z = np.where(np.isfinite(z), z, near)
    z = np.asarray(z, dtype=np.float64)
    z[mask] = np.nan
    return z


def nearest_time(times, target):
    pos = int(np.searchsorted(times, target))
    candidates = [max(0, min(len(times) - 1, pos)), max(0, min(len(times) - 1, pos - 1))]
    return float(min((times[i] for i in candidates), key=lambda x: (abs(x - target), x)))


def extract_case(label, data_path, targets, tags, output, X, Y, mask):
    print(f"Loading {label}: {data_path}", flush=True)
    pack = torch.load(data_path, map_location="cpu", weights_only=False)
    if not isinstance(pack, dict) or not isinstance(pack.get("arrays"), dict):
        raise ValueError(f"{data_path} is not a dataset pack with arrays")
    arrays = pack["arrays"]
    required = ("x", "y", "t", "p", "Sco2")
    missing = [key for key in required if key not in arrays]
    if missing: raise ValueError(f"{label} dataset missing {missing}")
    x = as_numpy(arrays["x"]).reshape(-1)
    y = as_numpy(arrays["y"]).reshape(-1)
    t = as_numpy(arrays["t"]).reshape(-1)
    p = as_numpy(arrays["p"]).reshape(-1)
    s = as_numpy(arrays["Sco2"]).reshape(-1)
    if not (len(x) == len(y) == len(t) == len(p) == len(s)):
        raise ValueError(f"{label} row-aligned arrays differ in length")
    times = np.unique(t.astype(np.float64, copy=False))
    rows = []
    case_dir = output / label
    case_dir.mkdir(parents=True, exist_ok=True)
    for tag, target in zip(tags, targets):
        actual = nearest_time(times, target)
        tol = max(1e-9, 1e-10 * max(1.0, abs(actual)))
        take = np.isclose(t, actual, rtol=0.0, atol=tol)
        if not np.any(take): raise RuntimeError(f"{label}: no rows at nearest time {actual}")
        sg = interpolate(x[take], y[take], np.clip(s[take], 0, 1), X, Y, mask)
        pg = interpolate(x[take], y[take], p[take], X, Y, mask)
        np.savez_compressed(case_dir / f"reference_{tag}.npz", X=X, Y=Y, saturation=sg, pressure=pg,
                            target_time_s=target, actual_time_s=actual, time_offset_s=actual-target)
        rows.append({"case": label, "tag": tag, "target_time_s": target, "actual_time_s": actual,
                     "offset_s": actual-target, "absolute_offset_s": abs(actual-target), "source_rows": int(take.sum())})
        print(f"{label} {tag}: target={target:.9g} actual={actual:.9g} rows={take.sum()}", flush=True)
    del pack, arrays, x, y, t, p, s
    gc.collect()
    return rows


def save_figure(fig, stem: Path, dpi: int):
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(stem.with_suffix("." + suffix), dpi=dpi if suffix == "png" else None, bbox_inches="tight")
    plt.close(fig)


def comparison_figures(output, tags, rows, dpi):
    for quantity, cmap, unit_scale, label in (
        ("saturation", "viridis", 1.0, r"$S_{CO_2}$"),
        ("pressure", "cividis", 1e-6, "Pressure (MPa)"),
    ):
        for tag in tags:
            h = np.load(output / "H" / f"reference_{tag}.npz")
            k = np.load(output / "K" / f"reference_{tag}.npz")
            ha, ka = h[quantity] * unit_scale, k[quantity] * unit_scale
            both = np.r_[ha[np.isfinite(ha)], ka[np.isfinite(ka)]]
            if quantity == "saturation": vmin, vmax = 0.0, 1.0
            else: vmin, vmax = np.percentile(both, [1, 99])
            diff = ka - ha
            lim = np.nanmax(np.abs(diff)) or 1.0
            fig, axes = plt.subplots(1, 3, figsize=(9.0, 2.8), constrained_layout=True)
            for ax, arr, title in zip(axes, (ha, ka, diff), ("H reference", "K reference", "K - H")):
                im = ax.pcolormesh(h["X"], h["Y"], arr, shading="auto", cmap=("coolwarm" if title == "K - H" else cmap),
                                   vmin=(-lim if title == "K - H" else vmin), vmax=(lim if title == "K - H" else vmax))
                ax.set(title=title, xlabel="x (m)", ylabel="y (m)", aspect="equal")
                fig.colorbar(im, ax=ax, label=(f"Difference in {label}" if title == "K - H" else label))
            htime = float(h["actual_time_s"]); ktime = float(k["actual_time_s"])
            fig.suptitle(f"Matched physical time: target {float(h['target_time_s']):.3f} s; H {htime:.3f} s; K {ktime:.3f} s")
            save_figure(fig, output / f"HK_reference_{quantity}_{tag}", dpi)


def permeability_figure(field_path, output, dpi, h_value):
    with np.load(field_path) as field:
        x, y, kval = field["x_grid"], field["y_grid"], field["permeability"]
        field_id = str(field["field_id"])
    fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.8), constrained_layout=True)
    h = np.full_like(kval, h_value)
    lo, hi = min(np.nanmin(h), np.nanmin(kval)), max(np.nanmax(h), np.nanmax(kval))
    for ax, val, title in zip(axes, (h, kval), ("H declared homogeneous", "K fixed heterogeneous")):
        im = ax.pcolormesh(x, y, val, shading="auto", cmap="magma", vmin=lo, vmax=hi)
        ax.set(title=title, xlabel="x (m)", ylabel="y (m)", aspect="equal")
        fig.colorbar(im, ax=ax, label=r"Permeability ($m^2$)")
    save_figure(fig, output / "HK_permeability_comparison", dpi)
    return field_id, float(np.nanmin(kval)), float(np.nanmax(kval))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--h-data", required=True, type=Path)
    ap.add_argument("--k-data", required=True, type=Path)
    ap.add_argument("--k-field", required=True, type=Path)
    ap.add_argument("--k-snapshot-metadata", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--grid-n", type=int, default=500)
    ap.add_argument("--dpi", type=int, default=600)
    ap.add_argument("--domain-length", type=float, default=5.0)
    ap.add_argument("--well-radius", type=float, default=0.5)
    ap.add_argument("--h-permeability", type=float, default=1e-14)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    meta = json.loads(args.k_snapshot_metadata.read_text(encoding="utf-8"))
    times_meta = meta.get("times", [])
    tags = [str(x["tag"]) for x in times_meta]
    targets = [float(x["time_seconds"]) for x in times_meta]
    if len(tags) != 5 or tags != ["t0", "early", "middle", "late", "final"]:
        raise ValueError(f"unexpected canonical K snapshot tags: {tags}")
    if max(targets) > 30986.943359375 + 1e-6:
        raise ValueError("K target times exceed the frozen K reference endpoint")
    axis = np.linspace(0, args.domain_length, args.grid_n)
    X, Y = np.meshgrid(axis, axis)
    mask = ((X**2 + Y**2) <= args.well_radius**2) | (((X-args.domain_length)**2 + (Y-args.domain_length)**2) <= args.well_radius**2)
    rows = []
    rows += extract_case("H", args.h_data.resolve(), targets, tags, args.output, X, Y, mask)
    rows += extract_case("K", args.k_data.resolve(), targets, tags, args.output, X, Y, mask)
    with (args.output / "common_time_mapping.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (args.output / "common_time_mapping.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    comparison_figures(args.output, tags, rows, args.dpi)
    field_id, kmin, kmax = permeability_figure(args.k_field, args.output, args.dpi, args.h_permeability)
    report = {
        "status": "completed", "scope": "reference-only H/K common-time extraction; no PINN inference or training",
        "canonical_time_source": str(args.k_snapshot_metadata.resolve()), "target_times_s": targets,
        "common_endpoint_s": max(targets), "k_extrapolation": False,
        "h_permeability_m2": args.h_permeability,
        "h_permeability_evidence_status": "declared_value_pending_original_simulator_input_evidence",
        "k_permeability_field_id": field_id, "k_permeability_min_m2": kmin, "k_permeability_max_m2": kmax,
        "well_mask": {"radius_m": args.well_radius, "centres_m": [[0, 0], [args.domain_length, args.domain_length]]},
        "mapping": rows,
    }
    (args.output / "HK_COMMON_TIME_REPORT.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    files = sorted(p for p in args.output.rglob("*") if p.is_file() and p.name != "SHA256SUMS")
    (args.output / "SHA256SUMS").write_text("".join(f"{sha256(p)}  {p.relative_to(args.output)}\n" for p in files), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
