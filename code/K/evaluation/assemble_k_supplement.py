#!/usr/bin/env python3
"""Validate all 20 supplementary runs, then assemble cross-seed evidence and figures."""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "Results")]
from configuration_k import all_formal_tasks, CANONICAL_METRICS
from permeability_field_k import file_sha256
from _nature_plot_style_M0_M7 import save_pub_figure
from aggregate_k_results import main as aggregate_formal

def verify(evaluation_root):
    tasks = all_formal_tasks()
    expected = {}
    for task in tasks:
        out = evaluation_root / "by_run" / task.role / task.run_id
        record = json.loads((out / "supplement_summary.json").read_text())
        if record["status"] != "completed" or record["task"] != task.as_dict():
            raise ValueError(f"Incomplete or mismatched supplemental task: {task.run_id}")
        if record["code_manifest_sha256"] != file_sha256(ROOT / "manifests/CODE_SHA256SUMS"):
            raise ValueError(f"Supplement code version mismatch: {task.run_id}")
        for relative, sha in record["outputs"].items():
            p = out / relative
            if not p.is_relative_to(out) or file_sha256(p) != sha:
                raise ValueError(f"Supplement artifact changed: {p}")
        grids = sorted(out.glob("snapshots/*/*/snapshot_grid.npz"))
        if len(grids) != 5 or {p.parent.name for p in grids} != {"t0","early","middle","late","final"}:
            raise ValueError(f"Expected five spatial snapshots: {task.run_id}")
        for grid in grids:
            for stem in ("S_true","S_pred","S_error","p_true","p_pred","p_error",
                         "diagnostic_panel","paper_snapshot_panel"):
                for ext in ("png","pdf","svg"):
                    p = grid.parent / f"{stem}.{ext}"
                    if str(p.relative_to(out)) not in record["outputs"]:
                        raise ValueError(f"Unrecorded spatial image: {p}")
        for filename in ("summary_training_stability.csv", "loss_points.csv"):
            if not (out / "stability" / filename).is_file():
                raise ValueError(f"Missing stability file: {task.run_id}/{filename}")
        for filename in ("multiray_front_by_ray.csv", "breakthrough_time.csv"):
            if len(list(out.glob("snapshots/*/" + filename))) != 1:
                raise ValueError(f"Missing diagnostic: {task.run_id}/{filename}")
        expected[task.run_id] = grids
    return expected

def save(fig, base):
    save_pub_figure(fig, base, dpi=300, export_tiff=False)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--evaluation-root", type=Path, required=True)
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()
    eroot = args.evaluation_root.resolve()
    grids = verify(eroot)
    if args.verify_only:
        print("PASS: 20 tasks, 100 spatial snapshots, all recorded checksums verified")
        return
    dest = eroot / "comparison"
    if dest.exists():
        raise ValueError(f"Comparison output already exists: {dest}")
    # Preserve the existing five-metric validator and evidence requirements.
    aggregate_formal(["--output-root", str(ROOT), "--evaluation-root", str(eroot),
                     "--data", str(ROOT / "data/tables_cache_0_775_step1.pt"),
                     "--field", str(ROOT / "permeability/K_field_unique_grid.npz"),
                     "--field-metadata", str(ROOT / "permeability/K_field_unique_grid.json"),
                     "--reference-provenance", str(ROOT / "manifests/k_reference_provenance.json")])
    dest.mkdir()
    tasks = all_formal_tasks()
    roles = list(dict.fromkeys(t.role for t in tasks))
    # Aggregate every numeric diagnostic, keeping time/ray/stage axes separate.
    tables = {}
    for task in tasks:
        out = eroot / "by_run" / task.role / task.run_id
        for p in sorted(out.rglob("*.csv")):
            if "checkpoint_view" in p.parts:
                continue
            try:
                df = pd.read_csv(p)
            except pd.errors.EmptyDataError:
                continue
            if df.empty:
                continue
            df["model_role"], df["training_seed"] = task.role, task.seed
            tables.setdefault(p.name, []).append(df)
    for name, pieces in tables.items():
        df = pd.concat(pieces, ignore_index=True)
        stem = Path(name).stem
        df.to_csv(dest / f"{stem}_multiseed_long.csv", index=False)
        axes = [c for c in ("time_index","time_s","time_seconds","time_tag","angle_deg",
                           "distance_m","iteration","it","x","kind","component","stage") if c in df]
        exclude = {"seed","training_seed","array_index","job_id"} | set(axes)
        numeric = [c for c in df.select_dtypes(include=np.number) if c not in exclude]
        if numeric:
            grouped = df.groupby(["model_role"] + axes, dropna=False)[numeric].agg(["count","mean","std"])
            grouped.columns = ["_".join(c) for c in grouped.columns]
            grouped.reset_index().to_csv(dest / f"{stem}_multiseed_summary.csv", index=False)

        curve_columns = {
            "by_time_global_accuracy.csv": ("time_s", ["S_rmse", "p_phys_RMSE_MPa"]),
            "by_time_front_resolution.csv": ("time_s", ["front_band_wide_005_030_rmse"]),
            "by_time_plume_support.csv": ("time_s", ["plume_IoU_dS005", "plume_IoU_S005"]),
            "pairgrad_metrics.csv": ("time_s", ["pairgrad_RMSE"]),
            "mass_metrics.csv": ("time_s", ["CO2_mass_rel_error"]),
            "breakthrough_curves.csv": ("time_s", ["S_true", "S_pred"]),
            "loss_points.csv": ("x", ["loss", "pde", "fv", "dataP", "dataS", "max_alloc", "max_reserv"]),
        }
        if name in curve_columns:
            axis, columns = curve_columns[name]
            for column in columns:
                if axis not in df or column not in df:
                    continue
                fig, ax = plt.subplots(figsize=(8,5), constrained_layout=True)
                for role in roles:
                    series = df[df.model_role == role].groupby(axis)[column].agg(["mean","std","count"])
                    if series.empty:
                        continue
                    x = series.index.to_numpy(dtype=float)
                    if axis == "time_s": x = x / 86400.
                    mean = series["mean"].to_numpy(dtype=float)
                    std = series["std"].to_numpy(dtype=float)
                    line, = ax.plot(x, mean, label=role)
                    ax.fill_between(x, mean-std, mean+std, color=line.get_color(), alpha=.13)
                ax.set(xlabel="Time (days)" if axis=="time_s" else "Training iteration",
                       ylabel=column, title="Mean and sample SD across four seeds")
                ax.legend(fontsize=7)
                save(fig, dest / f"{stem}_{column}_comparison")

    raw = pd.read_csv(eroot / "heterogeneous_metrics_by_seed.csv")
    paired = []
    for metric in CANONICAL_METRICS:
        wide = raw.pivot(index="seed", columns="model_role", values=metric)
        for comparator in roles:
            if comparator == "complete":
                continue
            for seed, row in wide.iterrows():
                delta = row[comparator] - row["complete"]
                paired.append(dict(metric=metric, comparator=comparator, seed=int(seed),
                    complete=float(row["complete"]), comparator_value=float(row[comparator]),
                    deterioration_absolute=float(delta),
                    deterioration_percent=float(100*delta/row["complete"]) if row["complete"] != 0 else np.nan))
    pd.DataFrame(paired).to_csv(dest / "paired_ablation_by_seed.csv", index=False)
    fig, axes = plt.subplots(2,3,figsize=(14,8),constrained_layout=True)
    for ax, metric in zip(axes.flat, CANONICAL_METRICS):
        data = raw.groupby("model_role")[metric].agg(["mean","std"]).reindex(roles)
        ax.bar(np.arange(len(roles)), data["mean"], yerr=data["std"], capsize=3)
        for j, role in enumerate(roles):
            ax.scatter(np.full(4,j), raw.loc[raw.model_role==role,metric], color="black", s=12, zorder=3)
        ax.set_xticks(np.arange(len(roles)), [r.replace("_","\n") for r in roles], fontsize=7)
        ax.set_title(metric)
    axes.flat[-1].axis("off")
    save(fig, dest / "five_metric_comparison_mean_std")
    # Shared colour ranges across all roles/seeds/times, signed errors centred at zero.
    ranges = {"pmin": np.inf, "pmax": -np.inf, "perr": 0., "serr": 0.}
    for paths in grids.values():
        for p in paths:
            with np.load(p) as z:
                ranges["pmin"] = min(ranges["pmin"], np.nanmin(z["p_true"]), np.nanmin(z["p_pred"]))
                ranges["pmax"] = max(ranges["pmax"], np.nanmax(z["p_true"]), np.nanmax(z["p_pred"]))
                ranges["perr"] = max(ranges["perr"], np.nanmax(np.abs(z["p_error"])))
                ranges["serr"] = max(ranges["serr"], np.nanmax(np.abs(z["S_error"])))
    (dest / "shared_colour_scales.json").write_text(json.dumps(ranges, indent=2))
    fields = ["S_true","S_pred","S_error","p_true","p_pred","p_error"]
    for seed in (0,1,2,42):
        for tag in ("t0","early","middle","late","final"):
            fig, axes = plt.subplots(5,6,figsize=(18,14),constrained_layout=True)
            for i, role in enumerate(roles):
                p = next(p for p in grids[f"{role}_seed{seed}"] if p.parent.name == tag)
                with np.load(p) as z:
                    extent=[float(z["X"].min()),float(z["X"].max()),float(z["Y"].min()),float(z["Y"].max())]
                    for j, field in enumerate(fields):
                        v = z[field]
                        if field.startswith("p"):
                            v = v * 1e-6
                            lo,hi = ranges["pmin"]*1e-6,ranges["pmax"]*1e-6
                            if field.endswith("error"):
                                hi=max(ranges["perr"]*1e-6,1e-12); lo=-hi
                        else:
                            lo,hi=0.,0.8
                            if field.endswith("error"):
                                hi=max(ranges["serr"],1e-12); lo=-hi
                        ax=axes[i,j]
                        im=ax.imshow(v,origin="lower",extent=extent,vmin=lo,vmax=hi,
                                     cmap="RdBu_r" if field.endswith("error") else "viridis",
                                     interpolation="nearest")
                        if i==0: ax.set_title(field + (" (MPa)" if field.startswith("p") else ""))
                        if j==0: ax.set_ylabel(role.replace("_","\n")+"\ny (m)")
                        if i==4: ax.set_xlabel("x (m)")
                        fig.colorbar(im, ax=ax, shrink=.65)
            fig.suptitle(f"Configuration K: seed {seed}, {tag}; shared scales across all snapshots")
            save(fig, dest / f"field_comparison_seed{seed}_{tag}")
    manifest = {str(p.relative_to(eroot)): file_sha256(p)
                for p in sorted(dest.rglob("*")) if p.is_file()}
    (eroot / "supplement_aggregation_summary.json").write_text(json.dumps(
        {"status":"completed","tasks":20,"spatial_snapshots":100,"comparison_files":manifest,
         "scope":"in-sample K checkpoints; no held-out or independent BL training performed"},indent=2))
    print("Completed formal aggregation and supplemental comparisons")

if __name__ == "__main__":
    main()
