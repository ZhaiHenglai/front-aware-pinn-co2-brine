#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Assemble and index candidate paper figures for the M0-M7 evaluation suite.

Purpose
-------
Copies selected publication-quality figures from global/front/plume/physics/
snapshots/stability/attribution into a curated figures_for_paper/ directory,
renames files using manuscript-oriented labels, and builds optional combined
panels using short labels (M0-M7 or L0-L6). This script does not recompute
metrics or predictions.

Figure style
------------
- Short labels only: M0-M7 and L0-L6.
- White background; high DPI; panel labels (a), (b), ... if requested.
- Expanded horizontal/vertical spacing to avoid overlap between panel titles,
  colour bars and legends in already-rendered images.
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import os
import shutil
from pathlib import Path
from typing import Any, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

try:
    from _nature_plot_style_M0_M7 import setup_nature_rcparams, save_pub_figure
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
        if export_svg:
            fig.savefig(str(base_without_ext) + ".svg", bbox_inches="tight", pad_inches=pad_inches)

DEFAULT_EVAL_ROOT = "./eval_M0_M7"
DEFAULT_OUT_DIR = "./eval_M0_M7/figures_for_paper"
DEFAULT_EXPERIMENTS = ",".join([f"M{i}" for i in range(8)])
DEFAULT_LEAVE_OUTS = "NONE,PLAIN_TWONET,FRONT_PLUME,PAIRGRAD,RAR,FV,COARSE_DETAIL"
DEFAULT_TIME_INDICES = "t0,early,middle,late,final"
DEFAULT_FIGURE_DPI = 600
DEFAULT_EXPORT_PDF = True
DEFAULT_EXPORT_TIFF = True
DEFAULT_EXPORT_SVG = True
EXPORT_SVG = DEFAULT_EXPORT_SVG
DEFAULT_S_IC_CO2 = 0.0
DEFAULT_SNR = 0.0

LOO_LABELS = {
    "NONE": "L0",
    "PLAIN_TWONET": "L1",
    "FRONT_PLUME": "L2",
    "PAIRGRAD": "L3",
    "RAR": "L4",
    "FV": "L5",
    "COARSE_DETAIL": "L6",
}

PREFERRED_EXTS = (".svg", ".pdf", ".png", ".tiff", ".tif")
RASTER_EXTS = (".png", ".tiff", ".tif", ".jpg", ".jpeg")


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def parse_list(s: str) -> list[str]:
    return [x.strip() for x in str(s).split(",") if x.strip()]


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


def json_dump(obj: Any, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True, ensure_ascii=False, default=str)


def find_existing(base_without_ext: Path, prefer_exts: tuple[str, ...] = PREFERRED_EXTS) -> list[Path]:
    found: list[Path] = []
    for ext in prefer_exts:
        p = base_without_ext.with_suffix(ext)
        if p.exists():
            found.append(p)
    return found


def copy_candidates(base_without_ext: Path, dest_base_without_ext: Path, description: str, category: str, manifest: list[dict[str, Any]], prefer_exts: tuple[str, ...] = PREFERRED_EXTS) -> None:
    files = find_existing(base_without_ext, prefer_exts=prefer_exts)
    if not files:
        manifest.append({
            "category": category,
            "description": description,
            "source_base": str(base_without_ext),
            "copied": 0,
            "note": "source not found",
        })
        return
    ensure_dir(dest_base_without_ext.parent)
    for src in files:
        dst = dest_base_without_ext.with_suffix(src.suffix)
        shutil.copy2(src, dst)
        manifest.append({
            "category": category,
            "description": description,
            "source": str(src),
            "destination": str(dst),
            "copied": 1,
            "extension": src.suffix,
        })


def first_raster(base_without_ext: Path) -> Optional[Path]:
    for ext in RASTER_EXTS:
        p = base_without_ext.with_suffix(ext)
        if p.exists():
            return p
    return None


def setup_rcparams(font_size: float = 7.5) -> None:
    setup_nature_rcparams(font_size=font_size, line_width=1.0)


def save_figure(fig, base_without_ext: Path, dpi: int, export_pdf: bool, export_tiff: bool) -> None:
    ensure_dir(base_without_ext.parent)
    save_pub_figure(fig, base_without_ext, dpi=dpi, export_pdf=export_pdf, export_tiff=export_tiff, export_svg=EXPORT_SVG, export_png=True, pad_inches=0.03)


def make_image_panel(
    entries: list[tuple[str, Path]],
    out_base: Path,
    ncols: int,
    title: str,
    dpi: int,
    export_pdf: bool,
    export_tiff: bool,
    panel_label: bool = True,
    title_mode: str = "short",
) -> Optional[Path]:
    existing = [(lab, p) for lab, p in entries if p.exists()]
    if not existing:
        return None
    n = len(existing)
    ncols = max(1, min(ncols, n))
    nrows = int(np.ceil(n / ncols))
    # Larger geometry to preserve embedded colour bars/legends without overlap.
    fig_w = 3.2 * ncols
    fig_h = 2.8 * nrows + (0.25 if title else 0.0)
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)
    letters = "abcdefghijklmnopqrstuvwxyz"
    for ax in axes.ravel():
        ax.axis("off")
    for i, (lab, path) in enumerate(existing):
        ax = axes.ravel()[i]
        try:
            img = mpimg.imread(str(path))
            ax.imshow(img)
        except Exception as e:
            ax.text(0.5, 0.5, f"Could not read\n{path.name}\n{e}", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        if title_mode != "none":
            ax.set_title(lab, pad=4.0)
        if panel_label:
            ax.text(0.01, 0.99, f"({letters[i]})", ha="left", va="top", transform=ax.transAxes,
                    fontsize=8, fontweight="bold", color="black",
                    bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=1.5))
    if title:
        fig.suptitle(title, y=0.995, fontsize=8.5)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.93 if title else 0.98, bottom=0.02, wspace=0.06, hspace=0.14)
    save_figure(fig, out_base, dpi=dpi, export_pdf=export_pdf, export_tiff=export_tiff)
    plt.close(fig)
    return out_base.with_suffix(".png")


def candidate_copy_plan(eval_root: Path) -> list[tuple[str, str, Path, Path]]:
    """Returns (category, description, source_base, dest_relative_base)."""
    return [
        ("global", "Global accuracy overview", eval_root / "global" / "figures" / "global_accuracy_overview", Path("main_text") / "Fig_global_accuracy_overview"),
        ("global", "Selected-time global RMSE bars", eval_root / "global" / "figures" / "selected_time_global_RMSE_bars", Path("supplementary") / "FigS_selected_time_global_RMSE_bars"),
        ("front", "Front/plume overview", eval_root / "front" / "figures" / "front_plume_overview", Path("main_text") / "Fig_front_plume_overview"),
        ("front", "Front-band RMSE vs time", eval_root / "front" / "figures" / "front_band_RMSE_wide_vs_time", Path("supplementary") / "FigS_front_band_RMSE_vs_time"),
        ("front", "Pair-gradient RMSE vs time", eval_root / "front" / "figures" / "pairgrad_RMSE_vs_time", Path("main_text") / "Fig_pairgrad_RMSE_vs_time"),
        ("front", "Speckle area ratio selected times", eval_root / "front" / "figures" / "speckle_area_ratio_dS002_selected_times", Path("main_text") / "Fig_speckle_area_ratio"),
        ("plume", "Plume IoU vs time", eval_root / "plume" / "figures" / "plume_IoU_dS005_vs_time", Path("main_text") / "Fig_plume_IoU_vs_time"),
        ("plume", "False positive plume area vs time", eval_root / "plume" / "figures" / "FP_area_ratio_dS005_vs_time", Path("supplementary") / "FigS_FP_area_ratio_vs_time"),
        ("plume", "False negative plume area vs time", eval_root / "plume" / "figures" / "FN_area_ratio_dS005_vs_time", Path("supplementary") / "FigS_FN_area_ratio_vs_time"),
        ("physics", "Physics summary overview", eval_root / "physics" / "figures" / "physics_summary_overview", Path("main_text") / "Fig_physics_summary_overview"),
        ("physics", "CO2 mass error vs time", eval_root / "physics" / "figures" / "CO2_mass_rel_error_vs_time", Path("main_text") / "Fig_CO2_mass_error_vs_time"),
        ("stability", "M-series stability panel", eval_root / "stability" / "figures" / "M_stability_summary_panel", Path("supplementary") / "FigS_M_stability_summary"),
        ("stability", "L-series stability panel", eval_root / "stability" / "figures" / "L_stability_summary_panel", Path("supplementary") / "FigS_L_stability_summary"),
        ("attribution", "Module primary improvement", eval_root / "attribution" / "figures" / "module_primary_improvement", Path("main_text") / "Fig_module_primary_improvement"),
        ("attribution", "Leave-one-out primary deterioration", eval_root / "attribution" / "figures" / "loo_primary_deterioration", Path("main_text") / "Fig_LOO_primary_deterioration"),
        ("attribution", "Module attribution heatmap", eval_root / "attribution" / "figures" / "module_attribution_heatmap", Path("main_text") / "Fig_module_attribution_heatmap"),
        ("attribution", "Leave-one-out deterioration heatmap", eval_root / "attribution" / "figures" / "leave_one_out_deterioration_heatmap", Path("main_text") / "Fig_LOO_deterioration_heatmap"),
        ("attribution", "M metric matrix normalized", eval_root / "attribution" / "figures" / "M_metric_matrix_normalized", Path("supplementary") / "FigS_M_metric_matrix_normalized"),
        ("attribution", "L metric matrix normalized", eval_root / "attribution" / "figures" / "L_metric_matrix_normalized", Path("supplementary") / "FigS_L_metric_matrix_normalized"),
    ]


def collect_snapshot_panel_entries(eval_root: Path, labels: list[str], time_tag: str, panel_name: str) -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    for lab in labels:
        base = eval_root / "snapshots" / lab / time_tag / panel_name
        raster = first_raster(base)
        if raster is not None:
            out.append((lab, raster))
    return out


def copy_snapshot_key_figures(eval_root: Path, out_dir: Path, time_indices: list[str], experiments: list[str], leave_outs: list[str], manifest: list[dict[str, Any]], export_exts: tuple[str, ...] = PREFERRED_EXTS) -> None:
    # Copy final M7 and selected LOO snapshots as manuscript candidates.
    loo_labs = [LOO_LABELS.get(x, x) for x in leave_outs]
    snapshot_specs: list[tuple[str, str, str, str]] = []
    for tag in [t for t in time_indices if t in ("late", "final")]:
        for lab in ["M7", "L0", "L3", "L5", "L6"]:
            for panel in ["diagnostic_panel", "paper_snapshot_panel"]:
                snapshot_specs.append((lab, tag, panel, f"{lab}_{tag}_{panel}"))
    for lab, tag, panel, dest_name in snapshot_specs:
        src = eval_root / "snapshots" / lab / tag / panel
        dst = out_dir / "main_text" / dest_name
        copy_candidates(src, dst, f"Snapshot {lab} {tag} {panel}", "snapshots", manifest, prefer_exts=export_exts)


def resolve_three_series_eval_root(eval_root: Path) -> Optional[Path]:
    """Find the parent tree containing M0_M7/L0_L6/train results, if present."""
    candidates = [eval_root]
    if eval_root.name in {"M0_M7", "L0_L6", "train"}:
        candidates.append(eval_root.parent)
    if eval_root.parent.name == "train":
        candidates.append(eval_root.parent.parent)
    seen: set[Path] = set()
    for cand in candidates:
        cand = cand.resolve()
        if cand in seen:
            continue
        seen.add(cand)
        if (cand / "M0_M7").exists():
            return cand
    return None


def run_nature_summary_figures(eval_root: Path, out_dir: Path, dpi: int, manifest: list[dict[str, Any]], out_subdir: str) -> None:
    """Generate the consolidated Nature-style result figures from evaluated CSV/NPZ files."""
    three_root = resolve_three_series_eval_root(eval_root)
    if three_root is None:
        manifest.append({
            "category": "nature_summary",
            "description": "Nature-style generated figure suite",
            "source_base": str(eval_root),
            "copied": 0,
            "note": "three-series root with M0_M7/ was not found",
        })
        return

    try:
        import _make_nature_results_figures_M0_M7 as nature_figs  # type: ignore
    except Exception as exc:
        manifest.append({
            "category": "nature_summary",
            "description": "Nature-style generated figure suite",
            "source_base": str(three_root),
            "copied": 0,
            "note": f"could not import helper: {exc}",
        })
        return

    nature_out = out_dir / out_subdir
    nature_figs.setup_nature_style()
    for sub in ("main_text", "supplementary", "source_data"):
        nature_figs.ensure_dir(nature_out / sub)

    generated: list[Path] = []
    generators = [
        ("M0-M7 global/front progression", nature_figs.figure_global_progression),
        ("Front/plume diagnostic panel", nature_figs.figure_front_plume_diagnostics),
        ("M7 final snapshot field panel", nature_figs.figure_m7_snapshot),
        ("LOO module attribution panel", nature_figs.figure_loo_attribution),
        ("Training-strategy comparison panel", nature_figs.figure_training_strategy),
    ]
    for desc, fn in generators:
        try:
            paths = list(fn(three_root, nature_out, dpi))
        except FileNotFoundError as exc:
            manifest.append({
                "category": "nature_summary",
                "description": desc,
                "source_base": str(three_root),
                "copied": 0,
                "note": str(exc),
            })
            continue
        except Exception as exc:
            manifest.append({
                "category": "nature_summary",
                "description": desc,
                "source_base": str(three_root),
                "copied": 0,
                "note": f"generation failed: {exc}",
            })
            continue
        generated.extend(paths)
        for path in paths:
            manifest.append({
                "category": "nature_summary",
                "description": desc,
                "source": "generated from evaluated CSV/NPZ",
                "destination": str(path),
                "copied": 1,
                "extension": path.suffix,
            })

    if generated:
        try:
            audit_path = nature_figs.audit_scaling(three_root, nature_out)
            manifest.append({
                "category": "nature_summary",
                "description": "Figure scaling audit",
                "destination": str(audit_path),
                "copied": 1,
                "extension": audit_path.suffix,
            })
        except Exception as exc:
            manifest.append({
                "category": "nature_summary",
                "description": "Figure scaling audit",
                "source_base": str(three_root),
                "copied": 0,
                "note": f"audit failed: {exc}",
            })
        nature_figs.write_readme(nature_out, three_root, generated)


def main() -> int:
    ap = argparse.ArgumentParser(description="Assemble curated paper figures for M0-M7 evaluation.")
    ap.add_argument("--eval-root", default=DEFAULT_EVAL_ROOT)
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--experiments", default=DEFAULT_EXPERIMENTS)
    ap.add_argument("--leave-outs", default=DEFAULT_LEAVE_OUTS)
    ap.add_argument("--time-indices", default=DEFAULT_TIME_INDICES)
    ap.add_argument("--figure-dpi", type=int, default=DEFAULT_FIGURE_DPI)
    ap.add_argument("--export-pdf", type=int, default=int(DEFAULT_EXPORT_PDF))
    ap.add_argument("--export-tiff", type=int, default=int(DEFAULT_EXPORT_TIFF))
    ap.add_argument("--export-svg", type=int, default=int(DEFAULT_EXPORT_SVG),
                    help="Export editable SVG files for assembled panels.")
    ap.add_argument("--s-ic-co2", type=float, default=DEFAULT_S_IC_CO2)
    ap.add_argument("--Snr", type=float, default=DEFAULT_SNR)
    ap.add_argument("--make-snapshot-panels", type=int, default=1)
    ap.add_argument("--snapshot-panel-time", default="final")
    ap.add_argument("--make-nature-summary-figures", type=int, default=1,
                    help="Generate consolidated Nature-style summary panels when a three-series eval root is available.")
    ap.add_argument("--nature-out-subdir", default="nature_figures")
    args = ap.parse_args()
    global EXPORT_SVG
    EXPORT_SVG = bool(args.export_svg)

    eval_root = Path(args.eval_root)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    ensure_dir(out_dir / "main_text")
    ensure_dir(out_dir / "supplementary")
    ensure_dir(out_dir / "source_copies")
    ensure_dir(out_dir / "combined_panels")

    setup_rcparams()
    experiments = parse_list(args.experiments)
    leave_outs = [x.upper() for x in parse_list(args.leave_outs)]
    time_indices = parse_list(args.time_indices)

    manifest: list[dict[str, Any]] = []

    # Copy summary figures from each evaluation family.
    for category, desc, src, dst_rel in candidate_copy_plan(eval_root):
        copy_candidates(src, out_dir / dst_rel, desc, category, manifest)

    # Copy selected snapshot figures.
    copy_snapshot_key_figures(eval_root, out_dir, time_indices, experiments, leave_outs, manifest)

    # Build combined snapshot panels from existing diagnostic panels if available.
    if int(args.make_snapshot_panels):
        time_tag = str(args.snapshot_panel_time)
        m_entries = collect_snapshot_panel_entries(eval_root, experiments, time_tag, "diagnostic_panel")
        out = make_image_panel(
            m_entries,
            out_dir / "combined_panels" / f"Fig_M0_M7_diagnostic_grid_{time_tag}",
            ncols=3,
            title=f"M0-M7 diagnostic panels ({time_tag})",
            dpi=int(args.figure_dpi),
            export_pdf=bool(args.export_pdf),
            export_tiff=bool(args.export_tiff),
        )
        manifest.append({"category": "combined_panel", "description": f"M0-M7 diagnostic grid {time_tag}", "destination": str(out) if out else None, "copied": int(out is not None)})

        loo_labels = [LOO_LABELS.get(x, x) for x in leave_outs]
        l_entries = collect_snapshot_panel_entries(eval_root, loo_labels, time_tag, "diagnostic_panel")
        out = make_image_panel(
            l_entries,
            out_dir / "combined_panels" / f"Fig_L0_L6_diagnostic_grid_{time_tag}",
            ncols=3,
            title=f"L0-L6 leave-one-out diagnostic panels ({time_tag})",
            dpi=int(args.figure_dpi),
            export_pdf=bool(args.export_pdf),
            export_tiff=bool(args.export_tiff),
        )
        manifest.append({"category": "combined_panel", "description": f"L0-L6 diagnostic grid {time_tag}", "destination": str(out) if out else None, "copied": int(out is not None)})

    if int(args.make_nature_summary_figures):
        run_nature_summary_figures(eval_root, out_dir, int(args.figure_dpi), manifest, str(args.nature_out_subdir))

    write_csv(manifest, out_dir / "figure_manifest.csv")
    cfg = {
        "created_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "eval_root": str(eval_root),
        "out_dir": str(out_dir),
        "experiments": experiments,
        "leave_outs": leave_outs,
        "time_indices": time_indices,
        "S_IC_CO2": args.s_ic_co2,
        "Snr": args.Snr,
        "figure_dpi": args.figure_dpi,
        "export_pdf": bool(args.export_pdf),
        "export_tiff": bool(args.export_tiff),
        "export_svg": bool(args.export_svg),
        "make_nature_summary_figures": bool(args.make_nature_summary_figures),
        "nature_out_subdir": str(args.nature_out_subdir),
        "short_label_policy": "Use M0-M7 and L0-L6 only; do not expose long checkpoint/run IDs in figure panels.",
        "spacing_policy": "Combined panels use expanded wspace/hspace and no external legend to avoid overlap with embedded colour bars.",
    }
    json_dump(cfg, out_dir / "figures_for_paper_config.json")
    readme = f"""# Figures for paper\n\nCreated: {cfg['created_at']}\n\nThis directory contains curated copies and combined panels from the full M0-M7 evaluation tree. Figures use short labels only. Raw Sco2 is shown in field images; excess saturation dS=S-S_IC_CO2 may be used by snapshot masks/leading-front diagnostics. S_IC_CO2={args.s_ic_co2}; Snr={args.Snr}.\n\nUse `figure_manifest.csv` to trace each copied figure back to its source path.\n"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")

    n_copied = sum(1 for r in manifest if r.get("copied") == 1)
    n_missing = sum(1 for r in manifest if r.get("copied") == 0)
    print(f"[figures] output: {out_dir}")
    print(f"[figures] copied/generated: {n_copied}")
    print(f"[figures] missing/skipped: {n_missing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
