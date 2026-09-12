#!/usr/bin/env python3
"""Strict, read-only verification of the completed K supplemental evaluation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


EXPECTED_DATA = "2132a9be710e2f257ea397cd080654e689093d9c561be3edb5be1e471283e761"
EXPECTED_FIELD = "fd7c4ef13b7d0cd3d30d32b3f24fe6819f91e62d4657fe01769d3830a75f0a47"
EXPECTED_FIELD_ID = "K775_fixed_permgrid128_2132a9be710e"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load(path: Path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-root", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    root = args.evaluation_root.resolve()
    failures: list[str] = []

    summaries = sorted((root / "by_run").rglob("supplement_summary.json"))
    eval_summaries = sorted((root / "by_run").rglob("evaluation_summary.json"))
    manifests = sorted((root / "by_run").rglob("evaluation_manifest.json"))
    grids = sorted((root / "by_run").rglob("snapshot_grid.npz"))
    if len(summaries) != 20: failures.append(f"supplement summaries={len(summaries)}, expected 20")
    if len(eval_summaries) != 20: failures.append(f"evaluation summaries={len(eval_summaries)}, expected 20")
    if len(manifests) != 20: failures.append(f"evaluation manifests={len(manifests)}, expected 20")
    if len(grids) != 100: failures.append(f"snapshot grids={len(grids)}, expected 100")

    run_ids: set[str] = set()
    output_files = 0
    output_bytes = 0
    for summary_path in summaries:
        record = load(summary_path)
        run_id = str(record.get("task", {}).get("run_id", ""))
        run_ids.add(run_id)
        if record.get("status") != "completed": failures.append(f"{run_id}: supplement status is not completed")
        if record.get("failures") != {}: failures.append(f"{run_id}: nonempty failures")
        base = summary_path.parent
        for relative, expected in record.get("outputs", {}).items():
            path = base / relative
            output_files += 1
            if not path.is_file():
                failures.append(f"missing: {path}")
                continue
            output_bytes += path.stat().st_size
            actual = sha256(path)
            if actual != expected: failures.append(f"sha256 mismatch: {path}")

    data_shas, field_shas, field_ids, pc_values = set(), set(), set(), set()
    for path in manifests:
        record = load(path)
        data_shas.add(record.get("dataset_sha256"))
        field_shas.add(record.get("permeability_field_sha256"))
        field_ids.add(record.get("permeability_field_id"))
        cfg = record.get("checkpoint_config", {})
        pc_values.add(bool(cfg.get("pc_enable", cfg.get("capillary_pressure_enabled", False))))
    if data_shas != {EXPECTED_DATA}: failures.append(f"dataset SHA set={sorted(map(str, data_shas))}")
    if field_shas != {EXPECTED_FIELD}: failures.append(f"field SHA set={sorted(map(str, field_shas))}")
    if field_ids != {EXPECTED_FIELD_ID}: failures.append(f"field ID set={sorted(map(str, field_ids))}")
    if pc_values != {False}: failures.append(f"Pc enabled set={sorted(pc_values)}")
    if len(run_ids) != 20 or "" in run_ids: failures.append(f"unique valid run IDs={len(run_ids - {''})}, expected 20")

    agg_path = root / "supplement_aggregation_summary.json"
    if not agg_path.is_file():
        failures.append("missing supplement_aggregation_summary.json")
        agg = {}
    else:
        agg = load(agg_path)
        if agg.get("status") != "completed": failures.append("aggregation status is not completed")
        if agg.get("tasks") != 20: failures.append(f"aggregation tasks={agg.get('tasks')}")
        if agg.get("spatial_snapshots") != 100: failures.append(f"aggregation snapshots={agg.get('spatial_snapshots')}")

    csv_path = root / "heterogeneous_metrics_by_seed.csv"
    rows = []
    if csv_path.is_file():
        with csv_path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
    if len(rows) != 20: failures.append(f"metric rows={len(rows)}, expected 20")
    roles = {row.get("model_role") for row in rows}
    seeds = {row.get("seed") for row in rows}
    rar_global = {row.get("rar_global_select_count") for row in rows}
    expected_roles = {"baseline", "complete", "no_front_supervision", "no_local_fv", "no_representation"}
    if roles != expected_roles: failures.append(f"model roles={sorted(map(str, roles))}")
    if seeds != {"0", "1", "2", "42"}: failures.append(f"seeds={sorted(map(str, seeds))}")
    if rar_global != {"0"}: failures.append(f"rar_global_select_count={sorted(map(str, rar_global))}, expected 0")
    for required in ("heterogeneous_metrics_summary.csv", "aggregation_summary.json", "formal_artifact_manifest.json"):
        if not (root / required).is_file(): failures.append(f"missing {required}")
    if not (root / "comparison").is_dir(): failures.append("missing comparison directory")

    report = {
        "status": "completed" if not failures else "failed",
        "evaluation_root": str(root),
        "supplement_summaries": len(summaries),
        "evaluation_summaries": len(eval_summaries),
        "evaluation_manifests": len(manifests),
        "unique_runs": len(run_ids - {""}),
        "spatial_snapshots": len(grids),
        "metric_rows": len(rows),
        "model_roles": sorted(map(str, roles)),
        "seeds": sorted(map(str, seeds)),
        "rar_global_select_count": sorted(map(str, rar_global)),
        "verified_manifested_outputs": output_files,
        "verified_output_bytes": output_bytes,
        "dataset_sha256": sorted(map(str, data_shas)),
        "permeability_field_id": sorted(map(str, field_ids)),
        "permeability_field_sha256": sorted(map(str, field_shas)),
        "capillary_pressure_enabled": sorted(pc_values),
        "failures": failures,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
