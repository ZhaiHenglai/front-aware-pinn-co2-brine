#!/usr/bin/env python3
"""Aggregate and validate the exact 20 formal Configuration K metric records."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import statistics
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configuration_k import CANONICAL_METRICS, all_formal_tasks
from k_evaluation_runtime import (
    EVALUATION_CODE_PATHS,
    extract_metric_record,
    validate_checkpoint_config,
)
from k_results_validation import (
    preregistered_direction_counts,
    validate_metric_records,
    write_metrics_csv,
)
from k_training_runtime import write_json_atomic


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True, help="Root containing formal runs/.")
    parser.add_argument("--evaluation-root", type=Path, default=None, help="Root containing by_run/; defaults to OUTPUT_ROOT/evaluation.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--field", type=Path, required=True)
    parser.add_argument("--field-metadata", type=Path, required=True)
    parser.add_argument("--reference-provenance", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true", help="List the frozen inputs without reading or writing them.")
    return parser


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"Required formal artifact not found: {path}")
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid JSON artifact: {path}") from exc
    if not isinstance(record, dict):
        raise ValueError(f"JSON artifact must contain an object: {path}")
    return record


def _require_status(path: Path, expected: str = "completed") -> dict[str, Any]:
    record = _read_json(path)
    if record.get("status") != expected:
        raise ValueError(f"Artifact status in {path} must be {expected!r}; got {record.get('status')!r}.")
    return record


def _require_positive_finite(record: dict[str, Any], key: str, *, label: str) -> float:
    try:
        value = float(record[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{label} requires numeric {key}.") from exc
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{label} {key} must be positive and finite; got {value!r}.")
    return value


def _validate_training_summary(record: dict[str, Any], checkpoint: Path) -> None:
    _require_positive_finite(record, "elapsed_seconds", label="Training summary")
    try:
        allocated = int(record["peak_gpu_memory_allocated_bytes"])
        reserved = int(record["peak_gpu_memory_reserved_bytes"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Training summary requires integer peak GPU-memory telemetry.") from exc
    if allocated <= 0 or reserved < allocated:
        raise ValueError(
            "Training summary peak GPU-memory telemetry must satisfy "
            f"0 < allocated <= reserved; got {allocated}, {reserved}."
        )
    if Path(str(record.get("checkpoint_path", ""))).resolve() != checkpoint.resolve():
        raise ValueError("Training summary checkpoint_path does not identify the formal checkpoint.")
    if record.get("error") is not None:
        raise ValueError("Completed training summary must record error=null.")


def _validate_evaluation_summary(record: dict[str, Any], metric_path: Path) -> None:
    _require_positive_finite(record, "elapsed_seconds", label="Evaluation summary")
    if Path(str(record.get("metrics_path", ""))).resolve() != metric_path.resolve():
        raise ValueError("Evaluation summary metrics_path does not identify the formal metric record.")
    if record.get("error") is not None:
        raise ValueError("Completed evaluation summary must record error=null.")


def _write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ("model_role", "metric", "n", "mean", "sample_std", "median", "minimum", "maximum")
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _summary_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for role, *_ in (
        ("baseline",),
        ("complete",),
        ("no_representation",),
        ("no_front_supervision",),
        ("no_local_fv",),
    ):
        role_records = [row for row in records if row["model_role"] == role]
        for metric in CANONICAL_METRICS:
            values = [float(row[metric]) for row in role_records]
            rows.append({
                "model_role": role,
                "metric": metric,
                "n": len(values),
                "mean": statistics.fmean(values),
                "sample_std": statistics.stdev(values),
                "median": statistics.median(values),
                "minimum": min(values),
                "maximum": max(values),
            })
    return rows


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    evaluation_root = args.evaluation_root or (args.output_root / "evaluation")
    tasks = all_formal_tasks()
    metric_paths = [
        evaluation_root / "by_run" / task.role / task.run_id / "metrics.json"
        for task in tasks
    ]
    raw_csv = evaluation_root / "heterogeneous_metrics_by_seed.csv"
    summary_csv = evaluation_root / "heterogeneous_metrics_summary.csv"
    summary_json = evaluation_root / "aggregation_summary.json"
    manifest_json = evaluation_root / "formal_artifact_manifest.json"
    if args.dry_run:
        print(json.dumps({
            "expected_metric_files": [str(path) for path in metric_paths],
            "raw_csv": str(raw_csv),
            "summary_csv": str(summary_csv),
            "manifest": str(manifest_json),
        }))
        return 0

    from permeability_field_k import file_sha256, validate_asset_bundle

    for target in (raw_csv, summary_csv, summary_json, manifest_json):
        if target.exists():
            raise ValueError(f"Refusing to overwrite an existing formal aggregate: {target}")
    evaluation_root.mkdir(parents=True, exist_ok=True)
    assets = validate_asset_bundle(
        args.data,
        args.field,
        args.field_metadata,
        args.reference_provenance,
        run_kind="formal",
    )
    code_manifest_path = PROJECT_ROOT / "manifests/CODE_SHA256SUMS"
    if not code_manifest_path.is_file():
        raise ValueError(f"Code checksum manifest not found: {code_manifest_path}")
    current_code_manifest_sha256 = file_sha256(code_manifest_path)

    metric_records = []
    artifact_records = []
    eos_record_hashes: set[str] = set()
    code_manifest_hashes: set[str] = set()
    for task, metric_path in zip(tasks, metric_paths):
        run_dir = args.output_root / "runs" / task.role / task.run_id
        eval_dir = metric_path.parent
        checkpoint = run_dir / "final_checkpoint.pt"
        resolved_path = run_dir / "resolved_config.json"
        training_summary_path = run_dir / "training_summary.json"
        evaluation_summary_path = eval_dir / "evaluation_summary.json"
        evaluation_manifest_path = eval_dir / "evaluation_manifest.json"
        if not checkpoint.is_file():
            raise ValueError(f"Formal checkpoint not found: {checkpoint}")
        training_summary = _require_status(training_summary_path)
        evaluation_summary = _require_status(evaluation_summary_path)
        _validate_training_summary(training_summary, checkpoint)
        _validate_evaluation_summary(evaluation_summary, metric_path)
        resolved = _read_json(resolved_path)
        validate_checkpoint_config(
            task,
            resolved,
            field_id=assets["field"]["field_id"],
            field_sha256=assets["field"]["sha256"],
            field_metadata_sha256=assets["field_metadata_sha256"],
            reference_provenance_sha256=assets["reference_provenance_sha256"],
            dataset_sha256=assets["dataset"]["sha256"],
        )
        checkpoint_sha256 = file_sha256(checkpoint)
        resolved_config_sha256 = file_sha256(resolved_path)
        evaluation_manifest_sha256 = file_sha256(evaluation_manifest_path)
        evaluation_manifest = _read_json(evaluation_manifest_path)
        expected_manifest_values = {
            "configuration": "K",
            "task": task.as_dict(),
            "checkpoint_sha256": checkpoint_sha256,
            "resolved_config_sha256": resolved_config_sha256,
            "dataset_sha256": assets["dataset"]["sha256"],
            "permeability_field_id": assets["field"]["field_id"],
            "permeability_field_sha256": assets["field"]["sha256"],
            "field_metadata_sha256": assets["field_metadata_sha256"],
            "reference_provenance_sha256": assets["reference_provenance_sha256"],
            "eos_record_sha256": resolved["eos_record"]["record_sha256"],
        }
        for key, expected in expected_manifest_values.items():
            if evaluation_manifest.get(key) != expected:
                raise ValueError(
                    f"Evaluation manifest {key} mismatch for {task.run_id}: "
                    f"expected {expected!r}, got {evaluation_manifest.get(key)!r}."
                )
        code_evidence = evaluation_manifest.get("code_manifest")
        if not isinstance(code_evidence, dict) or not isinstance(code_evidence.get("entries"), dict):
            raise ValueError(f"Evaluation manifest lacks code checksum evidence for {task.run_id}.")
        code_manifest_sha256 = str(code_evidence.get("sha256", "")).strip().lower()
        if code_manifest_sha256 != current_code_manifest_sha256:
            raise ValueError(
                f"Evaluation code manifest mismatch for {task.run_id}: "
                f"expected {current_code_manifest_sha256}, got {code_manifest_sha256}."
            )
        missing_code = sorted(set(EVALUATION_CODE_PATHS) - set(code_evidence["entries"]))
        if missing_code:
            raise ValueError(
                f"Evaluation code evidence for {task.run_id} is missing dependencies: {missing_code!r}."
            )
        if evaluation_summary.get("evaluation_manifest_sha256") != evaluation_manifest_sha256:
            raise ValueError(f"Evaluation summary manifest SHA-256 mismatch for {task.run_id}.")
        eos_record_sha256 = str(resolved["eos_record"]["record_sha256"]).strip().lower()
        eos_record_hashes.add(eos_record_sha256)
        code_manifest_hashes.add(code_manifest_sha256)
        metric_payload = _read_json(metric_path)
        metrics = extract_metric_record(
            metric_payload,
            checkpoint_sha256=checkpoint_sha256,
            resolved_config_sha256=resolved_config_sha256,
            permeability_field_sha256=assets["field"]["sha256"],
            field_metadata_sha256=assets["field_metadata_sha256"],
            reference_dataset_sha256=assets["dataset"]["sha256"],
            reference_provenance_sha256=assets["reference_provenance_sha256"],
            evaluation_manifest_sha256=evaluation_manifest_sha256,
            eos_record_sha256=eos_record_sha256,
            code_manifest_sha256=code_manifest_sha256,
        )
        metric_records.append(metrics)
        artifact_records.append({
            "task_index": task.array_index,
            "run_id": task.run_id,
            "checkpoint_path": str(checkpoint.resolve()),
            "checkpoint_sha256": checkpoint_sha256,
            "resolved_config_path": str(resolved_path.resolve()),
            "resolved_config_sha256": resolved_config_sha256,
            "metrics_path": str(metric_path.resolve()),
            "metrics_sha256": file_sha256(metric_path),
            "evaluation_manifest_path": str(evaluation_manifest_path.resolve()),
            "evaluation_manifest_sha256": evaluation_manifest_sha256,
            "eos_record_sha256": eos_record_sha256,
            "code_manifest_sha256": code_manifest_sha256,
            "training_summary_path": str(training_summary_path.resolve()),
            "evaluation_summary_path": str(evaluation_summary_path.resolve()),
        })

    if len(eos_record_hashes) != 1:
        raise ValueError(
            f"All 20 formal runs must use one identical EOS record; got {sorted(eos_record_hashes)!r}."
        )
    if code_manifest_hashes != {current_code_manifest_sha256}:
        raise ValueError("All 20 formal evaluations must use the current frozen code manifest.")

    validated = validate_metric_records(metric_records)
    directions = preregistered_direction_counts(validated)
    write_metrics_csv(raw_csv, validated)
    _write_summary_csv(summary_csv, _summary_rows(validated))
    write_json_atomic(manifest_json, {
        "configuration": "K",
        "formal_task_count": len(validated),
        "reference_dataset": assets["dataset"],
        "permeability_field": assets["field"],
        "field_metadata_sha256": assets["field_metadata_sha256"],
        "reference_provenance_path": assets["reference_provenance_path"],
        "reference_provenance_sha256": assets["reference_provenance_sha256"],
        "eos_record_sha256": next(iter(eos_record_hashes)),
        "code_manifest_path": str(code_manifest_path.resolve()),
        "code_manifest_sha256": current_code_manifest_sha256,
        "artifacts": artifact_records,
    })
    write_json_atomic(summary_json, {
        "status": "completed",
        "configuration": "K",
        "formal_row_count": len(validated),
        "raw_metrics_csv": str(raw_csv.resolve()),
        "role_summary_csv": str(summary_csv.resolve()),
        "formal_artifact_manifest": str(manifest_json.resolve()),
        "direction_diagnostics": directions,
    })
    print(json.dumps({"status": "pass", "row_count": len(validated), "raw_csv": str(raw_csv)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
