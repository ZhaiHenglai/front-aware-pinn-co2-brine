#!/usr/bin/env python3
"""Run all preregistered metrics for exactly one formal Configuration K checkpoint."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configuration_k import task_from_index
from k_evaluation_runtime import (
    EVALUATION_CODE_PATHS,
    attach_metric_evidence,
    build_evaluator_commands,
    build_metric_record,
    read_single_csv_row,
    validate_checkpoint_config,
    validate_evaluator_row_identity,
)
from k_training_runtime import write_json_atomic


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate one registered formal Configuration K checkpoint with the frozen five-metric protocol."
    )
    parser.add_argument("--task-index", type=int, required=True, help="Registered Configuration K task index (0..19).")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Formal final_checkpoint.pt for this task.")
    parser.add_argument("--data", type=Path, required=True, help="Frozen heterogeneous reference dataset.")
    parser.add_argument("--field", type=Path, required=True, help="Immutable K_field_unique_grid.npz.")
    parser.add_argument("--field-metadata", type=Path, required=True, help="Metadata JSON matching --field.")
    parser.add_argument("--reference-provenance", type=Path, required=True, help="Verified no-Pc reference provenance JSON.")
    parser.add_argument("--out-dir", type=Path, required=True, help="New output directory for this one checkpoint.")
    parser.add_argument("--python", type=Path, default=Path(sys.executable), help="Python executable used for evaluator subprocesses.")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--publication-figures", action="store_true", help="Also export PDF/SVG; numerical protocol unchanged.")
    parser.add_argument("--dry-run", action="store_true", help="Resolve task and commands without reading assets or writing files.")
    return parser


def _load_checkpoint_config(path: Path) -> dict[str, Any]:
    import torch

    if not path.is_file():
        raise ValueError(f"Formal checkpoint not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("config"), dict):
        raise ValueError(f"Checkpoint must contain a top-level config object: {path}")
    if not isinstance(payload.get("state_dict"), dict):
        raise ValueError(f"Checkpoint must contain a top-level state_dict object: {path}")
    return payload["config"]


def _load_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"Required formal JSON evidence not found: {path}")
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid formal JSON evidence: {path}") from exc
    if not isinstance(record, dict):
        raise ValueError(f"Formal JSON evidence must contain an object: {path}")
    return record


def _json_equivalent(left: Any, right: Any) -> bool:
    """Compare checkpoint and JSON evidence after JSON-type normalization."""

    try:
        left_normalized = json.loads(json.dumps(left, sort_keys=True, allow_nan=False))
        right_normalized = json.loads(json.dumps(right, sort_keys=True, allow_nan=False))
    except (TypeError, ValueError):
        return False
    return left_normalized == right_normalized


def _ensure_empty_output(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise ValueError(f"Refusing to overwrite a nonempty evaluation directory: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _validate_code_manifest(root: Path, file_sha256) -> dict[str, Any]:
    """Validate every execution file and return a sealed manifest snapshot."""

    manifest = root / "manifests/CODE_SHA256SUMS"
    if not manifest.is_file():
        raise ValueError(f"Code checksum manifest not found: {manifest}")
    entries: dict[str, str] = {}
    for line_number, raw_line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        if not raw_line.strip():
            continue
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", raw_line)
        if match is None:
            raise ValueError(f"Invalid code checksum line {line_number}: {raw_line!r}")
        expected, relative_text = match.groups()
        relative = Path(relative_text)
        target = (root / relative).resolve()
        try:
            target.relative_to(root.resolve())
        except ValueError as exc:
            raise ValueError(f"Code checksum path escapes project root: {relative_text}") from exc
        if relative.is_absolute() or not target.is_file():
            raise ValueError(f"Code checksum target not found: {relative_text}")
        actual = file_sha256(target)
        if actual != expected:
            raise ValueError(
                f"Code checksum mismatch for {relative_text}: expected {expected}, got {actual}."
            )
        entries[relative.as_posix()] = actual
    if not entries:
        raise ValueError("Code checksum manifest must contain at least one execution file.")
    missing = sorted(set(EVALUATION_CODE_PATHS) - set(entries))
    if missing:
        raise ValueError(f"Code checksum manifest is missing evaluator dependencies: {missing!r}")
    return {
        "path": str(manifest.resolve()),
        "sha256": file_sha256(manifest),
        "entries": entries,
    }


def _checkpoint_view(checkpoint: Path, output: Path, task) -> Path:
    view = output / "checkpoint_view"
    view.mkdir(parents=True, exist_ok=False)
    alias = view / (
        f"model_{task.exp_name}_{task.training_strategy}_LOO-{task.leave_out}_"
        f"seed{task.seed}_v100_final.pt"
    )
    alias.symlink_to(checkpoint.resolve())
    return view


def _run_command(name: str, command: list[str], output: Path) -> None:
    log_path = output / f"{name}_evaluation.log"
    with log_path.open("w", encoding="utf-8") as log:
        log.write("command=" + json.dumps(command) + "\n")
        log.flush()
        subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
            env=os.environ.copy(),
        )


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        task = task_from_index(args.task_index)
    except ValueError as exc:
        parser.error(str(exc))

    preview_commands = build_evaluator_commands(
        task,
        python_executable=args.python,
        project_root=PROJECT_ROOT,
        checkpoint_dir=args.checkpoint.parent,
        data_path=args.data,
        field_path=args.field,
        field_metadata_path=args.field_metadata,
        out_dir=args.out_dir,
        device=args.device,
        gpu_id=args.gpu_id,
    )
    if args.dry_run:
        print(json.dumps({
            "task": task.as_dict(),
            "commands": preview_commands,
            "content_validation": "deferred_until_execution",
        }))
        return 0

    if not args.python.is_file() or not os.access(args.python, os.X_OK):
        raise ValueError(f"Evaluator Python is not executable: {args.python}")
    start = time.monotonic()
    _ensure_empty_output(args.out_dir)

    try:
        from permeability_field_k import file_sha256, validate_asset_bundle

        assets = validate_asset_bundle(
            args.data,
            args.field,
            args.field_metadata,
            args.reference_provenance,
            run_kind="formal",
        )
        checkpoint_config = _load_checkpoint_config(args.checkpoint)
        validate_checkpoint_config(
            task,
            checkpoint_config,
            field_id=assets["field"]["field_id"],
            field_sha256=assets["field"]["sha256"],
            field_metadata_sha256=assets["field_metadata_sha256"],
            reference_provenance_sha256=assets["reference_provenance_sha256"],
            dataset_sha256=assets["dataset"]["sha256"],
        )
        resolved_config_path = args.checkpoint.parent / "resolved_config.json"
        resolved_config = _load_json_object(resolved_config_path)
        validate_checkpoint_config(
            task,
            resolved_config,
            field_id=assets["field"]["field_id"],
            field_sha256=assets["field"]["sha256"],
            field_metadata_sha256=assets["field_metadata_sha256"],
            reference_provenance_sha256=assets["reference_provenance_sha256"],
            dataset_sha256=assets["dataset"]["sha256"],
        )
        if not _json_equivalent(checkpoint_config, resolved_config):
            raise ValueError("Checkpoint config and resolved_config.json must be exactly equivalent.")
        checkpoint_sha256 = file_sha256(args.checkpoint)
        resolved_config_sha256 = file_sha256(resolved_config_path)
        code_evidence = _validate_code_manifest(PROJECT_ROOT, file_sha256)
        expected_trainer_sha256 = code_evidence["entries"][
            "pinn_experiment_M0_M7_BASE_leave_one_out.py"
        ]
        if checkpoint_config["training_script_sha256"] != expected_trainer_sha256:
            raise ValueError(
                "Checkpoint training_script_sha256 does not match the frozen code manifest."
            )
        checkpoint_view = _checkpoint_view(args.checkpoint, args.out_dir, task)
        commands = build_evaluator_commands(
            task,
            python_executable=args.python,
            project_root=PROJECT_ROOT,
            checkpoint_dir=checkpoint_view,
            data_path=args.data,
            field_path=args.field,
            field_metadata_path=args.field_metadata,
            out_dir=args.out_dir,
            device=args.device,
            gpu_id=args.gpu_id,
        )
        if args.publication_figures:
            for command in commands.values():
                for flag in ("--export-pdf", "--export-svg"):
                    command[command.index(flag) + 1] = "1"
        evaluation_manifest_path = args.out_dir / "evaluation_manifest.json"
        write_json_atomic(evaluation_manifest_path, {
            "configuration": "K",
            "task": task.as_dict(),
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_config": checkpoint_config,
            "resolved_config": str(resolved_config_path.resolve()),
            "resolved_config_sha256": resolved_config_sha256,
            "dataset_sha256": assets["dataset"]["sha256"],
            "permeability_field_id": assets["field"]["field_id"],
            "permeability_field_sha256": assets["field"]["sha256"],
            "field_metadata_sha256": assets["field_metadata_sha256"],
            "reference_provenance_sha256": assets["reference_provenance_sha256"],
            "eos_record_sha256": checkpoint_config["eos_record"]["record_sha256"],
            "code_manifest": code_evidence,
            "commands": commands,
        })
        evaluation_manifest_sha256 = file_sha256(evaluation_manifest_path)
        for name, command in commands.items():
            _run_command(name, command, args.out_dir)

        if file_sha256(args.checkpoint) != checkpoint_sha256:
            raise ValueError("Formal checkpoint changed during evaluation.")
        if file_sha256(resolved_config_path) != resolved_config_sha256:
            raise ValueError("resolved_config.json changed during evaluation.")
        final_code_evidence = _validate_code_manifest(PROJECT_ROOT, file_sha256)
        if final_code_evidence != code_evidence:
            raise ValueError("Code checksum evidence changed during evaluation.")
        if file_sha256(evaluation_manifest_path) != evaluation_manifest_sha256:
            raise ValueError("evaluation_manifest.json changed during evaluation.")

        global_row = read_single_csv_row(args.out_dir / "global/summary_global_accuracy.csv")
        front_row = read_single_csv_row(args.out_dir / "front/summary_front_resolution.csv")
        physics_row = read_single_csv_row(args.out_dir / "physics/summary_physics_consistency.csv")
        for row in (global_row, front_row, physics_row):
            validate_evaluator_row_identity(task, row)
        metrics = build_metric_record(
            task,
            global_row=global_row,
            front_row=front_row,
            physics_row=physics_row,
        )
        metric_payload = attach_metric_evidence(
            metrics,
            checkpoint_sha256=checkpoint_sha256,
            resolved_config_sha256=resolved_config_sha256,
            permeability_field_sha256=assets["field"]["sha256"],
            field_metadata_sha256=assets["field_metadata_sha256"],
            reference_dataset_sha256=assets["dataset"]["sha256"],
            reference_provenance_sha256=assets["reference_provenance_sha256"],
            evaluation_manifest_sha256=evaluation_manifest_sha256,
            eos_record_sha256=checkpoint_config["eos_record"]["record_sha256"],
            code_manifest_sha256=code_evidence["sha256"],
        )
        write_json_atomic(args.out_dir / "metrics.json", metric_payload)
        write_json_atomic(args.out_dir / "evaluation_summary.json", {
            "status": "completed",
            "elapsed_seconds": time.monotonic() - start,
            "metrics_path": str((args.out_dir / "metrics.json").resolve()),
            "evaluation_manifest_sha256": evaluation_manifest_sha256,
            "error": None,
        })
        print(json.dumps(metric_payload, sort_keys=True))
        return 0
    except Exception as exc:
        write_json_atomic(args.out_dir / "evaluation_summary.json", {
            "status": "failed",
            "elapsed_seconds": time.monotonic() - start,
            "metrics_path": None,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        })
        raise


if __name__ == "__main__":
    raise SystemExit(main())
