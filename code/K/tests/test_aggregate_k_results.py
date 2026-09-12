import json
from pathlib import Path
import subprocess
import sys

import pytest

from evaluation.aggregate_k_results import (
    _validate_evaluation_summary,
    _validate_training_summary,
)


ROOT = Path(__file__).resolve().parents[1]
AGGREGATOR = ROOT / "evaluation/aggregate_k_results.py"


def test_aggregator_dry_run_resolves_only_the_registered_twenty_metrics(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(AGGREGATOR),
            "--output-root", str(tmp_path),
            "--data", str(tmp_path / "reference.pt"),
            "--field", str(tmp_path / "field.npz"),
            "--field-metadata", str(tmp_path / "field.json"),
            "--reference-provenance", str(tmp_path / "provenance.json"),
            "--dry-run",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    assert len(payload["expected_metric_files"]) == 20
    assert payload["expected_metric_files"][0].endswith("by_run/baseline/baseline_seed0/metrics.json")
    assert payload["expected_metric_files"][-1].endswith("by_run/no_local_fv/no_local_fv_seed42/metrics.json")
    assert payload["raw_csv"].endswith("evaluation/heterogeneous_metrics_by_seed.csv")


def test_aggregator_help_documents_evidence_and_asset_roots():
    result = subprocess.run(
        [sys.executable, str(AGGREGATOR), "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    for option in ("--output-root", "--evaluation-root", "--data", "--field", "--reference-provenance"):
        assert option in result.stdout


def test_aggregate_requires_complete_runtime_and_gpu_telemetry(tmp_path):
    checkpoint = tmp_path / "final_checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    training = {
        "elapsed_seconds": 10.0,
        "peak_gpu_memory_allocated_bytes": 100,
        "peak_gpu_memory_reserved_bytes": 120,
        "checkpoint_path": str(checkpoint.resolve()),
        "error": None,
    }
    _validate_training_summary(training, checkpoint)

    with pytest.raises(ValueError, match="GPU-memory"):
        _validate_training_summary(
            {**training, "peak_gpu_memory_allocated_bytes": 0}, checkpoint
        )

    metric_path = tmp_path / "metrics.json"
    metric_path.write_text("{}", encoding="utf-8")
    evaluation = {
        "elapsed_seconds": 1.0,
        "metrics_path": str(metric_path.resolve()),
        "error": None,
    }
    _validate_evaluation_summary(evaluation, metric_path)
    with pytest.raises(ValueError, match="metrics_path"):
        _validate_evaluation_summary(
            {**evaluation, "metrics_path": str(tmp_path / "other.json")}, metric_path
        )
