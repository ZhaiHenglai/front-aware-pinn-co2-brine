import json
from pathlib import Path
import subprocess
import sys

from evaluation.run_one_k_evaluation import _json_equivalent


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "evaluation/run_one_k_evaluation.py"


def _dry_run(tmp_path, task_index):
    return subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--task-index", str(task_index),
            "--checkpoint", str(tmp_path / "final_checkpoint.pt"),
            "--data", str(tmp_path / "reference.pt"),
            "--field", str(tmp_path / "K_field_unique_grid.npz"),
            "--field-metadata", str(tmp_path / "K_field_unique_grid.json"),
            "--reference-provenance", str(tmp_path / "k_reference_provenance.json"),
            "--out-dir", str(tmp_path / "evaluation"),
            "--python", sys.executable,
            "--device", "cuda",
            "--gpu-id", "0",
            "--dry-run",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_runner_dry_run_resolves_task_and_all_three_evaluators(tmp_path):
    result = _dry_run(tmp_path, 19)

    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    assert payload["task"]["role"] == "no_local_fv"
    assert payload["task"]["seed"] == 42
    assert tuple(payload["commands"]) == ("global", "front", "physics")
    assert "--k-field" in payload["commands"]["physics"]
    assert payload["content_validation"] == "deferred_until_execution"


def test_runner_dry_run_rejects_unregistered_task_index(tmp_path):
    result = _dry_run(tmp_path, 20)

    assert result.returncode != 0
    assert "0..19" in result.stderr


def test_runner_help_documents_all_immutable_assets():
    result = subprocess.run(
        [sys.executable, str(RUNNER), "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    for option in (
        "--checkpoint",
        "--data",
        "--field",
        "--field-metadata",
        "--reference-provenance",
    ):
        assert option in result.stdout


def test_checkpoint_and_resolved_config_comparison_normalizes_tuple_to_json_list():
    assert _json_equivalent(
        {"architecture": {"scales": (1.0, 2.0, 4.0)}},
        {"architecture": {"scales": [1.0, 2.0, 4.0]}},
    )
    assert not _json_equivalent(
        {"architecture": {"scales": (1.0, 2.0, 4.0)}},
        {"architecture": {"scales": [1.0, 2.0, 8.0]}},
    )
