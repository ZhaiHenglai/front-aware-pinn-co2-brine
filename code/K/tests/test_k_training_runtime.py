import json

import pytest
import torch

from configuration_k import DATASET_SHA256, TaskSpec
from k_training_runtime import (
    build_resolved_config,
    checkpoint_paths,
    ensure_finite_gradients,
    ensure_finite_parameters,
    ensure_finite_training_loss,
    finalize_training_summary,
    write_json_atomic,
)


@pytest.fixture
def task():
    return TaskSpec("complete", "M7", "BASE", "NONE", 0)


@pytest.fixture
def assets():
    return {
        "status": "pass",
        "run_kind": "formal",
        "dataset": {"path": "/data/reference.pt", "sha256": DATASET_SHA256},
        "field": {
            "path": "/data/K_field_unique_grid.npz",
            "field_id": "K-fixed-001",
            "sha256": "a" * 64,
            "units": "m^2",
        },
        "field_metadata_path": "/data/K_field_unique_grid.json",
        "field_metadata_sha256": "c" * 64,
        "reference_provenance_path": "/data/k_reference_provenance.json",
        "reference_provenance_sha256": "d" * 64,
        "reference_provenance": {
            "verified": True,
            "capillary_pressure_enabled": False,
        },
    }


@pytest.fixture
def runtime():
    return {
        "width": 160,
        "dtype": "float64",
        "optimizer": "Adam",
        "learning_rate": 2e-4,
        "max_adam_iters": 20000,
        "rar_start_iteration": 3500,
        "rar_update_interval": 100,
        "rar_candidate_count": 30000,
        "rar_residual_select_count": 22500,
        "rar_global_select_count": 0,
        "python_version": "3.11.test",
        "torch_version": "test",
        "cuda_version": "test",
        "gpu_model": "Tesla V100",
        "slurm_job_id": "123",
        "training_script_sha256": "b" * 64,
    }


def test_resolved_config_records_no_pc_time_rar_and_hashes(task, assets, runtime):
    config = build_resolved_config(task, assets, runtime, run_kind="formal")

    assert config["configuration"] == "K"
    assert config["model_role"] == "complete"
    assert config["pc_enabled"] is False
    assert config["pc_entry_pressure_pa"] == 0.0
    assert config["t_ref_s"] == 31544.99609375
    assert config["analysis_t_max_s"] == 31544.99609375
    assert config["rar_global_select_count"] == 0
    assert config["permeability_field_id"] == "K-fixed-001"
    assert config["permeability_field_sha256"] == "a" * 64
    assert config["field_metadata_sha256"] == "c" * 64
    assert config["reference_dataset_sha256"] == DATASET_SHA256
    assert config["reference_provenance_sha256"] == "d" * 64


@pytest.mark.parametrize("missing", ["field_metadata_sha256", "reference_provenance_sha256"])
def test_resolved_config_requires_sidecar_and_provenance_hashes(
    task, assets, runtime, missing
):
    assets.pop(missing)

    with pytest.raises(ValueError, match=missing):
        build_resolved_config(task, assets, runtime, run_kind="formal")


@pytest.mark.parametrize(
    ("key", "bad_value", "message"),
    [
        ("width", 128, "width"),
        ("dtype", "float32", "dtype"),
        ("learning_rate", 1e-3, "learning_rate"),
        ("rar_update_interval", 10, "rar_update_interval"),
        ("rar_global_select_count", 7500, "rar_global_select_count"),
    ],
)
def test_formal_runtime_rejects_frozen_setting_drift(task, assets, runtime, key, bad_value, message):
    runtime[key] = bad_value

    with pytest.raises(ValueError, match=message):
        build_resolved_config(task, assets, runtime, run_kind="formal")


def test_smoke_allows_short_iteration_budget_but_not_scientific_drift(task, assets, runtime):
    assets["run_kind"] = "smoke"
    assets["reference_provenance"]["verified"] = False
    runtime["max_adam_iters"] = 100

    config = build_resolved_config(task, assets, runtime, run_kind="smoke")
    assert config["run_kind"] == "smoke"
    assert config["max_adam_iters"] == 100

    runtime["rar_candidate_count"] = 1000
    with pytest.raises(ValueError, match="rar_candidate_count"):
        build_resolved_config(task, assets, runtime, run_kind="smoke")


def test_atomic_json_writer_replaces_existing_complete_document(tmp_path):
    path = tmp_path / "resolved_config.json"
    path.write_text('{"old": true}', encoding="utf-8")

    write_json_atomic(path, {"configuration": "K", "seed": 0})

    assert json.loads(path.read_text(encoding="utf-8")) == {"configuration": "K", "seed": 0}
    assert not (tmp_path / "resolved_config.json.tmp").exists()


def test_training_summary_records_status_timing_and_peak_memory():
    summary = finalize_training_summary(
        status="completed",
        elapsed_seconds=12.5,
        peak_gpu_memory_allocated_bytes=1024,
        peak_gpu_memory_reserved_bytes=2048,
        checkpoint_path="/runs/final_checkpoint.pt",
        error=None,
    )

    assert summary == {
        "status": "completed",
        "elapsed_seconds": 12.5,
        "peak_gpu_memory_allocated_bytes": 1024,
        "peak_gpu_memory_reserved_bytes": 2048,
        "checkpoint_path": "/runs/final_checkpoint.pt",
        "error": None,
    }


def test_final_checkpoint_has_canonical_name_and_evaluator_alias(tmp_path, task):
    canonical, alias = checkpoint_paths(tmp_path, task, profile="v100", suffix="final")

    assert canonical == tmp_path / "final_checkpoint.pt"
    assert alias == tmp_path / "model_M7_BASE_LOO-NONE_seed0_v100_final.pt"


def test_nonfinite_training_loss_fails_with_iteration_context():
    with pytest.raises(FloatingPointError, match="loss.*iteration 17"):
        ensure_finite_training_loss(torch.tensor(float("nan")), iteration=17)


def test_nonfinite_gradient_names_the_bad_parameter():
    model = torch.nn.Linear(2, 1)
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    model.weight.grad[0, 0] = float("inf")

    with pytest.raises(FloatingPointError, match="gradient.*weight.*iteration 9"):
        ensure_finite_gradients(model.named_parameters(), iteration=9)


def test_nonfinite_parameter_after_optimizer_step_is_rejected():
    model = torch.nn.Linear(2, 1)
    model.bias.data[0] = float("-inf")

    with pytest.raises(FloatingPointError, match="parameter.*bias.*iteration 3"):
        ensure_finite_parameters(model.named_parameters(), iteration=3)
