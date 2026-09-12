"""Fail-closed evaluation helpers for the frozen Configuration K matrix."""

from __future__ import annotations

import csv
import math
from pathlib import Path
import re
from typing import Any, Mapping

from configuration_k import (
    ANALYSIS_T_MAX_S,
    CANONICAL_METRICS,
    DATASET_SHA256,
    FLOAT_DTYPE,
    FORMAL_RESULT_CONTRACT,
    FORMAL_MAX_ADAM_ITERS,
    FROZEN_DATA_TIME_CROP,
    LEARNING_RATE,
    NETWORK_WIDTH,
    OPTIMIZER,
    RAR_CANDIDATE_COUNT,
    RAR_EVERY,
    RAR_GLOBAL_SELECT_COUNT,
    RAR_RESIDUAL_SELECT_COUNT,
    RAR_START,
    RETAINED_UNIQUE_TIME_COUNT,
    T_REF_S,
    TaskSpec,
    validate_task,
)
from eos_k import validate_eos_record


EVALUATION_CODE_PATHS = (
    "configuration_k.py",
    "eos_k.py",
    "k_evaluation_runtime.py",
    "k_physics.py",
    "permeability_field_k.py",
    "Results/evaluate_global_accuracy_M0_M7_L6_dS_shortlabels.py",
    "Results/evaluate_front_plume_M0_M7_L6_dS.py",
    "Results/evaluate_physics_M0_M7_L6_dS.py",
    "evaluation/run_one_k_evaluation.py",
    "Results/evaluate_snapshot_k.py",
    "Results/evaluate_training_stability_k.py",
    "Results/k_spatial_diagnostics.py",
    "evaluation/run_k_supplement.py",
    "evaluation/assemble_k_supplement.py",
)


def _require_exact(config: Mapping[str, Any], key: str, expected: Any) -> None:
    if key not in config:
        raise ValueError(f"Checkpoint config is missing required key {key!r}.")
    actual = config[key]
    if isinstance(expected, float):
        try:
            matches = float(actual) == expected
        except (TypeError, ValueError):
            matches = False
    else:
        matches = actual == expected
    if not matches:
        raise ValueError(f"Checkpoint config {key} must be {expected!r}; got {actual!r}.")


def _require_block(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"Checkpoint config {key} must be an object.")
    return value


def _nested_matches(actual: Any, expected: Any) -> bool:
    if isinstance(expected, bool):
        return isinstance(actual, bool) and actual is expected
    if isinstance(expected, float):
        try:
            return float(actual) == expected
        except (TypeError, ValueError):
            return False
    if isinstance(expected, (list, tuple)):
        if not isinstance(actual, (list, tuple)) or len(actual) != len(expected):
            return False
        return all(_nested_matches(a, e) for a, e in zip(actual, expected))
    return actual == expected


def _require_nested_values(
    block: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    prefix: str,
) -> None:
    for key, expected_value in expected.items():
        if key not in block:
            raise ValueError(f"Checkpoint config {prefix}.{key} is required.")
        actual = block[key]
        if not _nested_matches(actual, expected_value):
            raise ValueError(
                f"Checkpoint config {prefix}.{key} must be {expected_value!r}; got {actual!r}."
            )


def _require_sha256_value(config: Mapping[str, Any], key: str, expected: str | None) -> str:
    if key not in config:
        raise ValueError(f"Checkpoint config is missing required key {key!r}.")
    digest = str(config[key]).strip().lower()
    if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError(f"Checkpoint config {key} must be a 64-character hexadecimal SHA-256.")
    if expected is not None and digest != str(expected).strip().lower():
        raise ValueError(
            f"Checkpoint config {key} must be {str(expected).strip().lower()!r}; got {digest!r}."
        )
    return digest


def _require_nonempty_text(config: Mapping[str, Any], key: str) -> str:
    if key not in config:
        raise ValueError(f"Checkpoint config is missing required key {key!r}.")
    value = str(config[key]).strip()
    if not value:
        raise ValueError(f"Checkpoint config {key} must be nonempty.")
    return value


_ROLE_EVALUATION_SWITCHES = {
    "baseline": {
        "model_type": "single",
        "architecture": {
            "use_twonet": False,
            "use_msfourier": False,
            "decouple_ps_input": False,
            "use_coarse_detail_s_branch": False,
            "use_plain_twonet": False,
        },
        "front_loss": False,
        "plume_loss": False,
        "pair_gradient": False,
        "rar": False,
        "fv": False,
    },
    "complete": {
        "model_type": "two_cd",
        "architecture": {
            "use_twonet": True,
            "use_msfourier": True,
            "decouple_ps_input": True,
            "use_coarse_detail_s_branch": True,
            "use_plain_twonet": False,
        },
        "front_loss": True,
        "plume_loss": True,
        "pair_gradient": True,
        "rar": True,
        "fv": True,
    },
    "no_representation": {
        "model_type": "plain_twonet",
        "architecture": {
            "use_twonet": True,
            "use_msfourier": False,
            "decouple_ps_input": False,
            "use_coarse_detail_s_branch": False,
            "use_plain_twonet": True,
        },
        "front_loss": True,
        "plume_loss": True,
        "pair_gradient": True,
        "rar": True,
        "fv": True,
    },
    "no_front_supervision": {
        "model_type": "two_cd",
        "architecture": {
            "use_twonet": True,
            "use_msfourier": True,
            "decouple_ps_input": True,
            "use_coarse_detail_s_branch": True,
            "use_plain_twonet": False,
        },
        "front_loss": False,
        "plume_loss": False,
        "pair_gradient": True,
        "rar": True,
        "fv": True,
    },
    "no_local_fv": {
        "model_type": "two_cd",
        "architecture": {
            "use_twonet": True,
            "use_msfourier": True,
            "decouple_ps_input": True,
            "use_coarse_detail_s_branch": True,
            "use_plain_twonet": False,
        },
        "front_loss": True,
        "plume_loss": True,
        "pair_gradient": True,
        "rar": True,
        "fv": False,
    },
}


def validate_checkpoint_config(
    task: TaskSpec,
    config: Mapping[str, Any],
    *,
    field_id: str | None = None,
    field_sha256: str,
    field_metadata_sha256: str | None = None,
    reference_provenance_sha256: str | None = None,
    dataset_sha256: str = DATASET_SHA256,
) -> Mapping[str, Any]:
    """Validate that a checkpoint is exactly one registered formal K run.

    Evaluation is intentionally stricter than model reconstruction: a legacy,
    smoke, exploratory, Pc-enabled, time-drifted, or asset-mismatched checkpoint
    is rejected before any metric is computed.
    """

    validate_task(task.role, task.exp_name, task.training_strategy, task.leave_out, task.seed)
    expected_field_id = str(field_id).strip() if field_id is not None else str(
        config.get("permeability_field_id", "")
    ).strip()
    if not expected_field_id:
        raise ValueError("Checkpoint config permeability_field_id must be nonempty.")
    frozen = {
        "configuration": "K",
        "run_kind": "formal",
        "model_role": task.role,
        "exp_name": task.exp_name,
        "training_strategy": task.training_strategy,
        "leave_out": task.leave_out,
        "seed": task.seed,
        "task_index": task.array_index,
        "task_key": task.key,
        "run_id": task.run_id,
        "pc_enabled": False,
        "pc_entry_pressure_pa": 0.0,
        "t_ref_s": T_REF_S,
        "analysis_t_max_s": ANALYSIS_T_MAX_S,
        "width": NETWORK_WIDTH,
        "dtype": FLOAT_DTYPE,
        "optimizer": OPTIMIZER,
        "learning_rate": LEARNING_RATE,
        "max_adam_iters": FORMAL_MAX_ADAM_ITERS,
        "rar_start_iteration": RAR_START,
        "rar_update_interval": RAR_EVERY,
        "rar_candidate_count": RAR_CANDIDATE_COUNT,
        "rar_residual_select_count": RAR_RESIDUAL_SELECT_COUNT,
        "rar_global_select_count": RAR_GLOBAL_SELECT_COUNT,
        "permeability_field_id": expected_field_id,
        "permeability_field_sha256": str(field_sha256).strip().lower(),
        "reference_dataset_sha256": str(dataset_sha256),
        "reference_provenance_verified": True,
        "retained_unique_time_count": RETAINED_UNIQUE_TIME_COUNT,
    }
    for key, expected in frozen.items():
        _require_exact(config, key, expected)
    _require_sha256_value(
        config,
        "field_metadata_sha256",
        field_metadata_sha256,
    )
    _require_sha256_value(
        config,
        "reference_provenance_sha256",
        reference_provenance_sha256,
    )
    _require_sha256_value(config, "training_script_sha256", None)
    for version_key in ("python_version", "torch_version", "cuda_version"):
        value = _require_nonempty_text(config, version_key)
        if value.lower() in {"none", "unknown"}:
            raise ValueError(f"Checkpoint config {version_key} must identify the formal runtime.")
    gpu_model = _require_nonempty_text(config, "gpu_model")
    if gpu_model.lower() in {"cpu", "none", "unknown"}:
        raise ValueError("Checkpoint config gpu_model must identify the CUDA GPU used for training.")
    slurm_job_id = _require_nonempty_text(config, "slurm_job_id")
    if re.fullmatch(r"[0-9]+", slurm_job_id) is None:
        raise ValueError(f"Checkpoint config slurm_job_id must be numeric; got {slurm_job_id!r}.")
    crop = _require_block(config, "data_time_crop")
    _require_nested_values(
        crop,
        FROZEN_DATA_TIME_CROP,
        prefix="data_time_crop",
    )

    physical = _require_block(config, "physical_parameters")
    _require_nested_values(
        physical,
        {
            "L_ref": 5.0,
            "T_ref": T_REF_S,
            "K": 1.0e-14,
            "permeability_field_id": expected_field_id,
            "permeability_field_sha256": str(field_sha256).strip().lower(),
            "pc_enable": False,
            "pc_entry_pressure": 0.0,
            "phi": 0.2,
            "rho_w_const": 1027.61,
            "S_ic_co2": 0.0,
            "S_inj_co2": 0.8,
            "mu_w": 2.5e-4,
            "mu_c": 2.25e-5,
            "mu_ref": 2.5e-4,
            "U_in": 5.4e-5,
            "U_ref": 5.4e-5,
            "k_ref": 1.0e-14,
            "p0": 10.0e6,
            "p_out": 10.0e6,
            "P_ref": 6.75e6,
            "A_time": 5.0 / (5.4e-5 * T_REF_S),
            "domain_x_m": [0.0, 5.0],
            "domain_y_m": [0.0, 5.0],
            "r_well": 0.5,
            "inj_center": [0.0, 0.0],
            "out_center": [5.0, 5.0],
            "Sw_irr": 0.2,
            "Snr": 0.0,
            "S_CO2_MAX": 0.8,
            "krw0": 1.0,
            "krc0": 1.0,
            "nw": 2.0,
            "nc": 2.0,
        },
        prefix="physical_parameters",
    )

    switches = _ROLE_EVALUATION_SWITCHES[task.role]
    _require_exact(config, "model_type", switches["model_type"])
    architecture = _require_block(config, "architecture")
    _require_nested_values(
        architecture,
        {
            **switches["architecture"],
            "msff_m_per_scale": 24,
            "msff_scales": [1.0, 2.0, 4.0, 8.0, 16.0, 32.0],
            "msff_base_scale": 3.0,
        },
        prefix="architecture",
    )
    coarse_detail = _require_block(architecture, "m3_coarse_detail")
    _require_nested_values(
        coarse_detail,
        {
            "coarse_m_per_scale": 24,
            "coarse_scales": [1.0, 2.0, 4.0, 8.0],
            "coarse_base_scale": 3.0,
            "detail_m_per_scale": 24,
            "detail_scales": [4.0, 8.0, 16.0, 32.0],
            "detail_base_scale": 3.0,
            "coarse_depth": 6,
            "detail_depth": 6,
            "detail_gain": 1.0,
            "detail_tanh": True,
        },
        prefix="architecture.m3_coarse_detail",
    )

    schedules = _require_block(config, "training_schedules")
    _require_nested_values(
        schedules,
        {
            "use_beta_curriculum": False,
            "beta_max": 20.0,
            "use_diffusion_decay": False,
        },
        prefix="training_schedules",
    )
    saturation = _require_block(config, "saturation_ic")
    _require_nested_values(
        saturation,
        {
            "s_ic_co2": 0.0,
            "s_inj_co2": 0.8,
            "s_co2_output_min": 1.0e-6,
            "s_co2_output_max": 0.8,
        },
        prefix="saturation_ic",
    )
    if "m4_hard_ic" in config:
        raise ValueError("Checkpoint config m4_hard_ic is not permitted in formal Configuration K.")

    front = _require_block(config, "m4_front_plume_loss")
    _require_nested_values(
        front,
        {
            "front_loss": switches["front_loss"],
            "front_s_min": 0.05,
            "front_s_max": 0.30,
            "front_center": 0.175,
            "front_sigma": 0.075,
            "front_weight_rel": 1.0,
            "plume_loss": switches["plume_loss"],
            "plume_threshold": 0.05,
            "plume_weight_rel": 1.0,
        },
        prefix="m4_front_plume_loss",
    )
    pair_gradient = _require_block(config, "m5_pairwise_front_gradient")
    _require_nested_values(
        pair_gradient,
        {
            "enabled": switches["pair_gradient"],
            "front_center": 0.175,
            "front_sigma": 0.075,
            "s_min": 0.05,
            "s_max": 0.30,
        },
        prefix="m5_pairwise_front_gradient",
    )
    rar = _require_block(config, "m6_rar")
    _require_nested_values(
        rar,
        {
            "enabled": switches["rar"],
            "rar_start": RAR_START,
            "rar_every": RAR_EVERY,
            "n_candidate": RAR_CANDIDATE_COUNT,
            "n_select_residual": RAR_RESIDUAL_SELECT_COUNT,
            "n_select_global": RAR_GLOBAL_SELECT_COUNT,
        },
        prefix="m6_rar",
    )
    fv = _require_block(config, "m7_fv_mass")
    _require_nested_values(
        fv,
        {
            "enabled": switches["fv"],
            "w_fv_mass_max": 0.30,
            "fv_start": 8000,
            "fv_ramp_steps": 6000,
            "fv_n_cells": 1024,
            "fv_h_tilde": 1.0 / 64.0,
            "fv_dt_tilde": 0.0015850067138671875,
            "fv_w_water": 0.05,
            "use_front_band_weight": True,
            "front_center": 0.175,
            "front_sigma": 0.075,
            "front_weight_floor": 0.05,
        },
        prefix="m7_fv_mass",
    )
    eos_record = config.get("eos_record")
    if not isinstance(eos_record, Mapping):
        raise ValueError("Checkpoint config eos_record must contain the frozen Configuration K EOS object.")
    validate_eos_record(eos_record, expected_dataset_sha256=str(dataset_sha256))
    return config


def _finite_metric(row: Mapping[str, Any], key: str) -> float:
    if key not in row:
        raise ValueError(f"Evaluation output is missing required metric {key!r}.")
    try:
        value = float(row[key])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Evaluation metric {key!r} must be numeric; got {row[key]!r}.") from exc
    if not math.isfinite(value):
        raise ValueError(f"Evaluation metric {key!r} must be finite; got {value!r}.")
    return value


def build_metric_record(
    task: TaskSpec,
    *,
    global_row: Mapping[str, Any],
    front_row: Mapping[str, Any],
    physics_row: Mapping[str, Any],
) -> dict[str, Any]:
    """Map evaluator-native columns onto the five preregistered K metrics."""

    validate_task(task.role, task.exp_name, task.training_strategy, task.leave_out, task.seed)
    record: dict[str, Any] = {
        "configuration": "K",
        "model_role": task.role,
        "exp_name": task.exp_name,
        "training_strategy": task.training_strategy,
        "leave_out": task.leave_out,
        "seed": task.seed,
        "task_index": task.array_index,
        "run_id": task.run_id,
        **FORMAL_RESULT_CONTRACT,
        "saturation_rel_l2": _finite_metric(global_row, "S_rel_l2"),
        "front_band_rmse": _finite_metric(front_row, "front_band_wide_005_030_rmse_mean"),
        "contour_chamfer_s0175": _finite_metric(front_row, "contour_chamfer_S0175_mean"),
        "local_fv_co2_rmse": _finite_metric(physics_row, "local_FV_CO2_RMSE"),
        "co2_mass_rel_error": _finite_metric(physics_row, "CO2_mass_rel_error_mean"),
    }
    if tuple(key for key in record if key in CANONICAL_METRICS) != CANONICAL_METRICS:
        raise RuntimeError("Internal canonical metric ordering drifted from configuration_k.CANONICAL_METRICS.")
    return record


def _sha256(value: Any, key: str) -> str:
    digest = str(value).strip().lower()
    if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError(f"Metric evidence {key} must be a 64-character hexadecimal SHA-256.")
    return digest


def attach_metric_evidence(
    metric_record: Mapping[str, Any],
    *,
    checkpoint_sha256: str,
    resolved_config_sha256: str,
    permeability_field_sha256: str,
    field_metadata_sha256: str,
    reference_dataset_sha256: str,
    reference_provenance_sha256: str,
    evaluation_manifest_sha256: str,
    eos_record_sha256: str,
    code_manifest_sha256: str,
) -> dict[str, Any]:
    if "evidence" in metric_record:
        raise ValueError("Canonical metric record must not already contain an evidence object.")
    payload = dict(metric_record)
    payload["evidence"] = {
        "checkpoint_sha256": _sha256(checkpoint_sha256, "checkpoint_sha256"),
        "resolved_config_sha256": _sha256(resolved_config_sha256, "resolved_config_sha256"),
        "permeability_field_sha256": _sha256(permeability_field_sha256, "permeability_field_sha256"),
        "field_metadata_sha256": _sha256(field_metadata_sha256, "field_metadata_sha256"),
        "reference_dataset_sha256": _sha256(reference_dataset_sha256, "reference_dataset_sha256"),
        "reference_provenance_sha256": _sha256(
            reference_provenance_sha256, "reference_provenance_sha256"
        ),
        "evaluation_manifest_sha256": _sha256(
            evaluation_manifest_sha256, "evaluation_manifest_sha256"
        ),
        "eos_record_sha256": _sha256(eos_record_sha256, "eos_record_sha256"),
        "code_manifest_sha256": _sha256(code_manifest_sha256, "code_manifest_sha256"),
    }
    return payload


def extract_metric_record(
    payload: Mapping[str, Any],
    *,
    checkpoint_sha256: str,
    resolved_config_sha256: str,
    permeability_field_sha256: str,
    field_metadata_sha256: str,
    reference_dataset_sha256: str,
    reference_provenance_sha256: str,
    evaluation_manifest_sha256: str,
    eos_record_sha256: str,
    code_manifest_sha256: str,
) -> dict[str, Any]:
    evidence = payload.get("evidence")
    if not isinstance(evidence, Mapping):
        raise ValueError("Metric payload is missing its evidence object.")
    expected = {
        "checkpoint_sha256": _sha256(checkpoint_sha256, "checkpoint_sha256"),
        "resolved_config_sha256": _sha256(resolved_config_sha256, "resolved_config_sha256"),
        "permeability_field_sha256": _sha256(permeability_field_sha256, "permeability_field_sha256"),
        "field_metadata_sha256": _sha256(field_metadata_sha256, "field_metadata_sha256"),
        "reference_dataset_sha256": _sha256(reference_dataset_sha256, "reference_dataset_sha256"),
        "reference_provenance_sha256": _sha256(
            reference_provenance_sha256, "reference_provenance_sha256"
        ),
        "evaluation_manifest_sha256": _sha256(
            evaluation_manifest_sha256, "evaluation_manifest_sha256"
        ),
        "eos_record_sha256": _sha256(eos_record_sha256, "eos_record_sha256"),
        "code_manifest_sha256": _sha256(code_manifest_sha256, "code_manifest_sha256"),
    }
    if set(evidence) != set(expected):
        raise ValueError(
            "Metric evidence must contain exactly the frozen artifact and evaluator SHA-256 values."
        )
    for key, expected_value in expected.items():
        actual = _sha256(evidence[key], key)
        if actual != expected_value:
            raise ValueError(f"Metric evidence {key} mismatch: expected {expected_value}, got {actual}.")
    return {key: value for key, value in payload.items() if key != "evidence"}


def read_single_csv_row(path: str | Path) -> dict[str, str]:
    source = Path(path)
    if not source.is_file():
        raise ValueError(f"Required evaluation CSV not found: {source}")
    with source.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one evaluated run in {source}; found {len(rows)} rows.")
    return rows[0]


def validate_evaluator_row_identity(task: TaskSpec, row: Mapping[str, Any]) -> None:
    expected = {
        "exp_name": task.exp_name,
        "training_strategy": task.training_strategy,
        "leave_out": task.leave_out,
        "seed": str(task.seed),
    }
    for key, value in expected.items():
        if key not in row:
            raise ValueError(f"Evaluator row is missing identity column {key!r}.")
        actual = str(row[key]).strip()
        if actual != value:
            raise ValueError(f"Evaluator row {key} must be {value!r}; got {actual!r}.")


def build_evaluator_commands(
    task: TaskSpec,
    *,
    python_executable: str | Path,
    project_root: str | Path,
    checkpoint_dir: str | Path,
    data_path: str | Path,
    field_path: str | Path,
    field_metadata_path: str | Path,
    out_dir: str | Path,
    device: str,
    gpu_id: int,
) -> dict[str, list[str]]:
    """Build the three deterministic one-checkpoint evaluator invocations."""

    validate_task(task.role, task.exp_name, task.training_strategy, task.leave_out, task.seed)
    root = Path(project_root)
    output = Path(out_dir)
    include_loo = "0" if task.leave_out == "NONE" else "1"
    common = [
        "--ckpt-dir", str(Path(checkpoint_dir)),
        "--data", str(Path(data_path)),
        "--analysis-t-max-s", str(ANALYSIS_T_MAX_S),
        "--experiments", task.exp_name,
        "--training-strategy", task.training_strategy,
        "--include-loo", include_loo,
        "--loo-exp", "M7",
        "--leave-outs", task.leave_out,
        "--seed", str(task.seed),
        "--device", str(device),
        "--gpu-id", str(int(gpu_id)),
        "--dtype", FLOAT_DTYPE,
        "--infer-batch-size", "65536",
        "--max-time-steps", str(RETAINED_UNIQUE_TIME_COUNT),
        "--max-sort-n", "500000000",
        "--random-seed", "0",
        "--time-indices", "t0,early,middle,late,final",
        "--time-unit", "days",
        "--figure-dpi", "600",
        "--export-pdf", "0",
        "--export-tiff", "0",
        "--export-svg", "0",
    ]
    python = str(python_executable)
    global_command = [
        python,
        str(root / "Results/evaluate_global_accuracy_M0_M7_L6_dS_shortlabels.py"),
        *common,
        "--out-dir", str(output / "global"),
        "--sample-per-time", "20000",
        "--pressure-display-unit", "MPa",
        "--s-ic-co2", "0.0",
        "--Snr", "0.0",
        "--Sw-irr", "0.2",
        "--plot-label-mode", "auto",
    ]
    front_command = [
        python,
        str(root / "Results/evaluate_front_plume_M0_M7_L6_dS.py"),
        *common,
        "--out-root", str(output),
        "--loo-labels", "0",
        "--sample-per-time", "20000",
        "--grid-nx", "420",
        "--grid-ny", "420",
        "--max-interp-input-points", "120000",
        "--front-low-wide", "0.05",
        "--front-high-wide", "0.30",
        "--front-low-narrow", "0.10",
        "--front-high-narrow", "0.25",
        "--contour-levels", "0.175",
        "--plume-thresholds", "0.05,0.10,0.20,0.40",
        "--excess-plume-thresholds", "0.02,0.05,0.10,0.20",
        "--s-ic-co2", "0.0",
        "--pairgrad-max-pairs-per-time", "25000",
        "--pairgrad-s-min", "0.05",
        "--pairgrad-s-max", "0.30",
        "--pairgrad-k-neighbors", "4",
        "--pairgrad-min-dist", "0.0002",
        "--pairgrad-max-dist", "0.0625",
        "--pairgrad-clip", "20.0",
        "--pairgrad-front-center", "0.175",
        "--pairgrad-front-sigma", "0.075",
        "--pairgrad-front-weight-floor", "0.01",
        "--speckle-pred-threshold-002", "0.02",
        "--speckle-true-background-002", "0.01",
        "--speckle-pred-threshold-005", "0.05",
        "--speckle-true-background-005", "0.02",
        "--speckle-min-component-pixels", "3",
    ]
    physics_command = [
        python,
        str(root / "Results/evaluate_physics_M0_M7_L6_dS.py"),
        *common,
        "--out-root", str(output),
        "--loo-labels", "0",
        "--k-field", str(Path(field_path)),
        "--k-field-metadata", str(Path(field_metadata_path)),
        "--pde-points", "4096",
        "--pde-batch-size", "512",
        "--bc-points", "2048",
        "--outer-points-each", "512",
        "--ic-points", "4096",
        "--mass-sample-per-time", "30000",
        "--mass-grid-nx", "256",
        "--mass-grid-ny", "256",
        "--fv-n-cells-eval", "2048",
        "--fv-batch-size", "256",
        "--fv-h-tilde", "0.015625",
        "--fv-dt-tilde", "0.0015850067138671875",
        "--s-ic-co2", "0.0",
        "--Snr", "0.0",
        "--Sw-irr", "0.2",
    ]
    return {"global": global_command, "front": front_command, "physics": physics_command}
