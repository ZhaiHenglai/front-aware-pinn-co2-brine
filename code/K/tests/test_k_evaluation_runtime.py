import copy
import csv
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from configuration_k import DATASET_SHA256, FORMAL_RESULT_CONTRACT, TaskSpec
from eos_k import fit_eos_record
from k_evaluation_runtime import (
    attach_metric_evidence,
    build_evaluator_commands,
    build_metric_record,
    extract_metric_record,
    read_single_csv_row,
    validate_checkpoint_config,
    validate_evaluator_row_identity,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "Results"
if str(RESULTS) not in sys.path:
    sys.path.insert(0, str(RESULTS))

from evaluate_front_plume_M0_M7_L6_dS import (  # noqa: E402
    merge_run_summaries,
    saturation_level_tag,
)
from evaluate_global_accuracy_M0_M7_L6_dS_shortlabels import load_dataset_pack  # noqa: E402


@pytest.fixture
def task():
    return TaskSpec("complete", "M7", "BASE", "NONE", 0)


@pytest.fixture(scope="session")
def eos_record():
    pressure = np.linspace(9.0e6, 12.0e6, 64, dtype=np.float64)
    density = 610.0 + 2.0e-5 * (pressure - 10.5e6)
    return fit_eos_record(
        pressure,
        density,
        reference_dataset_sha256=DATASET_SHA256,
    )


@pytest.fixture
def checkpoint_config(task, eos_record):
    return {
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
        "t_ref_s": 31544.99609375,
        "analysis_t_max_s": 31544.99609375,
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
        "permeability_field_id": "K-fixed-001",
        "permeability_field_sha256": "a" * 64,
        "field_metadata_sha256": "c" * 64,
        "reference_dataset_sha256": DATASET_SHA256,
        "reference_provenance_sha256": "d" * 64,
        "reference_provenance_verified": True,
        "python_version": "3.11.9",
        "torch_version": "2.4.0",
        "cuda_version": "12.1",
        "gpu_model": "NVIDIA V100-SXM2-32GB",
        "slurm_job_id": "12345",
        "training_script_sha256": "9" * 64,
        "data_time_crop": {
            "rows_before": 173688979,
            "rows_after": 173688979,
            "rows_removed": 0,
            "cutoff_s": 31544.99609375,
            "first_retained_time_s": 0.0,
            "last_retained_time_s": 30986.943359375,
        },
        "retained_unique_time_count": 776,
        "eos_record": eos_record,
        "model_type": "two_cd",
        "architecture": {
            "use_twonet": True,
            "use_msfourier": True,
            "msff_m_per_scale": 24,
            "msff_scales": [1.0, 2.0, 4.0, 8.0, 16.0, 32.0],
            "msff_base_scale": 3.0,
            "decouple_ps_input": True,
            "use_coarse_detail_s_branch": True,
            "use_plain_twonet": False,
            "m3_coarse_detail": {
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
        },
        "training_schedules": {
            "use_beta_curriculum": False,
            "beta_max": 20.0,
            "use_diffusion_decay": False,
        },
        "saturation_ic": {
            "s_ic_co2": 0.0,
            "s_inj_co2": 0.8,
            "s_co2_output_min": 1.0e-6,
            "s_co2_output_max": 0.8,
        },
        "m4_front_plume_loss": {
            "front_loss": True,
            "front_s_min": 0.05,
            "front_s_max": 0.30,
            "front_center": 0.175,
            "front_sigma": 0.075,
            "front_weight_rel": 1.0,
            "plume_loss": True,
            "plume_threshold": 0.05,
            "plume_weight_rel": 1.0,
        },
        "m5_pairwise_front_gradient": {
            "enabled": True,
            "front_center": 0.175,
            "front_sigma": 0.075,
            "s_min": 0.05,
            "s_max": 0.30,
        },
        "m6_rar": {
            "enabled": True,
            "rar_start": 3500,
            "rar_every": 100,
            "n_candidate": 30000,
            "n_select_residual": 22500,
            "n_select_global": 0,
        },
        "m7_fv_mass": {
            "enabled": True,
            "w_fv_mass_max": 0.30,
            "fv_start": 8000,
            "fv_ramp_steps": 6000,
            "fv_n_cells": 1024,
            "fv_h_tilde": 0.015625,
            "fv_dt_tilde": 0.0015850067138671875,
            "fv_w_water": 0.05,
            "use_front_band_weight": True,
            "front_center": 0.175,
            "front_sigma": 0.075,
            "front_weight_floor": 0.05,
        },
        "physical_parameters": {
            "L_ref": 5.0,
            "T_ref": 31544.99609375,
            "K": 1.0e-14,
            "permeability_field_id": "K-fixed-001",
            "permeability_field_sha256": "a" * 64,
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
            "A_time": 2.9352545271336377,
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
    }


def _validate_formal(task, config):
    return validate_checkpoint_config(
        task,
        config,
        field_id="K-fixed-001",
        field_sha256="a" * 64,
        field_metadata_sha256="c" * 64,
        reference_provenance_sha256="d" * 64,
        dataset_sha256=DATASET_SHA256,
    )


def _set_nested(mapping, path, value):
    target = mapping
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value


def _role_config(template, role, exp_name, leave_out):
    config = copy.deepcopy(template)
    task = TaskSpec(role, exp_name, "BASE", leave_out, 0)
    config.update({
        "model_role": role,
        "exp_name": exp_name,
        "leave_out": leave_out,
        "task_index": task.array_index,
        "task_key": task.key,
        "run_id": task.run_id,
    })
    switches = {
        "baseline": {
            "model_type": "single",
            "architecture": (False, False, False, False, False),
            "front": (False, False),
            "pair": False,
            "rar": False,
            "fv": False,
        },
        "complete": {
            "model_type": "two_cd",
            "architecture": (True, True, True, True, False),
            "front": (True, True),
            "pair": True,
            "rar": True,
            "fv": True,
        },
        "no_representation": {
            "model_type": "plain_twonet",
            "architecture": (True, False, False, False, True),
            "front": (True, True),
            "pair": True,
            "rar": True,
            "fv": True,
        },
        "no_front_supervision": {
            "model_type": "two_cd",
            "architecture": (True, True, True, True, False),
            "front": (False, False),
            "pair": True,
            "rar": True,
            "fv": True,
        },
        "no_local_fv": {
            "model_type": "two_cd",
            "architecture": (True, True, True, True, False),
            "front": (True, True),
            "pair": True,
            "rar": True,
            "fv": False,
        },
    }[role]
    config["model_type"] = switches["model_type"]
    arch = config["architecture"]
    (
        arch["use_twonet"],
        arch["use_msfourier"],
        arch["decouple_ps_input"],
        arch["use_coarse_detail_s_branch"],
        arch["use_plain_twonet"],
    ) = switches["architecture"]
    (
        config["m4_front_plume_loss"]["front_loss"],
        config["m4_front_plume_loss"]["plume_loss"],
    ) = switches["front"]
    config["m5_pairwise_front_gradient"]["enabled"] = switches["pair"]
    config["m6_rar"]["enabled"] = switches["rar"]
    config["m7_fv_mass"]["enabled"] = switches["fv"]
    return task, config


def test_dataset_loader_applies_common_time_crop_before_evaluation(tmp_path):
    path = tmp_path / "tiny.pt"
    torch.save(
        {
            "arrays": {
                "x": np.array([0.0, 1.0, 2.0]),
                "y": np.array([3.0, 4.0, 5.0]),
                "t": np.array([0.0, 31544.0, 32204.0]),
                "p": np.array([1.0, 2.0, 3.0]),
                "Sco2": np.array([0.0, 0.1, 0.2]),
                "non_row_metadata": np.array([7.0, 8.0]),
            },
            "time_unique": np.array([0.0, 31544.0, 32204.0]),
        },
        path,
    )

    arrays, time_unique = load_dataset_pack(path, analysis_t_max_s=31544.99609375)

    np.testing.assert_array_equal(arrays["t"], np.array([0.0, 31544.0]))
    np.testing.assert_array_equal(arrays["x"], np.array([0.0, 1.0]))
    np.testing.assert_array_equal(arrays["non_row_metadata"], np.array([7.0, 8.0]))
    np.testing.assert_array_equal(time_unique, np.array([0.0, 31544.0]))


def test_contour_tag_preserves_the_registered_saturation_level():
    assert saturation_level_tag(0.175) == "S0175"
    assert saturation_level_tag(0.40) == "S040"


def test_front_and_pair_summaries_merge_to_one_row_per_checkpoint():
    rows = merge_run_summaries(
        [{"run_id": "r0", "front_band_rmse_mean": 0.1}],
        [{"run_id": "r0", "pairgrad_rmse_mean": 0.2}],
    )

    assert rows == [{
        "run_id": "r0",
        "front_band_rmse_mean": 0.1,
        "pairgrad_rmse_mean": 0.2,
    }]


@pytest.mark.parametrize(
    ("key", "bad_value"),
    [
        ("configuration", "H"),
        ("task_key", "complete:seed42"),
        ("run_id", "complete_seed42"),
        ("pc_enabled", True),
        ("pc_entry_pressure_pa", 10.0),
        ("analysis_t_max_s", 32204.078125),
        ("rar_global_select_count", 7500),
        ("permeability_field_sha256", "b" * 64),
        ("reference_dataset_sha256", "c" * 64),
        ("reference_provenance_verified", False),
        ("training_script_sha256", "not-a-sha"),
        ("gpu_model", "CPU"),
        ("slurm_job_id", ""),
    ],
)
def test_checkpoint_validator_fails_closed_on_scientific_drift(
    task, checkpoint_config, key, bad_value
):
    bad = copy.deepcopy(checkpoint_config)
    bad[key] = bad_value

    with pytest.raises(ValueError, match=key):
        validate_checkpoint_config(
            task,
            bad,
            field_sha256="a" * 64,
            dataset_sha256=DATASET_SHA256,
        )


def test_checkpoint_validator_accepts_exact_registered_formal_task(task, checkpoint_config):
    resolved = _validate_formal(task, checkpoint_config)

    assert resolved["model_role"] == "complete"
    assert resolved["analysis_t_max_s"] == 31544.99609375


def test_checkpoint_validator_requires_an_untampered_frozen_eos_record(
    task, checkpoint_config
):
    missing = copy.deepcopy(checkpoint_config)
    missing.pop("eos_record")
    with pytest.raises(ValueError, match="eos_record"):
        validate_checkpoint_config(
            task,
            missing,
            field_sha256="a" * 64,
            dataset_sha256=DATASET_SHA256,
        )

    tampered = copy.deepcopy(checkpoint_config)
    tampered["eos_record"]["model"]["coefficients"][0] += 0.01
    with pytest.raises(ValueError, match="SHA-256"):
        validate_checkpoint_config(
            task,
            tampered,
            field_sha256="a" * 64,
            dataset_sha256=DATASET_SHA256,
        )


@pytest.mark.parametrize(
    ("path", "bad_value"),
    [
        (("physical_parameters", "T_ref"), 1.0e5),
        (("physical_parameters", "L_ref"), 10.0),
        (("physical_parameters", "k_ref"), 2.0e-14),
        (("physical_parameters", "pc_enable"), True),
        (("physical_parameters", "permeability_field_id"), "other-field"),
        (("physical_parameters", "mu_c"), 1.0e-5),
        (("physical_parameters", "krw0"), 0.5),
        (("physical_parameters", "r_well"), 0.6),
        (("physical_parameters", "inj_center"), [0.1, 0.0]),
        (("physical_parameters", "domain_x_m"), [0.0, 10.0]),
        (("m4_front_plume_loss", "front_s_min"), 0.10),
        (("m4_front_plume_loss", "front_center"), 0.20),
        (("m4_front_plume_loss", "front_sigma"), 0.10),
        (("m7_fv_mass", "fv_h_tilde"), 0.02),
        (("m7_fv_mass", "fv_dt_tilde"), 0.01),
        (("m7_fv_mass", "fv_n_cells"), 512),
        (("architecture", "use_msfourier"), False),
        (("m4_front_plume_loss", "front_loss"), False),
        (("m6_rar", "enabled"), False),
        (("m7_fv_mass", "enabled"), False),
        (("training_schedules", "use_beta_curriculum"), True),
        (("saturation_ic", "s_co2_output_max"), 1.0),
    ],
)
def test_checkpoint_validator_rejects_nested_evaluator_config_drift(
    task, checkpoint_config, path, bad_value
):
    bad = copy.deepcopy(checkpoint_config)
    _set_nested(bad, path, bad_value)

    with pytest.raises(ValueError, match=path[-1]):
        _validate_formal(task, bad)


@pytest.mark.parametrize(
    ("path", "bad_value"),
    [
        (("data_time_crop", "rows_before"), 173688978),
        (("data_time_crop", "rows_after"), 173688978),
        (("data_time_crop", "rows_removed"), 1),
        (("data_time_crop", "cutoff_s"), 32204.078125),
        (("data_time_crop", "first_retained_time_s"), 1.0),
        (("data_time_crop", "last_retained_time_s"), 31544.99609375),
    ],
)
def test_checkpoint_validator_rejects_time_crop_drift(
    task, checkpoint_config, path, bad_value
):
    bad = copy.deepcopy(checkpoint_config)
    _set_nested(bad, path, bad_value)

    with pytest.raises(ValueError, match=path[-1]):
        _validate_formal(task, bad)


def test_checkpoint_validator_rejects_retained_time_count_drift(task, checkpoint_config):
    bad = copy.deepcopy(checkpoint_config)
    bad["retained_unique_time_count"] = 775

    with pytest.raises(ValueError, match="retained_unique_time_count"):
        _validate_formal(task, bad)


@pytest.mark.parametrize(
    ("role", "exp_name", "leave_out"),
    [
        ("baseline", "M0", "NONE"),
        ("complete", "M7", "NONE"),
        ("no_representation", "M7", "PLAIN_TWONET"),
        ("no_front_supervision", "M7", "FRONT_PLUME"),
        ("no_local_fv", "M7", "FV"),
    ],
)
def test_checkpoint_validator_accepts_only_the_registered_role_switches(
    checkpoint_config, role, exp_name, leave_out
):
    task, config = _role_config(checkpoint_config, role, exp_name, leave_out)

    assert _validate_formal(task, config)["model_role"] == role


def test_checkpoint_validator_binds_sidecar_and_provenance_hashes(task, checkpoint_config):
    with pytest.raises(ValueError, match="field_metadata_sha256"):
        validate_checkpoint_config(
            task,
            checkpoint_config,
            field_id="K-fixed-001",
            field_sha256="a" * 64,
            field_metadata_sha256="e" * 64,
            reference_provenance_sha256="d" * 64,
            dataset_sha256=DATASET_SHA256,
        )

    with pytest.raises(ValueError, match="reference_provenance_sha256"):
        validate_checkpoint_config(
            task,
            checkpoint_config,
            field_id="K-fixed-001",
            field_sha256="a" * 64,
            field_metadata_sha256="c" * 64,
            reference_provenance_sha256="e" * 64,
            dataset_sha256=DATASET_SHA256,
        )


def test_metric_record_maps_exact_five_canonical_metrics(task):
    record = build_metric_record(
        task,
        global_row={"S_rel_l2": "0.11"},
        front_row={
            "front_band_wide_005_030_rmse_mean": "0.22",
            "contour_chamfer_S0175_mean": "0.33",
        },
        physics_row={
            "local_FV_CO2_RMSE": "0.44",
            "CO2_mass_rel_error_mean": "0.55",
        },
    )

    assert record == {
        "configuration": "K",
        "model_role": "complete",
        "exp_name": "M7",
        "training_strategy": "BASE",
        "leave_out": "NONE",
        "seed": 0,
        "task_index": 4,
        "run_id": "complete_seed0",
        **FORMAL_RESULT_CONTRACT,
        "saturation_rel_l2": 0.11,
        "front_band_rmse": 0.22,
        "contour_chamfer_s0175": 0.33,
        "local_fv_co2_rmse": 0.44,
        "co2_mass_rel_error": 0.55,
    }


def test_metric_record_rejects_missing_or_nonfinite_values(task):
    valid_global = {"S_rel_l2": 0.1}
    valid_front = {
        "front_band_wide_005_030_rmse_mean": 0.2,
        "contour_chamfer_S0175_mean": 0.3,
    }
    valid_physics = {"local_FV_CO2_RMSE": 0.4, "CO2_mass_rel_error_mean": 0.5}

    with pytest.raises(ValueError, match="contour_chamfer_S0175_mean"):
        build_metric_record(
            task,
            global_row=valid_global,
            front_row={"front_band_wide_005_030_rmse_mean": 0.2},
            physics_row=valid_physics,
        )

    with pytest.raises(ValueError, match="finite"):
        build_metric_record(
            task,
            global_row=valid_global,
            front_row=valid_front,
            physics_row={**valid_physics, "local_FV_CO2_RMSE": np.nan},
        )


def test_metric_payload_binds_checkpoint_config_field_and_dataset_hashes(task):
    record = build_metric_record(
        task,
        global_row={"S_rel_l2": 0.1},
        front_row={
            "front_band_wide_005_030_rmse_mean": 0.2,
            "contour_chamfer_S0175_mean": 0.3,
        },
        physics_row={"local_FV_CO2_RMSE": 0.4, "CO2_mass_rel_error_mean": 0.5},
    )
    payload = attach_metric_evidence(
        record,
        checkpoint_sha256="1" * 64,
        resolved_config_sha256="2" * 64,
        permeability_field_sha256="3" * 64,
        field_metadata_sha256="4" * 64,
        reference_dataset_sha256=DATASET_SHA256,
        reference_provenance_sha256="5" * 64,
        evaluation_manifest_sha256="6" * 64,
        eos_record_sha256="7" * 64,
        code_manifest_sha256="8" * 64,
    )

    assert payload["evidence"]["checkpoint_sha256"] == "1" * 64
    assert extract_metric_record(
        payload,
        checkpoint_sha256="1" * 64,
        resolved_config_sha256="2" * 64,
        permeability_field_sha256="3" * 64,
        field_metadata_sha256="4" * 64,
        reference_dataset_sha256=DATASET_SHA256,
        reference_provenance_sha256="5" * 64,
        evaluation_manifest_sha256="6" * 64,
        eos_record_sha256="7" * 64,
        code_manifest_sha256="8" * 64,
    ) == record

    with pytest.raises(ValueError, match="permeability_field_sha256"):
        extract_metric_record(
            payload,
            checkpoint_sha256="1" * 64,
            resolved_config_sha256="2" * 64,
            permeability_field_sha256="4" * 64,
            field_metadata_sha256="4" * 64,
            reference_dataset_sha256=DATASET_SHA256,
            reference_provenance_sha256="5" * 64,
            evaluation_manifest_sha256="6" * 64,
            eos_record_sha256="7" * 64,
            code_manifest_sha256="8" * 64,
        )


def test_single_row_reader_rejects_zero_or_multiple_runs(tmp_path):
    path = tmp_path / "metrics.csv"
    path.write_text("run_id,value\n", encoding="utf-8")
    with pytest.raises(ValueError, match="exactly one"):
        read_single_csv_row(path)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["run_id", "value"])
        writer.writeheader()
        writer.writerows([{"run_id": "a", "value": 1}, {"run_id": "b", "value": 2}])
    with pytest.raises(ValueError, match="exactly one"):
        read_single_csv_row(path)


def test_evaluator_row_identity_cannot_be_misassigned_to_another_task(task):
    row = {
        "exp_name": "M7",
        "training_strategy": "BASE",
        "leave_out": "NONE",
        "seed": "0",
    }
    validate_evaluator_row_identity(task, row)

    with pytest.raises(ValueError, match="seed"):
        validate_evaluator_row_identity(task, {**row, "seed": "42"})


def test_evaluator_commands_share_cutoff_assets_and_one_checkpoint_scope(tmp_path, task):
    commands = build_evaluator_commands(
        task,
        python_executable="/env/bin/python",
        project_root=ROOT,
        checkpoint_dir=tmp_path / "run",
        data_path=tmp_path / "reference.pt",
        field_path=tmp_path / "field.npz",
        field_metadata_path=tmp_path / "field.json",
        out_dir=tmp_path / "evaluation",
        device="cuda",
        gpu_id=0,
    )

    assert tuple(commands) == ("global", "front", "physics")
    for command in commands.values():
        assert command[0] == "/env/bin/python"
        assert command[command.index("--analysis-t-max-s") + 1] == "31544.99609375"
        assert command[command.index("--ckpt-dir") + 1] == str(tmp_path / "run")
        assert command[command.index("--experiments") + 1] == "M7"
        assert command[command.index("--seed") + 1] == "0"
    assert commands["physics"][commands["physics"].index("--k-field") + 1] == str(tmp_path / "field.npz")
    assert commands["physics"][commands["physics"].index("--k-field-metadata") + 1] == str(tmp_path / "field.json")

    expected_common = {
        "--infer-batch-size": "65536",
        "--max-time-steps": "776",
        "--max-sort-n": "500000000",
        "--random-seed": "0",
        "--time-indices": "t0,early,middle,late,final",
        "--time-unit": "days",
        "--figure-dpi": "600",
    }
    for command in commands.values():
        for flag, expected in expected_common.items():
            assert command[command.index(flag) + 1] == expected

    expected_global = {
        "--sample-per-time": "20000",
        "--pressure-display-unit": "MPa",
        "--s-ic-co2": "0.0",
        "--Snr": "0.0",
        "--Sw-irr": "0.2",
        "--plot-label-mode": "auto",
    }
    expected_front = {
        "--sample-per-time": "20000",
        "--grid-nx": "420",
        "--grid-ny": "420",
        "--max-interp-input-points": "120000",
        "--front-low-wide": "0.05",
        "--front-high-wide": "0.30",
        "--front-low-narrow": "0.10",
        "--front-high-narrow": "0.25",
        "--contour-levels": "0.175",
        "--s-ic-co2": "0.0",
    }
    expected_physics = {
        "--pde-points": "4096",
        "--pde-batch-size": "512",
        "--bc-points": "2048",
        "--outer-points-each": "512",
        "--ic-points": "4096",
        "--mass-sample-per-time": "30000",
        "--mass-grid-nx": "256",
        "--mass-grid-ny": "256",
        "--fv-n-cells-eval": "2048",
        "--fv-batch-size": "256",
        "--fv-h-tilde": "0.015625",
        "--fv-dt-tilde": "0.0015850067138671875",
        "--s-ic-co2": "0.0",
        "--Snr": "0.0",
        "--Sw-irr": "0.2",
    }
    for name, expected_flags in (
        ("global", expected_global),
        ("front", expected_front),
        ("physics", expected_physics),
    ):
        command = commands[name]
        for flag, expected in expected_flags.items():
            assert command[command.index(flag) + 1] == expected

def test_evaluator_clis_expose_frozen_k_inputs():
    env = os.environ.copy()
    for name in (
        "evaluate_global_accuracy_M0_M7_L6_dS_shortlabels.py",
        "evaluate_front_plume_M0_M7_L6_dS.py",
        "evaluate_physics_M0_M7_L6_dS.py",
    ):
        result = subprocess.run(
            [sys.executable, str(RESULTS / name), "--help"],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "--analysis-t-max-s" in result.stdout
        if name.startswith("evaluate_physics"):
            assert "--k-field" in result.stdout
            assert "--k-field-metadata" in result.stdout


def test_all_formal_evaluators_load_checkpoint_weights_strictly():
    for name in (
        "evaluate_global_accuracy_M0_M7_L6_dS_shortlabels.py",
        "evaluate_front_plume_M0_M7_L6_dS.py",
        "evaluate_physics_M0_M7_L6_dS.py",
    ):
        source = (RESULTS / name).read_text(encoding="utf-8")
        assert "strict=False" not in source
        assert "strict=True" in source


def test_front_evaluation_does_not_use_process_randomized_python_hash_for_sampling():
    source = (RESULTS / "evaluate_front_plume_M0_M7_L6_dS.py").read_text(encoding="utf-8")

    assert "hash(spec.run_id)" not in source
