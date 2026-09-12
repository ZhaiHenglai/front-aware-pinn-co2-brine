import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import configuration_k

from configuration_k import (
    ANALYSIS_T_MAX_S,
    CANONICAL_METRICS,
    DATASET_SHA256,
    FV_DT_TILDE,
    RAR_CANDIDATE_COUNT,
    RAR_EVERY,
    RAR_GLOBAL_SELECT_COUNT,
    RAR_RESIDUAL_SELECT_COUNT,
    RAR_START,
    T_REF_S,
    TaskSpec,
    all_formal_tasks,
    crop_row_aligned_arrays,
    task_from_index,
    validate_no_pc,
    validate_reference_provenance,
    validate_frozen_crop_report,
    validate_task,
)


def test_formal_matrix_is_exact_and_stably_indexed():
    tasks = all_formal_tasks()

    assert len(tasks) == 20
    assert len({task.key for task in tasks}) == 20
    assert tasks[0] == TaskSpec("baseline", "M0", "BASE", "NONE", 0)
    assert tasks[3] == TaskSpec("baseline", "M0", "BASE", "NONE", 42)
    assert tasks[4] == TaskSpec("complete", "M7", "BASE", "NONE", 0)
    assert tasks[-1] == TaskSpec("no_local_fv", "M7", "BASE", "FV", 42)
    assert task_from_index(19) == tasks[-1]


def test_formal_matrix_rejects_out_of_range_index():
    for index in (-1, 20):
        with pytest.raises(ValueError, match="0..19"):
            task_from_index(index)


def test_validate_task_rejects_unregistered_role_parameter_combination():
    with pytest.raises(ValueError, match="registered Configuration K task"):
        validate_task("complete", "M7", "BASE", "FV", 0)


def test_pc_is_fail_closed():
    validate_no_pc({"PINN_PC_ENABLE": "0", "PINN_PC_ENTRY": "0"})
    validate_no_pc({"PINN_PC_ENABLE": "false", "PINN_PC_ENTRY": "0.0"})

    with pytest.raises(ValueError, match="capillary pressure"):
        validate_no_pc({"PINN_PC_ENABLE": "1", "PINN_PC_ENTRY": "0"})
    with pytest.raises(ValueError, match="PINN_PC_ENTRY"):
        validate_no_pc({"PINN_PC_ENABLE": "0", "PINN_PC_ENTRY": "50000"})


def test_crop_applies_one_mask_to_every_row_aligned_numpy_array():
    arrays = {
        "t": np.array([0.0, T_REF_S, T_REF_S + 1.0]),
        "x": np.array([1.0, 2.0, 3.0]),
        "p": np.array([[10.0], [20.0], [30.0]]),
        "metadata": np.array([7, 8]),
    }

    cropped, report = crop_row_aligned_arrays(arrays, T_REF_S)

    assert cropped["t"].tolist() == [0.0, T_REF_S]
    assert cropped["x"].tolist() == [1.0, 2.0]
    assert cropped["p"].tolist() == [[10.0], [20.0]]
    assert cropped["metadata"].tolist() == [7, 8]
    assert report.rows_before == 3
    assert report.rows_after == 2
    assert report.rows_removed == 1
    assert report.last_retained_time_s == T_REF_S


def test_crop_rejects_nonfinite_time_values():
    arrays = {"t": np.array([0.0, np.nan]), "x": np.array([1.0, 2.0])}

    with pytest.raises(ValueError, match="finite"):
        crop_row_aligned_arrays(arrays, ANALYSIS_T_MAX_S)


def test_frozen_crop_report_rejects_a_different_cache_shape():
    expected = configuration_k.FROZEN_DATA_TIME_CROP
    report = configuration_k.CropReport(**expected)
    validate_frozen_crop_report(report, configuration_k.RETAINED_UNIQUE_TIME_COUNT)

    altered = configuration_k.CropReport(**{**expected, "rows_after": expected["rows_after"] - 1})
    with pytest.raises(ValueError, match="rows_after"):
        validate_frozen_crop_report(altered, configuration_k.RETAINED_UNIQUE_TIME_COUNT)
    with pytest.raises(ValueError, match="retained_unique_time_count"):
        validate_frozen_crop_report(report, configuration_k.RETAINED_UNIQUE_TIME_COUNT - 1)


def test_frozen_numeric_contract_matches_configuration_h():
    assert T_REF_S == 31544.99609375
    assert ANALYSIS_T_MAX_S == 31544.99609375
    assert RAR_START == 3500
    assert RAR_EVERY == 100
    assert RAR_CANDIDATE_COUNT == 30000
    assert RAR_RESIDUAL_SELECT_COUNT == 22500
    assert RAR_GLOBAL_SELECT_COUNT == 0
    assert FV_DT_TILDE == 0.0015850067138671875
    assert DATASET_SHA256 == "2132a9be710e2f257ea397cd080654e689093d9c561be3edb5be1e471283e761"
    assert configuration_k.DATASET_BASENAME == "tables_cache_0_775_step1.pt"
    assert configuration_k.DATASET_RELATIVE_PATH == "data/tables_cache_0_775_step1.pt"
    assert configuration_k.FROZEN_DATA_TIME_CROP == {
        "rows_before": 173_688_979,
        "rows_after": 173_688_979,
        "rows_removed": 0,
        "cutoff_s": 31544.99609375,
        "first_retained_time_s": 0.0,
        "last_retained_time_s": 30_986.943359375,
    }
    assert configuration_k.RETAINED_UNIQUE_TIME_COUNT == 776
    assert CANONICAL_METRICS == (
        "saturation_rel_l2",
        "front_band_rmse",
        "contour_chamfer_s0175",
        "local_fv_co2_rmse",
        "co2_mass_rel_error",
    )


def _write_provenance(path: Path, *, verified, pc_enabled):
    evidence = path.parent / "simulator_input.cfg"
    evidence.write_text("capillary_pressure = disabled\n", encoding="utf-8")
    path.write_text(
        json.dumps(
            {
                "configuration": "K",
                "reference_dataset_sha256": DATASET_SHA256,
                "verified": verified,
                "capillary_pressure_enabled": pc_enabled,
                "simulator": "fixture",
                "source_account": "fixture-account",
                "source_case_directory": "/fixture",
                "verification_notes": "fixture verification",
                "alignment_with_configuration_h": {
                    "geometry": True,
                    "boundary_conditions": True,
                    "initial_conditions": True,
                    "relative_permeability_model": True,
                    "fluid_properties": True,
                    "time_window_covers_common_analysis": True,
                    "numerical_quality_acceptable": True,
                },
                "evidence": [
                    {
                        "path": evidence.name,
                        "sha256": hashlib.sha256(evidence.read_bytes()).hexdigest(),
                        "description": "fixture simulator input explicitly disables capillary pressure",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_formal_reference_requires_verified_no_pc_provenance(tmp_path):
    path = tmp_path / "provenance.json"
    _write_provenance(path, verified=False, pc_enabled=False)

    with pytest.raises(ValueError, match="verified no-Pc provenance"):
        validate_reference_provenance(path, allow_unverified=False)

    _write_provenance(path, verified=True, pc_enabled=True)
    with pytest.raises(ValueError, match="capillary pressure disabled"):
        validate_reference_provenance(path, allow_unverified=False)


def test_smoke_may_use_explicitly_unverified_but_not_pc_enabled_reference(tmp_path):
    path = tmp_path / "provenance.json"
    _write_provenance(path, verified=False, pc_enabled=False)

    record = validate_reference_provenance(path, allow_unverified=True)
    assert record["verified"] is False

    _write_provenance(path, verified=False, pc_enabled=True)
    with pytest.raises(ValueError, match="capillary pressure disabled"):
        validate_reference_provenance(path, allow_unverified=True)


def test_verified_reference_rejects_unhashed_or_nonlocal_evidence(tmp_path):
    path = tmp_path / "provenance.json"
    _write_provenance(path, verified=True, pc_enabled=False)
    record = json.loads(path.read_text(encoding="utf-8"))
    record["evidence"] = ["simulator/input.cfg"]
    path.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ValueError, match="evidence entry"):
        validate_reference_provenance(path, allow_unverified=False)


def test_verified_reference_requires_source_account(tmp_path):
    path = tmp_path / "provenance.json"
    _write_provenance(path, verified=True, pc_enabled=False)
    record = json.loads(path.read_text(encoding="utf-8"))
    record.pop("source_account", None)
    path.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ValueError, match="source_account"):
        validate_reference_provenance(path, allow_unverified=False)
