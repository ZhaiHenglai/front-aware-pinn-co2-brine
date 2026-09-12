import hashlib
import json
import subprocess
import sys

import numpy as np
import pytest

from configuration_k import DATASET_SHA256
from permeability_field_k import validate_asset_bundle


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _synthetic_assets(tmp_path, *, verified=False, pc_enabled=False):
    dataset = tmp_path / "dataset.pt"
    dataset.write_bytes(b"small synthetic cache identity")
    dataset_sha256 = _sha256(dataset)

    field = tmp_path / "field.npz"
    k = np.array([[1e-14, 2e-14], [4e-14, 8e-14]], dtype=np.float64)
    np.savez(
        field,
        x_grid=np.array([0.0, 5.0]),
        y_grid=np.array([0.0, 5.0]),
        permeability=k,
        field_id=np.array("field-001"),
        field_seed=np.array(9, dtype=np.int64),
        units=np.array("m^2"),
        generation_method=np.array("fixture"),
        interpolation_method=np.array("bilinear interpolation of log(K)"),
        source_dataset_sha256=np.array(dataset_sha256),
        source_time_s=np.array(0.0, dtype=np.float64),
        source_permeability_key=np.array("permeability"),
    )
    metadata = tmp_path / "field.json"
    metadata.write_text(
        json.dumps(
            {
                "configuration": "K",
                "field_id": "field-001",
                "field_seed": 9,
                "units": "m^2",
                "generation_method": "fixture",
                "generation_parameters": {"fixture": "2x2 log-linear corners"},
                "interpolation_method": "bilinear interpolation of log(K)",
                "source_dataset_sha256": dataset_sha256,
                "source_time_s": 0.0,
                "source_permeability_key": "permeability",
                "statistics_weighting": "area-weighted trapezoidal quadrature",
                "boundary_and_well_treatment": "full grid",
                "directional_correlation_lengths_m": None,
                "directional_correlation_lengths_note": "not defined for fixture",
                "npz_sha256": _sha256(field),
                "shape": [2, 2],
                "x_range_m": [0.0, 5.0],
                "y_range_m": [0.0, 5.0],
                "statistics": {
                    "minimum_m2": float(k.min()),
                    "maximum_m2": float(k.max()),
                    "arithmetic_mean_m2": float(k.mean()),
                    "geometric_mean_m2": float(np.exp(np.log(k).mean())),
                    "log_k_variance": float(np.var(np.log(k))),
                },
            }
        ),
        encoding="utf-8",
    )
    evidence = tmp_path / "simulator_input.cfg"
    evidence.write_text("capillary_pressure = disabled\n", encoding="utf-8")
    provenance = tmp_path / "provenance.json"
    provenance.write_text(
        json.dumps(
            {
                "configuration": "K",
                "reference_dataset_sha256": DATASET_SHA256,
                "verified": verified,
                "capillary_pressure_enabled": pc_enabled,
                "permeability_field_id": "field-001",
                "permeability_field_sha256": _sha256(field),
                "simulator": "fixture-simulator",
                "source_account": "fixture-account",
                "source_case_directory": "/fixture/case-k",
                "verification_notes": "checked by test fixture",
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
                        "sha256": _sha256(evidence),
                        "description": "fixture simulator input explicitly disables capillary pressure",
                    }
                ] if verified else [],
            }
        ),
        encoding="utf-8",
    )
    return dataset, field, metadata, provenance


def test_formal_assets_reject_unverified_reference(tmp_path):
    dataset, field, metadata, provenance = _synthetic_assets(tmp_path, verified=False)

    with pytest.raises(ValueError, match="verified no-Pc provenance"):
        validate_asset_bundle(
            dataset,
            field,
            metadata,
            provenance,
            run_kind="formal",
            expected_dataset_sha256=_sha256(dataset),
        )


def test_verified_formal_bundle_reports_all_hashes(tmp_path):
    dataset, field, metadata, provenance = _synthetic_assets(tmp_path, verified=True)

    report = validate_asset_bundle(
        dataset,
        field,
        metadata,
        provenance,
        run_kind="formal",
        expected_dataset_sha256=_sha256(dataset),
    )

    assert report["status"] == "pass"
    assert report["run_kind"] == "formal"
    assert report["dataset"]["sha256"] == _sha256(dataset)
    assert report["field"]["field_id"] == "field-001"
    assert report["field_metadata_sha256"] == _sha256(metadata)
    assert report["reference_provenance_sha256"] == _sha256(provenance)
    assert report["reference_provenance"]["verified"] is True


def test_bundle_rejects_reference_provenance_for_a_different_field(tmp_path):
    dataset, field, metadata, provenance = _synthetic_assets(tmp_path, verified=True)
    record = json.loads(provenance.read_text(encoding="utf-8"))
    record["permeability_field_sha256"] = "f" * 64
    provenance.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ValueError, match="provenance permeability field SHA-256"):
        validate_asset_bundle(
            dataset,
            field,
            metadata,
            provenance,
            run_kind="formal",
            expected_dataset_sha256=_sha256(dataset),
        )


def test_bundle_rejects_unique_field_derived_from_a_different_dataset(tmp_path):
    dataset, field, metadata, provenance = _synthetic_assets(tmp_path, verified=True)
    wrong_dataset_sha = "a" * 64
    with np.load(field, allow_pickle=False) as pack:
        payload = {key: pack[key] for key in pack.files}
    payload["source_dataset_sha256"] = np.array(wrong_dataset_sha)
    np.savez(field, **payload)
    record = json.loads(metadata.read_text(encoding="utf-8"))
    record["source_dataset_sha256"] = wrong_dataset_sha
    record["npz_sha256"] = _sha256(field)
    metadata.write_text(json.dumps(record), encoding="utf-8")
    provenance_record = json.loads(provenance.read_text(encoding="utf-8"))
    provenance_record["permeability_field_sha256"] = _sha256(field)
    provenance.write_text(json.dumps(provenance_record), encoding="utf-8")

    with pytest.raises(ValueError, match="derived from the validated reference dataset"):
        validate_asset_bundle(
            dataset,
            field,
            metadata,
            provenance,
            run_kind="formal",
            expected_dataset_sha256=_sha256(dataset),
        )


def test_formal_bundle_rejects_unverified_h_alignment_dimension(tmp_path):
    dataset, field, metadata, provenance = _synthetic_assets(tmp_path, verified=True)
    record = json.loads(provenance.read_text(encoding="utf-8"))
    record["alignment_with_configuration_h"]["fluid_properties"] = False
    provenance.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ValueError, match="fluid_properties"):
        validate_asset_bundle(
            dataset,
            field,
            metadata,
            provenance,
            run_kind="formal",
            expected_dataset_sha256=_sha256(dataset),
        )


def test_smoke_allows_unverified_reference_but_never_pc_enabled(tmp_path):
    dataset, field, metadata, provenance = _synthetic_assets(tmp_path, verified=False)
    report = validate_asset_bundle(
        dataset,
        field,
        metadata,
        provenance,
        run_kind="smoke",
        expected_dataset_sha256=_sha256(dataset),
    )
    assert report["reference_provenance"]["verified"] is False

    provenance.write_text(
        provenance.read_text(encoding="utf-8").replace(
            '"capillary_pressure_enabled": false',
            '"capillary_pressure_enabled": true',
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="capillary pressure disabled"):
        validate_asset_bundle(
            dataset,
            field,
            metadata,
            provenance,
            run_kind="smoke",
            expected_dataset_sha256=_sha256(dataset),
        )


def test_asset_cli_returns_nonzero_and_json_when_formal_provenance_is_unverified(tmp_path):
    dataset, field, metadata, provenance = _synthetic_assets(tmp_path, verified=False)
    command = [
        sys.executable,
        "tools/validate_k_assets.py",
        "--dataset",
        str(dataset),
        "--field",
        str(field),
        "--field-metadata",
        str(metadata),
        "--reference-provenance",
        str(provenance),
    ]

    result = subprocess.run(command, text=True, capture_output=True, check=False)

    assert result.returncode != 0
    report = json.loads(result.stdout)
    assert report["status"] == "fail"
    assert "SHA-256" in report["error"] or "verified no-Pc provenance" in report["error"]
