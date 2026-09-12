import hashlib
import json

import numpy as np
import pytest
import torch

from permeability_field_k import (
    PermeabilityFieldK,
    area_weighted_field_statistics,
    load_field_grid,
    validate_field_artifacts,
)


SOURCE_DATASET_SHA256 = "d" * 64


def _sha256(path):
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _npz_payload(x, y, permeability):
    return {
        "x_grid": x,
        "y_grid": y,
        "permeability": permeability,
        "field_id": np.array("synthetic-fixed-field"),
        "field_seed": np.array(17, dtype=np.int64),
        "units": np.array("m^2"),
        "generation_method": np.array("synthetic test fixture"),
        "interpolation_method": np.array("bilinear interpolation of log(K)"),
        "source_dataset_sha256": np.array(SOURCE_DATASET_SHA256),
        "source_time_s": np.array(0.0, dtype=np.float64),
        "source_permeability_key": np.array("permeability"),
    }


def _write_valid_field(tmp_path):
    npz = tmp_path / "K_field_unique_grid.npz"
    x = np.array([0.0, 5.0], dtype=np.float64)
    y = np.array([0.0, 5.0], dtype=np.float64)
    permeability = np.array([[1e-14, 2e-14], [4e-14, 8e-14]], dtype=np.float64)
    np.savez(npz, **_npz_payload(x, y, permeability))
    metadata = tmp_path / "K_field_unique_grid.json"
    metadata.write_text(
        json.dumps(
            {
                "configuration": "K",
                "field_id": "synthetic-fixed-field",
                "field_seed": 17,
                "units": "m^2",
                "generation_method": "synthetic test fixture",
                "generation_parameters": {"fixture": "2x2 log-linear corners"},
                "interpolation_method": "bilinear interpolation of log(K)",
                "source_dataset_sha256": SOURCE_DATASET_SHA256,
                "source_time_s": 0.0,
                "source_permeability_key": "permeability",
                "statistics_weighting": "area-weighted trapezoidal quadrature",
                "boundary_and_well_treatment": "values defined on full grid",
                "directional_correlation_lengths_m": None,
                "directional_correlation_lengths_note": "not defined for synthetic fixture",
                "npz_sha256": _sha256(npz),
                "shape": [2, 2],
                "x_range_m": [0.0, 5.0],
                "y_range_m": [0.0, 5.0],
                "statistics": {
                    "minimum_m2": 1e-14,
                    "maximum_m2": 8e-14,
                    "arithmetic_mean_m2": 3.75e-14,
                    "geometric_mean_m2": float(np.exp(np.log(permeability).mean())),
                    "log_k_variance": float(np.var(np.log(permeability))),
                },
            }
        ),
        encoding="utf-8",
    )
    return npz, metadata, x, y, permeability


def test_log_bilinear_field_is_positive_and_hits_grid_corners(tmp_path):
    _, _, x, y, permeability = _write_valid_field(tmp_path)
    field = PermeabilityFieldK.from_arrays(x, y, permeability)

    out = field(
        torch.tensor([[0.0], [1.0]], dtype=torch.float64),
        torch.tensor([[0.0], [1.0]], dtype=torch.float64),
    )

    assert torch.all(out > 0)
    assert torch.allclose(
        out[:, 0],
        torch.tensor([1e-14, 8e-14], dtype=torch.float64),
        rtol=1e-12,
        atol=0.0,
    )


def test_log_bilinear_field_uses_geometric_interpolation_at_center(tmp_path):
    _, _, x, y, permeability = _write_valid_field(tmp_path)
    field = PermeabilityFieldK.from_arrays(x, y, permeability)

    out = field(torch.tensor([[0.5]], dtype=torch.float64), torch.tensor([[0.5]], dtype=torch.float64))

    assert out.item() == pytest.approx(float(np.exp(np.log(permeability).mean())), rel=1e-12)


def test_field_artifacts_round_trip_and_report_identity(tmp_path):
    npz, metadata, x, y, permeability = _write_valid_field(tmp_path)

    resolved = validate_field_artifacts(npz, metadata)
    loaded_x, loaded_y, loaded_k = load_field_grid(npz)

    assert resolved.field_id == "synthetic-fixed-field"
    assert resolved.sha256 == _sha256(npz)
    assert resolved.shape == (2, 2)
    assert resolved.source_dataset_sha256 == SOURCE_DATASET_SHA256
    np.testing.assert_array_equal(loaded_x, x)
    np.testing.assert_array_equal(loaded_y, y)
    np.testing.assert_array_equal(loaded_k, permeability)


def test_field_artifacts_reject_hash_or_unit_mismatch(tmp_path):
    npz, metadata, *_ = _write_valid_field(tmp_path)
    record = json.loads(metadata.read_text(encoding="utf-8"))
    record["npz_sha256"] = "0" * 64
    metadata.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ValueError, match="SHA-256"):
        validate_field_artifacts(npz, metadata)

    record["npz_sha256"] = _sha256(npz)
    record["units"] = "mD"
    metadata.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match=r"m\^2"):
        validate_field_artifacts(npz, metadata)


def test_field_grid_rejects_nonpositive_or_wrong_shape_values(tmp_path):
    npz = tmp_path / "bad.npz"
    np.savez(
        npz,
        **_npz_payload(
            np.array([0.0, 5.0]),
            np.array([0.0, 5.0]),
            np.array([[1e-14, 0.0]]),
        ),
    )

    with pytest.raises(ValueError, match="shape"):
        load_field_grid(npz)


def test_field_artifacts_must_cover_the_frozen_zero_to_five_metre_domain(tmp_path):
    npz, metadata, _, y, permeability = _write_valid_field(tmp_path)
    np.savez(npz, **_npz_payload(np.array([1.0, 6.0]), y, permeability))
    record = json.loads(metadata.read_text(encoding="utf-8"))
    record["npz_sha256"] = _sha256(npz)
    record["x_range_m"] = [1.0, 6.0]
    metadata.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ValueError, match="frozen Configuration K domain"):
        validate_field_artifacts(npz, metadata)


def test_undefined_directional_correlation_lengths_require_an_explicit_note(tmp_path):
    npz, metadata, *_ = _write_valid_field(tmp_path)
    record = json.loads(metadata.read_text(encoding="utf-8"))
    record["directional_correlation_lengths_note"] = ""
    metadata.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ValueError, match="directional_correlation_lengths_note"):
        validate_field_artifacts(npz, metadata)


def test_unique_npz_requires_embedded_field_identity_and_generation_contract(tmp_path):
    npz = tmp_path / "arrays_only.npz"
    np.savez(
        npz,
        x_grid=np.array([0.0, 5.0]),
        y_grid=np.array([0.0, 5.0]),
        permeability=np.full((2, 2), 1.0e-14),
    )

    with pytest.raises(ValueError, match="missing keys"):
        load_field_grid(npz)


def test_field_metadata_requires_seed_and_generation_parameters(tmp_path):
    npz, metadata, *_ = _write_valid_field(tmp_path)
    record = json.loads(metadata.read_text(encoding="utf-8"))
    record["field_seed"] = None
    metadata.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ValueError, match="field_seed"):
        validate_field_artifacts(npz, metadata)


def test_field_statistics_use_area_weights_not_repeated_or_unweighted_rows():
    x = np.array([0.0, 0.5, 1.0])
    y = np.array([0.0, 0.5, 1.0])
    permeability = np.ones((3, 3), dtype=np.float64)
    permeability[1, 1] = 9.0

    statistics = area_weighted_field_statistics(x, y, permeability)

    assert statistics["arithmetic_mean_m2"] == pytest.approx(3.0)
    assert statistics["arithmetic_mean_m2"] != pytest.approx(float(permeability.mean()))
    assert statistics["geometric_mean_m2"] == pytest.approx(9.0 ** 0.25)
