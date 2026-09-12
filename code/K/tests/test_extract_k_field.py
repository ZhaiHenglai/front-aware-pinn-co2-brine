"""Tests for adaptive-layer Configuration K permeability extraction."""

import hashlib
import json

import numpy as np
import pytest

from permeability_field_k import validate_field_artifacts
from tools.extract_k_field import (
    build_unique_field_from_earliest_time_arrays,
    write_unique_field_assets,
)


def _fixed_two_time_arrays():
    x = np.array([0.0, 5.0, 0.0, 5.0], dtype=np.float32)
    y = np.array([0.0, 0.0, 5.0, 5.0], dtype=np.float32)
    k = np.array([1.0e-14, 2.0e-14, 4.0e-14, 8.0e-14], dtype=np.float32)
    return {
        "x": np.tile(x, 2),
        "y": np.tile(y, 2),
        "t": np.repeat(np.array([0.0, 10.0], dtype=np.float32), 4),
        "permeability": np.tile(k, 2),
    }


def _fixed_two_time_arrays_with_partition_duplicate(*, conflicting=False):
    # The first two rows represent the same mesh location repeated by a
    # partitioned export.  The nearby third location maps to the same 2 x 2
    # regular-grid cell, so counting the duplicate would bias that cell.
    x = np.array([0.0, 0.0, 0.1, 5.0, 0.0, 5.0], dtype=np.float32)
    y = np.array([0.0, 0.0, 0.0, 0.0, 5.0, 5.0], dtype=np.float32)
    k = np.array([1.0e-14, 2.0e-14 if conflicting else 1.0e-14, 4.0e-14,
                  2.0e-14, 3.0e-14, 4.0e-14], dtype=np.float32)
    return {
        "x": np.tile(x, 2),
        "y": np.tile(y, 2),
        "t": np.repeat(np.array([0.0, 10.0], dtype=np.float32), x.size),
        "permeability": np.tile(k, 2),
    }


def _adaptive_two_time_arrays():
    return {
        "x": np.array([0.0, 5.0, 0.0, 5.0, 0.2, 4.8, 2.5], dtype=np.float32),
        "y": np.array([0.0, 0.0, 5.0, 5.0, 0.2, 4.8, 2.5], dtype=np.float32),
        "t": np.array([0.0, 0.0, 0.0, 0.0, 10.0, 10.0, 10.0], dtype=np.float32),
        "permeability": np.array(
            [1.0e-14, 2.0e-14, 4.0e-14, 8.0e-14, 1.1e-14, 7.9e-14, 3.0e-14],
            dtype=np.float32,
        ),
    }


def test_extraction_accepts_adaptive_time_layers_and_uses_the_earliest_layer():
    result = build_unique_field_from_earliest_time_arrays(
        _adaptive_two_time_arrays(),
        time_unique=np.array([0.0, 10.0]),
        grid_size=2,
    )

    assert result.source_time_s == 0.0
    assert result.source_rows == 4
    assert result.source_unique_coordinates == 4
    assert result.time_layer_count == 2
    np.testing.assert_allclose(
        result.permeability,
        np.array([[1.0e-14, 2.0e-14], [4.0e-14, 8.0e-14]]),
        rtol=2.0e-6,
    )


def test_extraction_rejects_declared_times_that_do_not_match_sorted_rows():
    arrays = _adaptive_two_time_arrays()
    arrays["t"][4:6] = 5.0

    with pytest.raises(ValueError, match="declared time_unique does not match"):
        build_unique_field_from_earliest_time_arrays(
            arrays,
            time_unique=np.array([0.0, 10.0]),
            grid_size=2,
        )

def test_extraction_uses_one_fixed_spatial_slice_and_preserves_corner_values():
    result = build_unique_field_from_earliest_time_arrays(
        _fixed_two_time_arrays(),
        time_unique=np.array([0.0, 10.0]),
        grid_size=2,
    )

    assert result.source_time_s == 0.0
    assert result.source_rows == 4
    assert result.time_layer_count == 2
    assert "time-independent simulator input" in result.field_stationarity_basis
    np.testing.assert_allclose(
        result.permeability,
        np.array([[1.0e-14, 2.0e-14], [4.0e-14, 8.0e-14]]),
        rtol=2.0e-6,
    )


def test_extraction_uses_the_earliest_layer_when_later_adaptive_values_differ():
    arrays = _fixed_two_time_arrays()
    arrays["permeability"][-1] *= 2.0

    result = build_unique_field_from_earliest_time_arrays(
        arrays,
        time_unique=np.array([0.0, 10.0]),
        grid_size=2,
    )

    np.testing.assert_allclose(
        result.permeability,
        np.array([[1.0e-14, 2.0e-14], [4.0e-14, 8.0e-14]]),
        rtol=2.0e-6,
    )


def test_extraction_counts_each_exact_spatial_coordinate_once():
    result = build_unique_field_from_earliest_time_arrays(
        _fixed_two_time_arrays_with_partition_duplicate(),
        time_unique=np.array([0.0, 10.0]),
        grid_size=2,
    )

    assert result.source_rows == 6
    assert result.source_unique_coordinates == 5
    # The lower-left cell contains the two unique locations with K=1e-14 and
    # K=4e-14, hence its log-space mean is 2e-14.  Counting the repeated first
    # row would instead produce 4**(1/3)e-14.
    np.testing.assert_allclose(result.permeability[0, 0], 2.0e-14, rtol=2.0e-6)


def test_extraction_rejects_conflicting_k_at_an_exact_duplicate_coordinate():
    with pytest.raises(ValueError, match="duplicate coordinate.*conflicting permeability"):
        build_unique_field_from_earliest_time_arrays(
            _fixed_two_time_arrays_with_partition_duplicate(conflicting=True),
            time_unique=np.array([0.0, 10.0]),
            grid_size=2,
        )


def test_written_assets_bind_field_to_source_dataset_hash(tmp_path):
    dataset_sha = "2132a9be710e2f257ea397cd080654e689093d9c561be3edb5be1e471283e761"
    result = build_unique_field_from_earliest_time_arrays(
        _fixed_two_time_arrays(),
        time_unique=np.array([0.0, 10.0]),
        grid_size=2,
    )
    field_path = tmp_path / "K_field_unique_grid.npz"
    metadata_path = tmp_path / "K_field_unique_grid.json"

    write_unique_field_assets(
        result,
        field_path=field_path,
        metadata_path=metadata_path,
        dataset_sha256=dataset_sha,
        dataset_basename="tables_cache_0_775_step1.pt",
    )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["source_dataset_sha256"] == dataset_sha
    assert metadata["source_time_s"] == 0.0
    assert metadata["source_permeability_key"] == "permeability"
    assert metadata["field_id"].startswith("K775_fixed_permgrid2_")
    assert metadata["npz_sha256"] == hashlib.sha256(field_path.read_bytes()).hexdigest()
    validated = validate_field_artifacts(field_path, metadata_path)
    assert validated.source_dataset_sha256 == dataset_sha
