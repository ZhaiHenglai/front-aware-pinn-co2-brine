#!/usr/bin/env python3
"""Derive the immutable Configuration K grid from the frozen reference cache.

The reference cache stores permeability at every exported row and may use an
adaptive mesh with a different row count at each output time. This tool selects
the earliest physical-time layer, deweights exact duplicate coordinates, and
projects its log-K values onto a 128 x 128 regular grid. Later adaptive layers
therefore never frequency-weight the immutable spatial field.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re
import sys
from typing import Any, Mapping

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configuration_k import (  # noqa: E402
    DATASET_BASENAME,
    DATASET_SHA256,
    DOMAIN_X_M,
    DOMAIN_Y_M,
)
from permeability_field_k import (  # noqa: E402
    area_weighted_field_statistics,
    file_sha256,
    validate_field_artifacts,
)


@dataclass(frozen=True)
class ExtractedField:
    x_grid: np.ndarray
    y_grid: np.ndarray
    permeability: np.ndarray
    source_time_s: float
    source_rows: int
    source_unique_coordinates: int
    time_layer_count: int
    field_stationarity_basis: str
    filled_grid_cells_from_source: int
    source_minimum_m2: float
    source_maximum_m2: float
    source_geometric_mean_m2: float


def _as_numpy_1d(value: Any, *, name: str) -> np.ndarray:
    if value.__class__.__module__.split(".", 1)[0] == "torch" and hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.ndim == 2 and array.shape[1] == 1:
        array = array[:, 0]
    if array.ndim != 1:
        raise ValueError(f"Dataset array {name!r} must be one-dimensional; got {array.shape}.")
    return array


def _shift_no_wrap(array: np.ndarray, shift: int, axis: int) -> np.ndarray:
    shifted = np.full_like(array, np.nan)
    if axis == 0:
        if shift == 1:
            shifted[1:, :] = array[:-1, :]
        else:
            shifted[:-1, :] = array[1:, :]
    else:
        if shift == 1:
            shifted[:, 1:] = array[:, :-1]
        else:
            shifted[:, :-1] = array[:, 1:]
    return shifted


def _project_logk_to_regular_grid(
    x: np.ndarray,
    y: np.ndarray,
    permeability: np.ndarray,
    *,
    grid_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    size = int(grid_size)
    if size < 2:
        raise ValueError(f"grid_size must be at least 2; got {grid_size!r}.")
    xmin, xmax = DOMAIN_X_M
    ymin, ymax = DOMAIN_Y_M
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(permeability) & (permeability > 0.0)
    if not finite.all():
        raise ValueError("The reference spatial slice contains nonfinite coordinates or nonpositive K.")
    tolerance = 1.0e-6
    if x.min() < xmin - tolerance or x.max() > xmax + tolerance:
        raise ValueError(f"Reference x coordinates fall outside the frozen domain {DOMAIN_X_M} m.")
    if y.min() < ymin - tolerance or y.max() > ymax + tolerance:
        raise ValueError(f"Reference y coordinates fall outside the frozen domain {DOMAIN_Y_M} m.")

    ix = np.clip(np.rint((x - xmin) / (xmax - xmin) * (size - 1)).astype(np.int64), 0, size - 1)
    iy = np.clip(np.rint((y - ymin) / (ymax - ymin) * (size - 1)).astype(np.int64), 0, size - 1)
    flat = iy * size + ix
    logk = np.log(permeability.astype(np.float64, copy=False))
    sums = np.bincount(flat, weights=logk, minlength=size * size).astype(np.float64)
    counts = np.bincount(flat, minlength=size * size).astype(np.float64)
    grid = np.full(size * size, np.nan, dtype=np.float64)
    occupied = counts > 0.0
    grid[occupied] = sums[occupied] / counts[occupied]
    grid = grid.reshape(size, size)

    iterations = 0
    while np.isnan(grid).any() and iterations < 4 * size:
        iterations += 1
        missing = np.isnan(grid)
        accumulated = np.zeros_like(grid)
        neighbour_count = np.zeros_like(grid)
        for shift, axis in ((1, 0), (-1, 0), (1, 1), (-1, 1)):
            neighbour = _shift_no_wrap(grid, shift, axis)
            valid = ~np.isnan(neighbour)
            accumulated[valid] += neighbour[valid]
            neighbour_count[valid] += 1.0
        fillable = missing & (neighbour_count > 0.0)
        grid[fillable] = accumulated[fillable] / neighbour_count[fillable]
    if np.isnan(grid).any():
        grid[np.isnan(grid)] = float(np.mean(logk))

    # Preserve the archived interpolation convention: persist a float32 log-grid and
    # exponentiate it for the immutable positive K artifact.
    permeability_grid = np.exp(grid.astype(np.float32).astype(np.float64))
    return (
        np.linspace(xmin, xmax, size, dtype=np.float64),
        np.linspace(ymin, ymax, size, dtype=np.float64),
        permeability_grid,
        int(occupied.sum()),
    )


def build_unique_field_from_earliest_time_arrays(
    arrays: Mapping[str, Any],
    *,
    time_unique: Any,
    grid_size: int = 128,
) -> ExtractedField:
    required = ("x", "y", "t", "permeability")
    missing = [key for key in required if key not in arrays]
    if missing:
        raise ValueError(f"Reference cache is missing arrays: {missing}.")
    x = _as_numpy_1d(arrays["x"], name="x")
    y = _as_numpy_1d(arrays["y"], name="y")
    t = _as_numpy_1d(arrays["t"], name="t")
    permeability = _as_numpy_1d(arrays["permeability"], name="permeability")
    lengths = {int(value.size) for value in (x, y, t, permeability)}
    if len(lengths) != 1:
        raise ValueError("Reference x, y, t, and permeability arrays must have equal row counts.")

    times = np.asarray(time_unique, dtype=np.float64).reshape(-1)
    if times.size < 1 or not np.isfinite(times).all() or not np.all(np.diff(times) > 0.0):
        raise ValueError("Reference time_unique must be finite, nonempty, and strictly increasing.")
    if not np.isfinite(t).all():
        raise ValueError("Reference time values must be finite.")
    if np.any(t[1:] < t[:-1]):
        raise ValueError("Reference rows must be ordered by nondecreasing physical time.")
    transitions = np.flatnonzero(t[1:] != t[:-1]) + 1
    observed_times = np.concatenate((t[:1], t[transitions])).astype(np.float64, copy=False)
    if observed_times.shape != times.shape or not np.allclose(
        observed_times, times, rtol=0.0, atol=1.0e-5
    ):
        raise ValueError("Reference declared time_unique does not match the sorted row times.")
    if not np.isclose(float(t[0]), float(times[0]), rtol=0.0, atol=1.0e-5):
        raise ValueError("Reference rows do not start at the first declared time layer.")
    if not np.isclose(float(t[-1]), float(times[-1]), rtol=0.0, atol=1.0e-5):
        raise ValueError("Reference rows do not end at the last declared time layer.")

    earliest = np.asarray(times[0], dtype=t.dtype)
    rows_per_time = int(np.searchsorted(t, earliest, side="right"))
    if rows_per_time < 1 or not np.all(t[:rows_per_time] == earliest):
        raise ValueError("Reference cache does not contain a contiguous earliest time layer.")
    x_reference = x[:rows_per_time]
    y_reference = y[:rows_per_time]
    k_reference = permeability[:rows_per_time]

    finite_positive = np.isfinite(k_reference) & (k_reference > 0.0)
    if not finite_positive.all():
        raise ValueError("Reference permeability slice contains nonpositive or nonfinite values.")
    coordinate_pairs = np.column_stack((x_reference, y_reference))
    unique_coordinates, first_indices, inverse = np.unique(
        coordinate_pairs,
        axis=0,
        return_index=True,
        return_inverse=True,
    )
    k_unique = k_reference[first_indices]
    if not np.array_equal(k_reference, k_unique[inverse]):
        first_conflict = int(np.flatnonzero(k_reference != k_unique[inverse])[0])
        conflict_x = float(x_reference[first_conflict])
        conflict_y = float(y_reference[first_conflict])
        raise ValueError(
            "Reference duplicate coordinate has conflicting permeability values at "
            f"(x={conflict_x:.17g}, y={conflict_y:.17g})."
        )
    unique_coordinate_count = int(unique_coordinates.shape[0])
    x_grid, y_grid, k_grid, occupied = _project_logk_to_regular_grid(
        unique_coordinates[:, 0].astype(np.float64, copy=False),
        unique_coordinates[:, 1].astype(np.float64, copy=False),
        k_unique.astype(np.float64, copy=False),
        grid_size=grid_size,
    )
    source_logk = np.log(k_unique.astype(np.float64, copy=False))
    return ExtractedField(
        x_grid=x_grid,
        y_grid=y_grid,
        permeability=k_grid,
        source_time_s=float(times[0]),
        source_rows=rows_per_time,
        source_unique_coordinates=unique_coordinate_count,
        time_layer_count=int(times.size),
        field_stationarity_basis=(
            "permeability is a time-independent simulator input; the immutable grid is "
            "derived only from the earliest adaptive-mesh export layer"
        ),
        filled_grid_cells_from_source=occupied,
        source_minimum_m2=float(k_unique.min()),
        source_maximum_m2=float(k_unique.max()),
        source_geometric_mean_m2=float(np.exp(source_logk.mean())),
    )


def write_unique_field_assets(
    result: ExtractedField,
    *,
    field_path: str | Path,
    metadata_path: str | Path,
    dataset_sha256: str,
    dataset_basename: str,
) -> dict[str, Any]:
    dataset_sha = str(dataset_sha256).strip().lower()
    if re.fullmatch(r"[0-9a-f]{64}", dataset_sha) is None:
        raise ValueError("dataset_sha256 must be a valid SHA-256.")
    basename = Path(dataset_basename).name
    if not basename:
        raise ValueError("dataset_basename must be nonempty.")
    size = int(result.x_grid.size)
    match = re.search(r"tables_cache_0_(\d+)_step1\.pt$", basename)
    dataset_label = match.group(1) if match else "reference"
    field_id = f"K{dataset_label}_fixed_permgrid{size}_{dataset_sha[:12]}"
    field_seed = "source_simulator_seed_unrecovered_from_cache"
    generation_method = (
        "deterministic exact-coordinate-deduplicated log-K projection of the "
        "earliest spatial layer from the frozen adaptive-mesh reference cache"
    )
    interpolation_method = "bilinear interpolation of log(K)"
    destination = Path(field_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        destination,
        x_grid=result.x_grid,
        y_grid=result.y_grid,
        permeability=result.permeability,
        field_id=np.array(field_id),
        field_seed=np.array(field_seed),
        units=np.array("m^2"),
        generation_method=np.array(generation_method),
        interpolation_method=np.array(interpolation_method),
        source_dataset_sha256=np.array(dataset_sha),
        source_time_s=np.array(result.source_time_s, dtype=np.float64),
        source_permeability_key=np.array("permeability"),
    )
    npz_sha256 = file_sha256(destination)
    statistics = area_weighted_field_statistics(
        result.x_grid, result.y_grid, result.permeability
    )
    metadata: dict[str, Any] = {
        "configuration": "K",
        "field_id": field_id,
        "field_seed": field_seed,
        "units": "m^2",
        "generation_method": generation_method,
        "generation_parameters": {
            "source_dataset_basename": basename,
            "source_dataset_sha256": dataset_sha,
            "source_time_s": result.source_time_s,
            "source_rows": result.source_rows,
            "source_unique_coordinates": result.source_unique_coordinates,
            "source_duplicate_coordinate_rows_removed": (
                result.source_rows - result.source_unique_coordinates
            ),
            "source_time_layer_count_verified": result.time_layer_count,
            "field_stationarity_basis": result.field_stationarity_basis,
            "grid_size": size,
            "coordinate_deduplication": (
                "exact (x,y) identity; conflicting permeability at a duplicate coordinate fails"
            ),
            "grid_assignment": "nearest regular-grid node using numpy.rint",
            "within_cell_reducer": "arithmetic mean of log(K)",
            "empty_cell_fill": "non-periodic four-neighbour dilation, then source geometric mean",
            "archived_interpolation_compatibility": (
                "same permgrid128 log-K projection and interpolation contract, with "
                "duplicate-coordinate deweighting required for a unique spatial field"
            ),
        },
        "interpolation_method": interpolation_method,
        "source_dataset_sha256": dataset_sha,
        "source_time_s": result.source_time_s,
        "source_permeability_key": "permeability",
        "statistics_weighting": "area-weighted trapezoidal quadrature on the unique regular grid",
        "boundary_and_well_treatment": (
            "The regular grid spans [0,5] x [0,5] m. Empty bins, including well-hole or "
            "under-sampled bins, use non-periodic four-neighbour dilation without wrap-around."
        ),
        "directional_correlation_lengths_m": None,
        "directional_correlation_lengths_note": (
            "Not estimated from the derived grid; the original simulator field seed and "
            "generation parameters remain a provenance item to recover."
        ),
        "npz_sha256": npz_sha256,
        "shape": [int(result.permeability.shape[0]), int(result.permeability.shape[1])],
        "x_range_m": [float(result.x_grid[0]), float(result.x_grid[-1])],
        "y_range_m": [float(result.y_grid[0]), float(result.y_grid[-1])],
        "statistics": statistics,
        "source_slice_statistics": {
            "minimum_m2": result.source_minimum_m2,
            "maximum_m2": result.source_maximum_m2,
            "geometric_mean_m2": result.source_geometric_mean_m2,
        },
        "filled_grid_cells_from_source": result.filled_grid_cells_from_source,
    }
    metadata_destination = Path(metadata_path)
    metadata_destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = metadata_destination.with_name(metadata_destination.name + ".tmp")
    temporary.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(metadata_destination)
    validate_field_artifacts(destination, metadata_destination)
    return metadata


def _load_trusted_reference_cache(path: Path) -> tuple[Mapping[str, Any], Any]:
    try:
        import torch
    except ImportError as exc:
        raise ValueError("PyTorch is required to read the reference cache.") from exc
    pack = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(pack, dict) or not isinstance(pack.get("arrays"), Mapping):
        raise ValueError("Reference cache must be a dictionary containing an 'arrays' mapping.")
    if "time_unique" not in pack:
        raise ValueError("Reference cache is missing time_unique.")
    return pack["arrays"], pack["time_unique"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--field", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--grid-size", type=int, default=128)
    parser.add_argument("--expected-dataset-sha256", default=DATASET_SHA256)
    parser.add_argument(
        "--trust-local-pickle",
        action="store_true",
        help=(
            "Acknowledge that this legacy NumPy .pt needs pickle-compatible loading. "
            "Use only after the local file SHA-256 matches the frozen identity."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    actual_sha = file_sha256(args.dataset)
    expected_sha = str(args.expected_dataset_sha256).strip().lower()
    if actual_sha != expected_sha:
        raise SystemExit(
            f"Dataset SHA-256 mismatch: expected {expected_sha}, got {actual_sha}."
        )
    if not args.trust_local_pickle:
        raise SystemExit(
            "The frozen reference file uses a legacy NumPy pickle payload that weights_only=True cannot read. "
            "Re-run with --trust-local-pickle only for the hash-verified local project asset."
        )
    arrays, time_unique = _load_trusted_reference_cache(args.dataset)
    result = build_unique_field_from_earliest_time_arrays(
        arrays, time_unique=time_unique, grid_size=args.grid_size
    )
    metadata = write_unique_field_assets(
        result,
        field_path=args.field,
        metadata_path=args.metadata,
        dataset_sha256=actual_sha,
        dataset_basename=args.dataset.name or DATASET_BASENAME,
    )
    print(
        json.dumps(
            {
                "status": "pass",
                "dataset_sha256": actual_sha,
                "field_id": metadata["field_id"],
                "field_sha256": metadata["npz_sha256"],
                "source_rows": result.source_rows,
                "source_time_layer_count": result.time_layer_count,
                "source_time_s": result.source_time_s,
                "field_shape": metadata["shape"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
