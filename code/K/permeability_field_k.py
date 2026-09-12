"""Immutable unique-grid permeability support for Configuration K."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from configuration_k import (
    DATASET_SHA256,
    DOMAIN_X_M,
    DOMAIN_Y_M,
    validate_reference_provenance,
)


def file_sha256(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    source = Path(path)
    if not source.is_file():
        raise ValueError(f"Required file not found: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _require_regular_axis(name: str, values: np.ndarray) -> np.ndarray:
    axis = np.asarray(values, dtype=np.float64)
    if axis.ndim != 1 or axis.size < 2:
        raise ValueError(f"{name} must be a one-dimensional array with at least two points.")
    if not np.isfinite(axis).all() or not np.all(np.diff(axis) > 0.0):
        raise ValueError(f"{name} must contain finite strictly increasing coordinates.")
    spacing = np.diff(axis)
    if not np.allclose(spacing, spacing[0], rtol=1e-8, atol=1e-12):
        raise ValueError(f"{name} must be regularly spaced for bilinear grid interpolation.")
    return axis


_REQUIRED_NPZ_KEYS = {
    "x_grid",
    "y_grid",
    "permeability",
    "field_id",
    "field_seed",
    "units",
    "generation_method",
    "interpolation_method",
    "source_dataset_sha256",
    "source_time_s",
    "source_permeability_key",
}


def _scalar_value(pack: Any, key: str) -> Any:
    value = np.asarray(pack[key])
    if value.size != 1:
        raise ValueError(f"Unique permeability NPZ key {key!r} must contain exactly one scalar value.")
    scalar = value.reshape(-1)[0].item() if isinstance(value.reshape(-1)[0], np.generic) else value.reshape(-1)[0]
    if isinstance(scalar, bytes):
        scalar = scalar.decode("utf-8")
    return scalar


def _embedded_text(pack: Any, key: str) -> str:
    value = str(_scalar_value(pack, key)).strip()
    if not value:
        raise ValueError(f"Unique permeability NPZ key {key!r} must be nonempty.")
    return value


def _normalize_field_seed(value: Any, *, source: str) -> int | str:
    if isinstance(value, bool):
        raise ValueError(f"{source} field_seed must be an integer or a nonempty descriptive string.")
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, str) and value.strip():
        return value.strip()
    raise ValueError(
        f"{source} field_seed must be an integer or a nonempty descriptive string; "
        "use an explicit 'not_applicable_...' string for a deterministic non-random field."
    )


def _load_field_payload(
    npz_path: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    source = Path(npz_path)
    if not source.is_file():
        raise ValueError(f"Unique permeability field not found: {source}")
    try:
        with np.load(source, allow_pickle=False) as pack:
            missing = _REQUIRED_NPZ_KEYS - set(pack.files)
            if missing:
                raise ValueError(f"Unique permeability NPZ is missing keys: {sorted(missing)}")
            x_grid = _require_regular_axis("x_grid", pack["x_grid"])
            y_grid = _require_regular_axis("y_grid", pack["y_grid"])
            permeability = np.asarray(pack["permeability"], dtype=np.float64)
            embedded = {
                "field_id": _embedded_text(pack, "field_id"),
                "field_seed": _normalize_field_seed(
                    _scalar_value(pack, "field_seed"), source="Unique permeability NPZ"
                ),
                "units": _embedded_text(pack, "units"),
                "generation_method": _embedded_text(pack, "generation_method"),
                "interpolation_method": _embedded_text(pack, "interpolation_method"),
                "source_dataset_sha256": _embedded_text(pack, "source_dataset_sha256").lower(),
                "source_time_s": float(_scalar_value(pack, "source_time_s")),
                "source_permeability_key": _embedded_text(pack, "source_permeability_key"),
            }
    except OSError as exc:
        raise ValueError(f"Unable to read unique permeability field: {source}") from exc

    expected_shape = (y_grid.size, x_grid.size)
    if permeability.shape != expected_shape:
        raise ValueError(
            f"permeability shape must be {expected_shape} (len(y_grid), len(x_grid)); "
            f"got {permeability.shape}."
        )
    if not np.isfinite(permeability).all() or not np.all(permeability > 0.0):
        raise ValueError("permeability must contain only positive finite values in m^2.")
    return x_grid, y_grid, permeability, embedded


def load_field_grid(npz_path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_grid, y_grid, permeability, _ = _load_field_payload(npz_path)
    return x_grid, y_grid, permeability


def area_weighted_field_statistics(
    x_grid: np.ndarray, y_grid: np.ndarray, permeability: np.ndarray
) -> dict[str, float]:
    """Statistics of the unique spatial field using 2-D trapezoidal area weights.

    This treats the regular endpoint grid as samples of the bilinearly
    interpolated field.  It never weights a location by the number of time rows
    or adaptive-mesh occurrences in the reference cache.
    """

    x = _require_regular_axis("x_grid", x_grid)
    y = _require_regular_axis("y_grid", y_grid)
    k = np.asarray(permeability, dtype=np.float64)
    if k.shape != (y.size, x.size):
        raise ValueError(f"permeability shape must be {(y.size, x.size)}; got {k.shape}.")
    if not np.isfinite(k).all() or not np.all(k > 0.0):
        raise ValueError("permeability must contain only positive finite values in m^2.")

    wx = np.ones(x.size, dtype=np.float64)
    wy = np.ones(y.size, dtype=np.float64)
    wx[[0, -1]] = 0.5
    wy[[0, -1]] = 0.5
    weights = np.outer(wy, wx)
    weights /= weights.sum()
    log_k = np.log(k)
    mean_log_k = float(np.sum(weights * log_k))
    return {
        "minimum_m2": float(k.min()),
        "maximum_m2": float(k.max()),
        "arithmetic_mean_m2": float(np.sum(weights * k)),
        "geometric_mean_m2": float(np.exp(mean_log_k)),
        "log_k_variance": float(np.sum(weights * (log_k - mean_log_k) ** 2)),
    }


@dataclass(frozen=True)
class FieldMetadata:
    field_id: str
    field_seed: Any
    units: str
    generation_method: str
    generation_parameters: dict[str, Any]
    interpolation_method: str
    source_dataset_sha256: str
    source_time_s: float
    source_permeability_key: str
    statistics_weighting: str
    boundary_and_well_treatment: str
    sha256: str
    shape: tuple[int, int]
    x_range_m: tuple[float, float]
    y_range_m: tuple[float, float]
    minimum_m2: float
    maximum_m2: float
    arithmetic_mean_m2: float
    geometric_mean_m2: float
    log_k_variance: float
    directional_correlation_lengths_m: dict[str, float] | None
    directional_correlation_lengths_note: str | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _required_text(record: dict[str, Any], key: str) -> str:
    value = str(record.get(key, "")).strip()
    if not value:
        raise ValueError(f"Permeability metadata requires nonempty {key!r}.")
    return value


def _close(label: str, actual: float, recorded: Any) -> None:
    try:
        expected = float(recorded)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Permeability metadata statistic {label!r} must be numeric.") from exc
    if not np.isclose(actual, expected, rtol=1e-8, atol=max(abs(actual), 1e-30) * 1e-10):
        raise ValueError(
            f"Permeability metadata statistic {label!r} mismatch: "
            f"artifact={actual:.17g}, metadata={expected:.17g}."
        )


def validate_field_artifacts(
    npz_path: str | Path, metadata_path: str | Path
) -> FieldMetadata:
    npz_source = Path(npz_path)
    metadata_source = Path(metadata_path)
    x_grid, y_grid, permeability, embedded = _load_field_payload(npz_source)
    if not metadata_source.is_file():
        raise ValueError(f"Permeability metadata file not found: {metadata_source}")
    try:
        record = json.loads(metadata_source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid permeability metadata JSON: {metadata_source}") from exc
    if not isinstance(record, dict) or record.get("configuration") != "K":
        raise ValueError("Permeability metadata must be a JSON object with configuration='K'.")

    units = _required_text(record, "units")
    if units != "m^2":
        raise ValueError(f"Permeability metadata units must be m^2; got {units!r}.")
    field_id = _required_text(record, "field_id")
    field_seed = _normalize_field_seed(record.get("field_seed"), source="Permeability metadata")
    generation_method = _required_text(record, "generation_method")
    generation_parameters = record.get("generation_parameters")
    if not isinstance(generation_parameters, dict) or not generation_parameters:
        raise ValueError("Permeability metadata requires a nonempty generation_parameters object.")
    interpolation_method = _required_text(record, "interpolation_method")
    source_dataset_sha256 = _required_text(record, "source_dataset_sha256").lower()
    if re.fullmatch(r"[0-9a-f]{64}", source_dataset_sha256) is None:
        raise ValueError("Permeability metadata source_dataset_sha256 must be a valid SHA-256.")
    try:
        source_time_s = float(record.get("source_time_s"))
    except (TypeError, ValueError) as exc:
        raise ValueError("Permeability metadata source_time_s must be finite and numeric.") from exc
    if not np.isfinite(source_time_s) or source_time_s < 0.0:
        raise ValueError("Permeability metadata source_time_s must be finite and nonnegative.")
    source_permeability_key = _required_text(record, "source_permeability_key")
    embedded_expected = {
        "field_id": field_id,
        "field_seed": field_seed,
        "units": units,
        "generation_method": generation_method,
        "interpolation_method": interpolation_method,
        "source_dataset_sha256": source_dataset_sha256,
        "source_permeability_key": source_permeability_key,
    }
    for key, expected in embedded_expected.items():
        if embedded.get(key) != expected:
            raise ValueError(
                f"Permeability metadata {key!r} does not match the value embedded in the NPZ: "
                f"metadata={expected!r}, npz={embedded.get(key)!r}."
            )
    if not np.isclose(
        float(embedded.get("source_time_s")), source_time_s, rtol=0.0, atol=1.0e-12
    ):
        raise ValueError(
            "Permeability metadata 'source_time_s' does not match the value embedded in the NPZ."
        )
    sha256 = file_sha256(npz_source)
    if record.get("npz_sha256") != sha256:
        raise ValueError("Unique permeability field SHA-256 does not match its metadata.")

    expected_shape = [int(permeability.shape[0]), int(permeability.shape[1])]
    if record.get("shape") != expected_shape:
        raise ValueError(f"Permeability metadata shape must be {expected_shape}.")
    x_range = [float(x_grid[0]), float(x_grid[-1])]
    y_range = [float(y_grid[0]), float(y_grid[-1])]
    if record.get("x_range_m") != x_range or record.get("y_range_m") != y_range:
        raise ValueError("Permeability metadata coordinate ranges do not match the NPZ grid.")
    if not np.allclose(x_range, DOMAIN_X_M, rtol=0.0, atol=1.0e-12) or not np.allclose(
        y_range, DOMAIN_Y_M, rtol=0.0, atol=1.0e-12
    ):
        raise ValueError(
            "Unique permeability coordinates must cover the frozen Configuration K domain "
            f"x={DOMAIN_X_M} m and y={DOMAIN_Y_M} m; got x={x_range}, y={y_range}."
        )

    statistics = record.get("statistics")
    if not isinstance(statistics, dict):
        raise ValueError("Permeability metadata requires a statistics object.")
    statistics_weighting = _required_text(record, "statistics_weighting")
    weighting_lower = statistics_weighting.lower()
    if "area" not in weighting_lower or "trapezoid" not in weighting_lower:
        raise ValueError(
            "Permeability statistics_weighting must document area-weighted trapezoidal quadrature."
        )
    actual_stats = area_weighted_field_statistics(x_grid, y_grid, permeability)
    for key, actual in actual_stats.items():
        _close(key, actual, statistics.get(key))

    if "log" not in interpolation_method.lower() or "bilinear" not in interpolation_method.lower():
        raise ValueError("Permeability interpolation_method must document bilinear interpolation of log(K).")

    correlation_lengths = record.get("directional_correlation_lengths_m")
    correlation_note: str | None = None
    if correlation_lengths is None:
        correlation_note = _required_text(record, "directional_correlation_lengths_note")
    elif isinstance(correlation_lengths, dict) and set(correlation_lengths) == {"x_m", "y_m"}:
        normalized_lengths = {}
        for key in ("x_m", "y_m"):
            try:
                value = float(correlation_lengths[key])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Directional correlation length {key!r} must be numeric.") from exc
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"Directional correlation length {key!r} must be positive and finite.")
            normalized_lengths[key] = value
        correlation_lengths = normalized_lengths
        note_value = str(record.get("directional_correlation_lengths_note", "")).strip()
        correlation_note = note_value or None
    else:
        raise ValueError(
            "directional_correlation_lengths_m must be null or an object with positive x_m and y_m values."
        )

    return FieldMetadata(
        field_id=field_id,
        field_seed=field_seed,
        units=units,
        generation_method=generation_method,
        generation_parameters=generation_parameters,
        interpolation_method=interpolation_method,
        source_dataset_sha256=source_dataset_sha256,
        source_time_s=source_time_s,
        source_permeability_key=source_permeability_key,
        statistics_weighting=statistics_weighting,
        boundary_and_well_treatment=_required_text(record, "boundary_and_well_treatment"),
        sha256=sha256,
        shape=(expected_shape[0], expected_shape[1]),
        x_range_m=(x_range[0], x_range[1]),
        y_range_m=(y_range[0], y_range[1]),
        directional_correlation_lengths_m=correlation_lengths,
        directional_correlation_lengths_note=correlation_note,
        **actual_stats,
    )


class PermeabilityFieldK(nn.Module):
    """Positive K(x,y) from bilinear interpolation of a regular log-K grid.

    Inputs are nondimensional coordinates in [0, 1]. The physical coordinate
    ranges are retained as buffers for provenance, while normalized positions
    map directly onto the full unique grid.
    """

    def __init__(self, x_grid: np.ndarray, y_grid: np.ndarray, permeability: np.ndarray):
        super().__init__()
        x = _require_regular_axis("x_grid", x_grid)
        y = _require_regular_axis("y_grid", y_grid)
        k = np.asarray(permeability, dtype=np.float64)
        if k.shape != (y.size, x.size):
            raise ValueError(f"permeability shape must be {(y.size, x.size)}; got {k.shape}.")
        if not np.isfinite(k).all() or not np.all(k > 0.0):
            raise ValueError("permeability must contain only positive finite values.")
        self.nx = int(x.size)
        self.ny = int(y.size)
        self.register_buffer("x_extent_m", torch.tensor([x[0], x[-1]], dtype=torch.float64))
        self.register_buffer("y_extent_m", torch.tensor([y[0], y[-1]], dtype=torch.float64))
        self.register_buffer("log_k_flat", torch.from_numpy(np.log(k).reshape(-1)).to(torch.float64))

    @classmethod
    def from_arrays(
        cls, x_grid: np.ndarray, y_grid: np.ndarray, permeability: np.ndarray
    ) -> "PermeabilityFieldK":
        return cls(x_grid, y_grid, permeability)

    @classmethod
    def from_npz(cls, npz_path: str | Path) -> "PermeabilityFieldK":
        return cls(*load_field_grid(npz_path))

    def forward(self, x_t: torch.Tensor, y_t: torch.Tensor) -> torch.Tensor:
        if x_t.shape != y_t.shape:
            raise ValueError(f"x_t and y_t must have identical shapes; got {x_t.shape} and {y_t.shape}.")
        original_shape = x_t.shape
        xq = x_t.reshape(-1).clamp(0.0, 1.0) * (self.nx - 1)
        yq = y_t.reshape(-1).clamp(0.0, 1.0) * (self.ny - 1)
        i0 = xq.detach().floor().clamp(0, self.nx - 2).to(torch.long)
        j0 = yq.detach().floor().clamp(0, self.ny - 2).to(torch.long)
        i1 = i0 + 1
        j1 = j0 + 1
        tx = xq - i0.to(xq.dtype)
        ty = yq - j0.to(yq.dtype)

        values = self.log_k_flat.to(device=x_t.device, dtype=x_t.dtype)
        base0 = j0 * self.nx
        base1 = j1 * self.nx
        v00 = values[base0 + i0]
        v01 = values[base0 + i1]
        v10 = values[base1 + i0]
        v11 = values[base1 + i1]
        log_k = (
            v00 * (1.0 - tx) * (1.0 - ty)
            + v01 * tx * (1.0 - ty)
            + v10 * (1.0 - tx) * ty
            + v11 * tx * ty
        )
        return torch.exp(log_k).reshape(original_shape)


def validate_asset_bundle(
    dataset_path: str | Path,
    field_path: str | Path,
    field_metadata_path: str | Path,
    reference_provenance_path: str | Path,
    *,
    run_kind: str,
    expected_dataset_sha256: str = DATASET_SHA256,
) -> dict[str, Any]:
    kind = str(run_kind).strip().lower()
    if kind not in {"formal", "smoke"}:
        raise ValueError(f"run_kind must be 'formal' or 'smoke'; got {run_kind!r}.")
    field_metadata = validate_field_artifacts(field_path, field_metadata_path)
    provenance = validate_reference_provenance(
        reference_provenance_path, allow_unverified=(kind == "smoke")
    )
    if provenance.get("permeability_field_id") != field_metadata.field_id:
        raise ValueError(
            "Reference provenance permeability_field_id does not match the validated unique field."
        )
    if provenance.get("permeability_field_sha256") != field_metadata.sha256:
        raise ValueError(
            "Reference provenance permeability field SHA-256 does not match the validated unique field."
        )
    # Validate small externally supplied evidence before hashing the very large
    # reference cache, so an incomplete upload fails quickly.
    dataset_source = Path(dataset_path)
    dataset_sha = file_sha256(dataset_source)
    if dataset_sha != expected_dataset_sha256:
        raise ValueError(
            "Reference dataset SHA-256 mismatch: "
            f"expected {expected_dataset_sha256}, got {dataset_sha}."
        )
    if field_metadata.source_dataset_sha256 != dataset_sha:
        raise ValueError(
            "Unique permeability field was not derived from the validated reference dataset: "
            f"field source={field_metadata.source_dataset_sha256}, dataset={dataset_sha}."
        )
    field_metadata_source = Path(field_metadata_path)
    provenance_source = Path(reference_provenance_path)
    return {
        "status": "pass",
        "run_kind": kind,
        "dataset": {"path": str(dataset_source.resolve()), "sha256": dataset_sha},
        "field": {"path": str(Path(field_path).resolve()), **field_metadata.as_dict()},
        "field_metadata_path": str(field_metadata_source.resolve()),
        "field_metadata_sha256": file_sha256(field_metadata_source),
        "reference_provenance_path": str(provenance_source.resolve()),
        "reference_provenance_sha256": file_sha256(provenance_source),
        "reference_provenance": provenance,
    }
