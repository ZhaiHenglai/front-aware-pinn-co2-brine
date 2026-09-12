"""Deterministic, integrity-checked CO2 EOS for frozen Configuration K runs.

The training seed is deliberately not an input to this module.  A fixed
sampling/validation protocol derives one Chebyshev density record from the
cropped reference arrays; the record is sealed with a canonical JSON SHA-256
and is then reused verbatim by training and formal physics evaluation.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn

from configuration_k import ANALYSIS_T_MAX_S, DATASET_SHA256


EOS_SCHEMA = "configuration-k-chebyshev-density-v1"
EOS_FIT_SAMPLE_COUNT = 2_000_000
EOS_FIT_SAMPLE_SEED = 0
EOS_VALIDATION_SAMPLE_COUNT = 20_000
EOS_VALIDATION_SEED = 0
EOS_CANDIDATE_DEGREES = (10, 12, 14, 16, 18, 20)
EOS_PRESSURE_CLAMP_EPS = 1.0e-12
EOS_MAX_ALLOWED_RELATIVE_ERROR = 0.05
EOS_POSITIVITY_CHECK_POINTS = 2049

_TOP_LEVEL_KEYS = {
    "schema",
    "reference_dataset_sha256",
    "analysis_t_max_s",
    "fit_protocol",
    "model",
    "fit_diagnostics",
    "record_sha256",
}
_FIT_PROTOCOL_KEYS = {
    "sampling",
    "sample_count_requested",
    "sample_count_used",
    "valid_sample_count",
    "sample_seed",
    "model_seed_used",
    "pressure_aggregation",
    "candidate_degrees",
    "validation_sample_count",
    "validation_seed",
    "selection_metric",
}
_MODEL_KEYS = {
    "basis",
    "selected_degree",
    "coefficients",
    "rho_ref_kg_m3",
    "pressure_min_pa",
    "pressure_max_pa",
    "pressure_clamp_epsilon",
}
_DIAGNOSTIC_KEYS = {
    "distinct_pressure_count",
    "max_relative_error",
    "rms_relative_error",
}


def _canonical_bytes(record_without_sha: Mapping[str, Any]) -> bytes:
    try:
        encoded = json.dumps(
            record_without_sha,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("EOS record must be finite and canonical-JSON serializable.") from exc
    return encoded.encode("ascii")


def eos_record_sha256(record: Mapping[str, Any]) -> str:
    """Return the canonical digest, excluding the self-referential digest key."""

    payload = dict(record)
    payload.pop("record_sha256", None)
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def seal_eos_record(record_without_sha: Mapping[str, Any]) -> dict[str, Any]:
    """Copy and seal an EOS payload with its stable canonical JSON digest."""

    if "record_sha256" in record_without_sha:
        raise ValueError("Unsealed EOS payload must not already contain record_sha256.")
    # JSON normalization removes NumPy scalar/tuple ambiguity from checkpoint
    # and resolved-config serialization while retaining exact Python floats.
    try:
        normalized = json.loads(_canonical_bytes(record_without_sha).decode("ascii"))
    except json.JSONDecodeError as exc:  # pragma: no cover - canonical encoder guarantees JSON
        raise ValueError("EOS record could not be normalized as canonical JSON.") from exc
    normalized["record_sha256"] = eos_record_sha256(normalized)
    return normalized


def _array_length(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.numel())
    return int(np.asarray(value).size)


def _take_float64(value: Any, indices: np.ndarray) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        flat = value.detach().cpu().reshape(-1)
        index = torch.from_numpy(indices.astype(np.int64, copy=False))
        return flat.index_select(0, index).to(torch.float64).numpy()
    flat = np.asarray(value).reshape(-1)
    return flat[indices].astype(np.float64, copy=False)


def _sample_pair(p_pa: Any, rho_kg_m3: Any) -> tuple[np.ndarray, np.ndarray, int]:
    n_pressure = _array_length(p_pa)
    n_density = _array_length(rho_kg_m3)
    if n_pressure != n_density:
        raise ValueError(
            f"EOS pressure and density arrays must have equal length; got {n_pressure} and {n_density}."
        )
    if n_pressure < 2:
        raise ValueError("EOS fitting requires at least two pressure-density rows.")
    n_used = min(n_pressure, EOS_FIT_SAMPLE_COUNT)
    rng = np.random.default_rng(EOS_FIT_SAMPLE_SEED)
    indices = rng.integers(0, n_pressure, size=n_used, dtype=np.int64)
    return _take_float64(p_pa, indices), _take_float64(rho_kg_m3, indices), n_used


def _compress_by_pressure(p_pa: np.ndarray, rho_kg_m3: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(p_pa, kind="mergesort")
    pressure_sorted = p_pa[order]
    density_sorted = rho_kg_m3[order]
    pressure_unique, starts, counts = np.unique(
        pressure_sorted,
        return_index=True,
        return_counts=True,
    )
    density_mean = np.add.reduceat(density_sorted, starts) / counts
    return pressure_unique.astype(np.float64, copy=False), density_mean.astype(np.float64, copy=False)


def fit_eos_record(
    p_pa: Any,
    rho_kg_m3: Any,
    *,
    reference_dataset_sha256: str = DATASET_SHA256,
) -> dict[str, Any]:
    """Fit and seal the one Configuration K EOS record.

    The fixed protocol RNGs are local and never consult the model seed or the
    process-wide NumPy/Torch random state.
    """

    dataset_sha = str(reference_dataset_sha256).strip().lower()
    if re.fullmatch(r"[0-9a-f]{64}", dataset_sha) is None:
        raise ValueError("reference_dataset_sha256 must be a 64-character hexadecimal SHA-256.")

    pressure, density, sample_count_used = _sample_pair(p_pa, rho_kg_m3)
    valid = np.isfinite(pressure) & np.isfinite(density) & (density > 0.0)
    valid_count = int(valid.sum())
    if valid_count < 2:
        raise ValueError(
            "EOS fit needs at least two sampled finite pressure-density rows with positive density; "
            f"got {valid_count}."
        )
    pressure = pressure[valid]
    density = density[valid]
    pressure_unique, density_unique = _compress_by_pressure(pressure, density)
    distinct_count = int(pressure_unique.size)
    if distinct_count < 2:
        raise ValueError(
            f"EOS fit needs at least two distinct sampled pressure values; got {distinct_count}."
        )

    pressure_min = float(pressure_unique[0])
    pressure_max = float(pressure_unique[-1])
    pressure_mid = 0.5 * (pressure_min + pressure_max)
    pressure_half_range = 0.5 * (pressure_max - pressure_min)
    if not math.isfinite(pressure_half_range) or pressure_half_range <= 0.0:
        raise ValueError(
            f"EOS sampled pressure range must be positive and finite; got [{pressure_min}, {pressure_max}]."
        )
    rho_ref = float(np.interp(pressure_mid, pressure_unique, density_unique))
    if not math.isfinite(rho_ref) or rho_ref <= 0.0:
        raise ValueError(f"EOS reference density must be positive and finite; got {rho_ref!r}.")

    z_fit = (pressure_unique - pressure_mid) / pressure_half_range
    density_tilde = density_unique / rho_ref
    validation_rng = np.random.default_rng(EOS_VALIDATION_SEED)
    validation_pressure = validation_rng.uniform(
        pressure_min,
        pressure_max,
        size=EOS_VALIDATION_SAMPLE_COUNT,
    )
    validation_true = np.interp(validation_pressure, pressure_unique, density_unique)
    z_validation = (validation_pressure - pressure_mid) / pressure_half_range

    effective_degrees = []
    maximum_degree = distinct_count - 1
    for candidate in EOS_CANDIDATE_DEGREES:
        degree = min(int(candidate), maximum_degree)
        if degree >= 1 and degree not in effective_degrees:
            effective_degrees.append(degree)
    if not effective_degrees:
        raise ValueError("EOS fit has no valid Chebyshev degree for the sampled pressure support.")

    best: tuple[float, float, int, np.ndarray] | None = None
    for degree in effective_degrees:
        coefficients = np.polynomial.chebyshev.chebfit(z_fit, density_tilde, deg=degree)
        prediction = rho_ref * np.polynomial.chebyshev.chebval(z_validation, coefficients)
        relative_error = np.abs((prediction - validation_true) / validation_true)
        maximum_relative_error = float(np.max(relative_error))
        rms_relative_error = float(np.sqrt(np.mean(relative_error**2)))
        candidate_result = (
            maximum_relative_error,
            rms_relative_error,
            degree,
            np.asarray(coefficients, dtype=np.float64),
        )
        if best is None or candidate_result[0] < best[0]:
            best = candidate_result
    if best is None:  # pragma: no cover - effective_degrees is nonempty
        raise RuntimeError("EOS fit did not produce a candidate model.")

    maximum_relative_error, rms_relative_error, selected_degree, coefficients = best
    payload = {
        "schema": EOS_SCHEMA,
        "reference_dataset_sha256": dataset_sha,
        "analysis_t_max_s": float(ANALYSIS_T_MAX_S),
        "fit_protocol": {
            "sampling": "numpy-pcg64-with-replacement",
            "sample_count_requested": int(EOS_FIT_SAMPLE_COUNT),
            "sample_count_used": int(sample_count_used),
            "valid_sample_count": int(valid_count),
            "sample_seed": int(EOS_FIT_SAMPLE_SEED),
            "model_seed_used": False,
            "pressure_aggregation": "mean-density-per-exact-pressure",
            "candidate_degrees": [int(v) for v in EOS_CANDIDATE_DEGREES],
            "validation_sample_count": int(EOS_VALIDATION_SAMPLE_COUNT),
            "validation_seed": int(EOS_VALIDATION_SEED),
            "selection_metric": "maximum-relative-error",
        },
        "model": {
            "basis": "chebyshev",
            "selected_degree": int(selected_degree),
            "coefficients": [float(v) for v in coefficients],
            "rho_ref_kg_m3": float(rho_ref),
            "pressure_min_pa": float(pressure_min),
            "pressure_max_pa": float(pressure_max),
            "pressure_clamp_epsilon": float(EOS_PRESSURE_CLAMP_EPS),
        },
        "fit_diagnostics": {
            "distinct_pressure_count": int(distinct_count),
            "max_relative_error": float(maximum_relative_error),
            "rms_relative_error": float(rms_relative_error),
        },
    }
    record = seal_eos_record(payload)
    validate_eos_record(record, expected_dataset_sha256=dataset_sha)
    return record


def _require_keys(value: Any, expected: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"EOS {label} must be an object.")
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"EOS {label} keys must be exactly {sorted(expected)!r}; got {sorted(actual)!r}."
        )
    return value


def _finite_float(value: Any, label: str, *, positive: bool = False) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"EOS {label} must be numeric; got {value!r}.") from exc
    if not math.isfinite(number) or (positive and number <= 0.0):
        qualifier = "positive and finite" if positive else "finite"
        raise ValueError(f"EOS {label} must be {qualifier}; got {number!r}.")
    return number


def validate_eos_record(
    record: Mapping[str, Any],
    *,
    expected_dataset_sha256: str = DATASET_SHA256,
) -> Mapping[str, Any]:
    """Fail closed unless a record matches the frozen protocol and its digest."""

    top = _require_keys(record, _TOP_LEVEL_KEYS, "record")
    supplied_sha = str(top["record_sha256"]).strip().lower()
    if re.fullmatch(r"[0-9a-f]{64}", supplied_sha) is None:
        raise ValueError("EOS record SHA-256 must be 64 lowercase hexadecimal characters.")
    calculated_sha = eos_record_sha256(top)
    if supplied_sha != calculated_sha:
        raise ValueError(
            f"EOS record SHA-256 mismatch: expected {calculated_sha}, got {supplied_sha}."
        )

    if top["schema"] != EOS_SCHEMA:
        raise ValueError(f"EOS schema must be {EOS_SCHEMA!r}; got {top['schema']!r}.")
    dataset_sha = str(top["reference_dataset_sha256"]).strip().lower()
    expected_sha = str(expected_dataset_sha256).strip().lower()
    if dataset_sha != expected_sha:
        raise ValueError(
            f"EOS reference_dataset_sha256 must be {expected_sha}; got {dataset_sha}."
        )
    if _finite_float(top["analysis_t_max_s"], "analysis_t_max_s", positive=True) != ANALYSIS_T_MAX_S:
        raise ValueError(
            f"EOS analysis_t_max_s must be {ANALYSIS_T_MAX_S}; got {top['analysis_t_max_s']!r}."
        )

    protocol = _require_keys(top["fit_protocol"], _FIT_PROTOCOL_KEYS, "fit_protocol")
    frozen_protocol = {
        "sampling": "numpy-pcg64-with-replacement",
        "sample_count_requested": EOS_FIT_SAMPLE_COUNT,
        "sample_seed": EOS_FIT_SAMPLE_SEED,
        "model_seed_used": False,
        "pressure_aggregation": "mean-density-per-exact-pressure",
        "candidate_degrees": list(EOS_CANDIDATE_DEGREES),
        "validation_sample_count": EOS_VALIDATION_SAMPLE_COUNT,
        "validation_seed": EOS_VALIDATION_SEED,
        "selection_metric": "maximum-relative-error",
    }
    for key, expected in frozen_protocol.items():
        if protocol[key] != expected:
            raise ValueError(f"EOS fit_protocol.{key} must be {expected!r}; got {protocol[key]!r}.")
    try:
        sample_count_used = int(protocol["sample_count_used"])
        valid_sample_count = int(protocol["valid_sample_count"])
    except (TypeError, ValueError) as exc:
        raise ValueError("EOS sampled row counts must be integers.") from exc
    if not (2 <= valid_sample_count <= sample_count_used <= EOS_FIT_SAMPLE_COUNT):
        raise ValueError(
            "EOS sampled row counts must satisfy 2 <= valid <= used <= requested; "
            f"got valid={valid_sample_count}, used={sample_count_used}."
        )

    model = _require_keys(top["model"], _MODEL_KEYS, "model")
    if model["basis"] != "chebyshev":
        raise ValueError(f"EOS model basis must be 'chebyshev'; got {model['basis']!r}.")
    coefficients_value = model["coefficients"]
    if not isinstance(coefficients_value, list) or len(coefficients_value) < 2:
        raise ValueError("EOS model coefficients must be a list with at least two values.")
    coefficients = [_finite_float(value, "model.coefficients") for value in coefficients_value]
    try:
        selected_degree = int(model["selected_degree"])
    except (TypeError, ValueError) as exc:
        raise ValueError("EOS model.selected_degree must be an integer.") from exc
    if selected_degree != len(coefficients) - 1:
        raise ValueError(
            "EOS selected degree must match the coefficient vector length; "
            f"got degree={selected_degree}, coefficients={len(coefficients)}."
        )
    if selected_degree < 1 or selected_degree > max(EOS_CANDIDATE_DEGREES):
        raise ValueError(f"EOS selected degree is outside the frozen candidate support: {selected_degree}.")
    rho_ref = _finite_float(model["rho_ref_kg_m3"], "model.rho_ref_kg_m3", positive=True)
    pressure_min = _finite_float(model["pressure_min_pa"], "model.pressure_min_pa")
    pressure_max = _finite_float(model["pressure_max_pa"], "model.pressure_max_pa")
    if pressure_max <= pressure_min:
        raise ValueError(
            f"EOS model pressure range must be increasing; got [{pressure_min}, {pressure_max}]."
        )
    if _finite_float(model["pressure_clamp_epsilon"], "model.pressure_clamp_epsilon", positive=True) != EOS_PRESSURE_CLAMP_EPS:
        raise ValueError(
            f"EOS pressure clamp epsilon must be {EOS_PRESSURE_CLAMP_EPS}; "
            f"got {model['pressure_clamp_epsilon']!r}."
        )

    diagnostics = _require_keys(top["fit_diagnostics"], _DIAGNOSTIC_KEYS, "fit_diagnostics")
    try:
        distinct_count = int(diagnostics["distinct_pressure_count"])
    except (TypeError, ValueError) as exc:
        raise ValueError("EOS distinct_pressure_count must be an integer.") from exc
    if distinct_count < 2 or distinct_count > valid_sample_count:
        raise ValueError(
            f"EOS distinct pressure count must be in [2, {valid_sample_count}]; got {distinct_count}."
        )
    maximum_relative_error = _finite_float(
        diagnostics["max_relative_error"], "fit_diagnostics.max_relative_error"
    )
    rms_relative_error = _finite_float(
        diagnostics["rms_relative_error"], "fit_diagnostics.rms_relative_error"
    )
    for key, value in (
        ("max_relative_error", maximum_relative_error),
        ("rms_relative_error", rms_relative_error),
    ):
        if value < 0.0:
            raise ValueError(f"EOS fit_diagnostics.{key} must be nonnegative; got {value}.")
    if rms_relative_error > maximum_relative_error:
        raise ValueError("EOS RMS fit error cannot exceed its maximum fit error.")
    if maximum_relative_error > EOS_MAX_ALLOWED_RELATIVE_ERROR:
        raise ValueError(
            "EOS maximum fit error exceeds the frozen 5% acceptance threshold: "
            f"{maximum_relative_error}."
        )

    z_check = np.linspace(
        -1.0 + EOS_PRESSURE_CLAMP_EPS,
        1.0 - EOS_PRESSURE_CLAMP_EPS,
        EOS_POSITIVITY_CHECK_POINTS,
        dtype=np.float64,
    )
    density_check = rho_ref * np.polynomial.chebyshev.chebval(z_check, coefficients)
    if not np.isfinite(density_check).all() or not np.all(density_check > 0.0):
        raise ValueError("EOS polynomial must produce positive density throughout its pressure support.")
    # Use the validated values so static analyzers see these checks as consumed.
    if not math.isfinite(rho_ref):  # pragma: no cover
        raise ValueError("EOS rho_ref validation failed.")
    return record


class ChebyshevRhoCO2(nn.Module):
    """Differentiable density model instantiated from one validated record."""

    def __init__(self, record: Mapping[str, Any], *, dtype: torch.dtype):
        super().__init__()
        model = record["model"]
        self.register_buffer("c", torch.tensor(model["coefficients"], dtype=dtype))
        self.rho_ref = float(model["rho_ref_kg_m3"])
        self.pmin = float(model["pressure_min_pa"])
        self.pmax = float(model["pressure_max_pa"])
        self.eps = float(model["pressure_clamp_epsilon"])
        self.pmid = 0.5 * (self.pmin + self.pmax)
        self.prng = 0.5 * (self.pmax - self.pmin)

    def _chebval(self, z: torch.Tensor) -> torch.Tensor:
        if self.c.numel() == 1:  # pragma: no cover - validator requires degree >= 1
            return self.c[0] + 0.0 * z
        b1 = torch.zeros_like(z)
        b2 = torch.zeros_like(z)
        for coefficient in torch.flip(self.c[1:], dims=[0]):
            b0 = 2.0 * z * b1 - b2 + coefficient
            b2 = b1
            b1 = b0
        return z * b1 - b2 + self.c[0]

    def forward(self, p_pa: torch.Tensor) -> torch.Tensor:
        pressure = p_pa.to(dtype=self.c.dtype)
        z = (pressure - self.pmid) / self.prng
        z = torch.clamp(z, -1.0 + self.eps, 1.0 - self.eps)
        return self.rho_ref * self._chebval(z)


def torch_eos_from_record(
    record: Mapping[str, Any],
    *,
    dtype: torch.dtype,
    device: torch.device | str,
    expected_dataset_sha256: str = DATASET_SHA256,
) -> tuple[ChebyshevRhoCO2, float]:
    """Validate and instantiate exactly the EOS encoded in ``record``."""

    validate_eos_record(record, expected_dataset_sha256=expected_dataset_sha256)
    model = ChebyshevRhoCO2(record, dtype=dtype).to(device)
    return model, float(record["model"]["rho_ref_kg_m3"])
