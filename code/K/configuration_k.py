"""Frozen scientific contract for Configuration K.

This module is intentionally import-safe: it does not import the trainer, load
the large reference cache, or inspect CUDA.  Training, evaluation, Slurm
launchers, and result validation all consume this single contract.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping

import numpy as np


CONFIGURATION = "K"
FORMAL_SEEDS = (0, 1, 2, 42)
ROLE_SPECS = (
    ("baseline", "M0", "BASE", "NONE"),
    ("complete", "M7", "BASE", "NONE"),
    ("no_representation", "M7", "BASE", "PLAIN_TWONET"),
    ("no_front_supervision", "M7", "BASE", "FRONT_PLUME"),
    ("no_local_fv", "M7", "BASE", "FV"),
)

T_REF_S = 31544.99609375
ANALYSIS_T_MAX_S = 31544.99609375
DOMAIN_X_M = (0.0, 5.0)
DOMAIN_Y_M = (0.0, 5.0)
DATASET_BASENAME = "tables_cache_0_775_step1.pt"
DATASET_RELATIVE_PATH = f"data/{DATASET_BASENAME}"
DATASET_SHA256 = "2132a9be710e2f257ea397cd080654e689093d9c561be3edb5be1e471283e761"

# Exact audit of the SHA-pinned K dataset after applying the shared H time
# window.  These values are part of the formal evidence contract: physics
# evaluators use the retained time bounds, so they must never trust an
# unchecked checkpoint-provided crop report.
FROZEN_DATA_TIME_CROP = {
    "rows_before": 173_688_979,
    "rows_after": 173_688_979,
    "rows_removed": 0,
    "cutoff_s": ANALYSIS_T_MAX_S,
    "first_retained_time_s": 0.0,
    "last_retained_time_s": 30_986.943359375,
}
RETAINED_UNIQUE_TIME_COUNT = 776

NETWORK_WIDTH = 160
FLOAT_DTYPE = "float64"
OPTIMIZER = "Adam"
LEARNING_RATE = 2.0e-4
FORMAL_MAX_ADAM_ITERS = 20000

RAR_START = 3500
RAR_EVERY = 100
RAR_CANDIDATE_COUNT = 30000
RAR_RESIDUAL_SELECT_COUNT = 22500
RAR_GLOBAL_SELECT_COUNT = 0

FV_DT_TILDE = 0.0015850067138671875

# Self-contained columns copied into every raw metric row.  Checkpoint and
# resolved-config validation remains the source of evidence; duplicating these
# frozen values in the CSV lets the standalone table validator detect protocol
# drift, including the historical RAR global-selection conflict.
FORMAL_RESULT_CONTRACT = {
    "run_kind": "formal",
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
}

CANONICAL_METRICS = (
    "saturation_rel_l2",
    "front_band_rmse",
    "contour_chamfer_s0175",
    "local_fv_co2_rmse",
    "co2_mass_rel_error",
)

REFERENCE_ALIGNMENT_KEYS = (
    "geometry",
    "boundary_conditions",
    "initial_conditions",
    "relative_permeability_model",
    "fluid_properties",
    "time_window_covers_common_analysis",
    "numerical_quality_acceptable",
)


@dataclass(frozen=True)
class TaskSpec:
    role: str
    exp_name: str
    training_strategy: str
    leave_out: str
    seed: int

    @property
    def key(self) -> str:
        return f"{self.role}:seed{self.seed}"

    @property
    def run_id(self) -> str:
        return f"{self.role}_seed{self.seed}"

    @property
    def array_index(self) -> int:
        return all_formal_tasks().index(self)

    def as_dict(self) -> dict[str, Any]:
        return {
            "array_index": self.array_index,
            "role": self.role,
            "exp_name": self.exp_name,
            "training_strategy": self.training_strategy,
            "leave_out": self.leave_out,
            "seed": self.seed,
            "task_key": self.key,
            "run_id": self.run_id,
        }


@dataclass(frozen=True)
class CropReport:
    rows_before: int
    rows_after: int
    rows_removed: int
    cutoff_s: float
    first_retained_time_s: float
    last_retained_time_s: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "rows_before": self.rows_before,
            "rows_after": self.rows_after,
            "rows_removed": self.rows_removed,
            "cutoff_s": self.cutoff_s,
            "first_retained_time_s": self.first_retained_time_s,
            "last_retained_time_s": self.last_retained_time_s,
        }


def all_formal_tasks() -> tuple[TaskSpec, ...]:
    return tuple(
        TaskSpec(role, exp_name, strategy, leave_out, seed)
        for role, exp_name, strategy, leave_out in ROLE_SPECS
        for seed in FORMAL_SEEDS
    )


def task_from_index(index: int) -> TaskSpec:
    tasks = all_formal_tasks()
    try:
        resolved = int(index)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Configuration K task index must be in 0..{len(tasks) - 1}; got {index!r}.") from exc
    if resolved < 0 or resolved >= len(tasks):
        raise ValueError(f"Configuration K task index must be in 0..{len(tasks) - 1}; got {index!r}.")
    return tasks[resolved]


def validate_task(
    role: str,
    exp_name: str,
    training_strategy: str,
    leave_out: str,
    seed: int,
) -> TaskSpec:
    candidate = TaskSpec(
        str(role).strip(),
        str(exp_name).strip().upper(),
        str(training_strategy).strip().upper(),
        str(leave_out).strip().upper(),
        int(seed),
    )
    if candidate not in all_formal_tasks():
        raise ValueError(
            "Not a registered Configuration K task: "
            f"role={candidate.role}, exp={candidate.exp_name}, "
            f"strategy={candidate.training_strategy}, leave_out={candidate.leave_out}, "
            f"seed={candidate.seed}."
        )
    return candidate


def parse_bool(value: Any, *, name: str = "value") -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value; got {value!r}.")


def validate_no_pc(env: Mapping[str, Any]) -> None:
    enabled = parse_bool(env.get("PINN_PC_ENABLE", "0"), name="PINN_PC_ENABLE")
    entry = float(env.get("PINN_PC_ENTRY", "0"))
    if enabled:
        raise ValueError("Configuration K requires capillary pressure to be disabled (PINN_PC_ENABLE=0).")
    if entry != 0.0:
        raise ValueError("Configuration K requires PINN_PC_ENTRY=0 when capillary pressure is disabled.")


def _is_torch_tensor(value: Any) -> bool:
    return value.__class__.__module__.split(".", 1)[0] == "torch" and hasattr(value, "detach")


def _time_numpy_1d(value: Any) -> np.ndarray:
    if _is_torch_tensor(value):
        return value.detach().cpu().reshape(-1).to(dtype=value.new_empty((), device="cpu").double().dtype).numpy()
    return np.asarray(value, dtype=np.float64).reshape(-1)


def crop_row_aligned_arrays(
    arrays: Mapping[str, Any], cutoff_s: float = ANALYSIS_T_MAX_S
) -> tuple[dict[str, Any], CropReport]:
    if "t" not in arrays:
        raise ValueError("Dataset arrays must contain physical time key 't'.")
    cutoff = float(cutoff_s)
    if not np.isfinite(cutoff) or cutoff <= 0.0:
        raise ValueError(f"Analysis time cutoff must be positive and finite; got {cutoff_s!r}.")

    t_np = _time_numpy_1d(arrays["t"])
    if t_np.size == 0 or not np.isfinite(t_np).all():
        raise ValueError("Dataset physical time values must all be finite and nonempty.")
    keep_np = t_np <= cutoff
    if not keep_np.any():
        raise ValueError(f"No dataset rows remain at t <= {cutoff:.12g} s.")

    n_rows = int(t_np.size)
    cropped: dict[str, Any] = {}
    for key, value in arrays.items():
        shape = getattr(value, "shape", None)
        if shape is not None and len(shape) >= 1 and int(shape[0]) == n_rows:
            if _is_torch_tensor(value):
                import torch

                keep = torch.from_numpy(keep_np).to(device=value.device)
                cropped[key] = value[keep]
            else:
                cropped[key] = np.asarray(value)[keep_np]
        else:
            cropped[key] = value

    kept_times = t_np[keep_np]
    report = CropReport(
        rows_before=n_rows,
        rows_after=int(keep_np.sum()),
        rows_removed=int((~keep_np).sum()),
        cutoff_s=cutoff,
        first_retained_time_s=float(kept_times.min()),
        last_retained_time_s=float(kept_times.max()),
    )
    return cropped, report


def validate_frozen_crop_report(report: CropReport, unique_time_count: int) -> None:
    """Require the exact crop audited for the SHA-pinned 775-step K cache."""

    actual = report.as_dict()
    for key, expected in FROZEN_DATA_TIME_CROP.items():
        if actual.get(key) != expected:
            raise ValueError(
                f"Frozen Configuration K data_time_crop.{key} must be {expected!r}; "
                f"got {actual.get(key)!r}."
            )
    if int(unique_time_count) != RETAINED_UNIQUE_TIME_COUNT:
        raise ValueError(
            "Frozen Configuration K retained_unique_time_count must be "
            f"{RETAINED_UNIQUE_TIME_COUNT}; got {unique_time_count!r}."
        )


def validate_reference_provenance(
    path: str | Path, *, allow_unverified: bool = False
) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise ValueError(f"Reference provenance file not found: {source}")
    try:
        record = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid reference provenance JSON: {source}") from exc
    if not isinstance(record, dict):
        raise ValueError("Reference provenance must be a JSON object.")
    if record.get("configuration") != CONFIGURATION:
        raise ValueError("Reference provenance must identify configuration='K'.")
    if record.get("reference_dataset_sha256") != DATASET_SHA256:
        raise ValueError("Reference provenance dataset SHA-256 does not match the frozen K dataset.")

    pc_enabled = record.get("capillary_pressure_enabled")
    if pc_enabled is True:
        raise ValueError("Configuration K reference provenance must state capillary pressure disabled.")
    if not allow_unverified and pc_enabled is not False:
        raise ValueError("Formal Configuration K requires explicit capillary pressure disabled evidence.")

    verified = record.get("verified") is True
    if not allow_unverified and not verified:
        raise ValueError("Formal Configuration K requires verified no-Pc provenance.")
    if verified:
        evidence = record.get("evidence")
        if not isinstance(evidence, list) or not evidence:
            raise ValueError("Verified reference provenance must list at least one evidence artifact.")
        evidence_root = source.parent.resolve()
        for index, item in enumerate(evidence):
            if not isinstance(item, dict):
                raise ValueError(
                    f"Verified reference provenance evidence entry {index} must be an object "
                    "with path, sha256, and description."
                )
            relative_text = str(item.get("path", "")).strip()
            if not relative_text:
                raise ValueError(f"Verified reference provenance evidence entry {index} requires a path.")
            relative_path = Path(relative_text)
            if relative_path.is_absolute():
                raise ValueError(
                    f"Verified reference provenance evidence entry {index} path must be relative "
                    "to the provenance file directory."
                )
            evidence_path = (evidence_root / relative_path).resolve()
            try:
                evidence_path.relative_to(evidence_root)
            except ValueError as exc:
                raise ValueError(
                    f"Verified reference provenance evidence entry {index} escapes the provenance directory."
                ) from exc
            if not evidence_path.is_file():
                raise ValueError(
                    f"Verified reference provenance evidence file not found: {evidence_path}"
                )
            expected_sha = str(item.get("sha256", "")).strip().lower()
            if re.fullmatch(r"[0-9a-f]{64}", expected_sha) is None:
                raise ValueError(
                    f"Verified reference provenance evidence entry {index} requires a valid SHA-256."
                )
            digest = hashlib.sha256()
            with evidence_path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != expected_sha:
                raise ValueError(
                    f"Verified reference provenance evidence SHA-256 mismatch: {evidence_path}"
                )
            if not str(item.get("description", "")).strip():
                raise ValueError(
                    f"Verified reference provenance evidence entry {index} requires a description."
                )
        for key in ("simulator", "source_account", "source_case_directory", "verification_notes"):
            if not str(record.get(key, "")).strip():
                raise ValueError(f"Verified reference provenance requires nonempty {key!r}.")
        alignment = record.get("alignment_with_configuration_h")
        if not isinstance(alignment, dict):
            raise ValueError("Verified reference provenance requires alignment_with_configuration_h evidence.")
        for key in REFERENCE_ALIGNMENT_KEYS:
            if alignment.get(key) is not True:
                raise ValueError(
                    "Verified reference provenance requires "
                    f"alignment_with_configuration_h.{key}=true."
                )
    return record
