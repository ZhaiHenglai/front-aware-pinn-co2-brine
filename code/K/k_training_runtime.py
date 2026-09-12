"""Run-evidence helpers for the Configuration K trainer."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

from configuration_k import (
    ANALYSIS_T_MAX_S,
    DATASET_SHA256,
    FLOAT_DTYPE,
    FORMAL_MAX_ADAM_ITERS,
    LEARNING_RATE,
    NETWORK_WIDTH,
    OPTIMIZER,
    RAR_CANDIDATE_COUNT,
    RAR_EVERY,
    RAR_GLOBAL_SELECT_COUNT,
    RAR_RESIDUAL_SELECT_COUNT,
    RAR_START,
    T_REF_S,
    TaskSpec,
    validate_task,
)


def checkpoint_paths(
    run_dir: str | Path, task: TaskSpec, *, profile: str, suffix: str
) -> tuple[Path, Path]:
    directory = Path(run_dir)
    evaluator_name = (
        f"model_{task.exp_name}_{task.training_strategy}_LOO-{task.leave_out}_"
        f"seed{task.seed}_{profile}_{suffix}.pt"
    )
    alias = directory / evaluator_name
    canonical = directory / "final_checkpoint.pt" if suffix == "final" else alias
    return canonical, alias


def _require_equal(name: str, actual: Any, expected: Any) -> None:
    if isinstance(expected, float):
        try:
            matches = float(actual) == expected
        except (TypeError, ValueError):
            matches = False
    else:
        matches = actual == expected
    if not matches:
        raise ValueError(f"Frozen Configuration K {name} must be {expected!r}; got {actual!r}.")


def _validate_runtime(runtime: Mapping[str, Any], run_kind: str) -> None:
    frozen = {
        "width": NETWORK_WIDTH,
        "dtype": FLOAT_DTYPE,
        "optimizer": OPTIMIZER,
        "learning_rate": LEARNING_RATE,
        "rar_start_iteration": RAR_START,
        "rar_update_interval": RAR_EVERY,
        "rar_candidate_count": RAR_CANDIDATE_COUNT,
        "rar_residual_select_count": RAR_RESIDUAL_SELECT_COUNT,
        "rar_global_select_count": RAR_GLOBAL_SELECT_COUNT,
    }
    for key, expected in frozen.items():
        if key not in runtime:
            raise ValueError(f"Runtime evidence is missing required key {key!r}.")
        _require_equal(key, runtime[key], expected)

    if "max_adam_iters" not in runtime:
        raise ValueError("Runtime evidence is missing required key 'max_adam_iters'.")
    iterations = int(runtime["max_adam_iters"])
    if run_kind == "formal":
        _require_equal("max_adam_iters", iterations, FORMAL_MAX_ADAM_ITERS)
    elif not 50 <= iterations <= 200:
        raise ValueError(f"Smoke max_adam_iters must be between 50 and 200; got {iterations}.")


def build_resolved_config(
    task: TaskSpec,
    assets: Mapping[str, Any],
    runtime: Mapping[str, Any],
    *,
    run_kind: str,
) -> dict[str, Any]:
    kind = str(run_kind).strip().lower()
    if kind not in {"formal", "smoke"}:
        raise ValueError(f"run_kind must be 'formal' or 'smoke'; got {run_kind!r}.")
    validate_task(task.role, task.exp_name, task.training_strategy, task.leave_out, task.seed)
    if assets.get("status") != "pass" or assets.get("run_kind") != kind:
        raise ValueError("Asset evidence must be a passing report for the same run kind.")
    dataset = assets.get("dataset", {})
    field = assets.get("field", {})
    provenance = assets.get("reference_provenance", {})
    if dataset.get("sha256") != DATASET_SHA256:
        raise ValueError("Runtime asset evidence does not contain the frozen reference dataset SHA-256.")
    if not str(field.get("field_id", "")).strip() or len(str(field.get("sha256", ""))) != 64:
        raise ValueError("Runtime asset evidence lacks a valid permeability field identity and SHA-256.")
    for key in ("field_metadata_sha256", "reference_provenance_sha256"):
        digest = str(assets.get(key, "")).strip().lower()
        if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError(f"Runtime asset evidence lacks a valid {key}.")
    if provenance.get("capillary_pressure_enabled") is True:
        raise ValueError("Configuration K reference evidence reports capillary pressure enabled.")
    if kind == "formal" and provenance.get("verified") is not True:
        raise ValueError("Formal Configuration K requires verified no-Pc reference provenance.")
    _validate_runtime(runtime, kind)

    result: dict[str, Any] = {
        "configuration": "K",
        "run_kind": kind,
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
        "max_adam_iters": int(runtime["max_adam_iters"]),
        "rar_start_iteration": RAR_START,
        "rar_update_interval": RAR_EVERY,
        "rar_candidate_count": RAR_CANDIDATE_COUNT,
        "rar_residual_select_count": RAR_RESIDUAL_SELECT_COUNT,
        "rar_global_select_count": RAR_GLOBAL_SELECT_COUNT,
        "permeability_field_id": field["field_id"],
        "permeability_field_sha256": field["sha256"],
        "permeability_field_path": field.get("path"),
        "permeability_field_metadata_path": assets.get("field_metadata_path"),
        "field_metadata_sha256": str(assets["field_metadata_sha256"]).lower(),
        "reference_dataset_sha256": dataset["sha256"],
        "reference_dataset_path": dataset.get("path"),
        "reference_provenance_path": assets.get("reference_provenance_path"),
        "reference_provenance_sha256": str(assets["reference_provenance_sha256"]).lower(),
        "reference_provenance_verified": provenance.get("verified") is True,
    }
    for key in (
        "python_version",
        "torch_version",
        "cuda_version",
        "gpu_model",
        "slurm_job_id",
        "training_script_sha256",
    ):
        result[key] = runtime.get(key)
    return result


def write_json_atomic(path: str | Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, destination)


def finalize_training_summary(
    *,
    status: str,
    elapsed_seconds: float,
    peak_gpu_memory_allocated_bytes: int,
    peak_gpu_memory_reserved_bytes: int,
    checkpoint_path: str | None,
    error: str | None,
) -> dict[str, Any]:
    normalized = str(status).strip().lower()
    if normalized not in {"completed", "failed"}:
        raise ValueError(f"Training summary status must be 'completed' or 'failed'; got {status!r}.")
    return {
        "status": normalized,
        "elapsed_seconds": float(elapsed_seconds),
        "peak_gpu_memory_allocated_bytes": int(peak_gpu_memory_allocated_bytes),
        "peak_gpu_memory_reserved_bytes": int(peak_gpu_memory_reserved_bytes),
        "checkpoint_path": checkpoint_path,
        "error": error,
    }


def ensure_finite_training_loss(loss: Any, *, iteration: int) -> None:
    """Fail before backpropagation when the scalar objective is NaN or Inf."""

    import torch

    if not torch.is_tensor(loss) or loss.numel() != 1:
        raise ValueError("Training loss must be a scalar torch.Tensor.")
    if not bool(torch.isfinite(loss.detach()).all().item()):
        raise FloatingPointError(
            f"Nonfinite training loss detected at iteration {int(iteration)}."
        )


def _ensure_finite_named_tensors(
    named_tensors: Iterable[tuple[str, Any]], *, kind: str, iteration: int
) -> None:
    import torch

    tensors = tuple((str(name), value) for name, value in named_tensors if value is not None)
    if not tensors:
        return
    all_finite = torch.stack(
        [torch.isfinite(value.detach()).all() for _, value in tensors]
    ).all()
    if bool(all_finite.item()):
        return
    for name, value in tensors:
        if not bool(torch.isfinite(value.detach()).all().item()):
            raise FloatingPointError(
                f"Nonfinite {kind} in parameter {name!r} detected at iteration {int(iteration)}."
            )
    raise FloatingPointError(f"Nonfinite {kind} detected at iteration {int(iteration)}.")


def ensure_finite_gradients(named_parameters: Iterable[tuple[str, Any]], *, iteration: int) -> None:
    """Fail when any materialized model gradient is NaN or Inf."""

    _ensure_finite_named_tensors(
        ((name, parameter.grad) for name, parameter in named_parameters),
        kind="gradient",
        iteration=iteration,
    )


def ensure_finite_parameters(named_parameters: Iterable[tuple[str, Any]], *, iteration: int) -> None:
    """Fail when an optimizer step creates a NaN or Inf model parameter."""

    _ensure_finite_named_tensors(
        ((name, parameter) for name, parameter in named_parameters),
        kind="parameter",
        iteration=iteration,
    )
