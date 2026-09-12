"""Exact raw-result schema and non-selective direction diagnostics for K."""

from __future__ import annotations

import csv
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

from configuration_k import CANONICAL_METRICS, FORMAL_RESULT_CONTRACT, all_formal_tasks


IDENTITY_COLUMNS = (
    "configuration",
    "model_role",
    "exp_name",
    "training_strategy",
    "leave_out",
    "seed",
    "task_index",
    "run_id",
)
CONTRACT_COLUMNS = tuple(FORMAL_RESULT_CONTRACT)
RAW_METRIC_COLUMNS = IDENTITY_COLUMNS + CONTRACT_COLUMNS + CANONICAL_METRICS


def _as_int(value: Any, key: str) -> int:
    try:
        integer = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Result column {key!r} must be an integer; got {value!r}.") from exc
    if str(value).strip() not in {str(integer), f"{integer}.0"}:
        raise ValueError(f"Result column {key!r} must be an exact integer; got {value!r}.")
    return integer


def _as_metric(value: Any, key: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Result metric {key!r} must be numeric; got {value!r}.") from exc
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"Result metric {key!r} must be finite and nonnegative; got {number!r}.")
    return number


def _contract_value(value: Any, key: str, expected: Any) -> Any:
    if isinstance(expected, bool):
        if isinstance(value, bool):
            actual = value
        else:
            normalized = str(value).strip().lower()
            if normalized not in {"true", "false"}:
                raise ValueError(f"Result contract column {key!r} must be boolean; got {value!r}.")
            actual = normalized == "true"
    elif isinstance(expected, int):
        actual = _as_int(value, key)
    elif isinstance(expected, float):
        try:
            actual = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Result contract column {key!r} must be numeric; got {value!r}.") from exc
    else:
        actual = str(value).strip()
    if actual != expected:
        raise ValueError(f"Result contract column {key!r} must be {expected!r}; got {actual!r}.")
    return expected


def validate_metric_records(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return the exact registered 20 rows in frozen task-index order."""

    expected_tasks = {task.array_index: task for task in all_formal_tasks()}
    by_index: dict[int, dict[str, Any]] = {}
    for position, source in enumerate(records):
        if set(source) != set(RAW_METRIC_COLUMNS):
            missing = sorted(set(RAW_METRIC_COLUMNS) - set(source))
            extra = sorted(set(source) - set(RAW_METRIC_COLUMNS))
            raise ValueError(
                f"Result row {position} must use the exact raw metric schema; "
                f"missing={missing}, extra={extra}."
            )
        index = _as_int(source["task_index"], "task_index")
        seed = _as_int(source["seed"], "seed")
        if index not in expected_tasks:
            raise ValueError(f"Result task_index must be registered in 0..19; got {index}.")
        if index in by_index:
            raise ValueError(f"Duplicate formal Configuration K task_index: {index}.")
        task = expected_tasks[index]
        expected_identity = {
            "configuration": "K",
            "model_role": task.role,
            "exp_name": task.exp_name,
            "training_strategy": task.training_strategy,
            "leave_out": task.leave_out,
            "seed": seed,
            "task_index": index,
            "run_id": task.run_id,
        }
        if seed != task.seed:
            raise ValueError(f"Result task_index {index} seed must be {task.seed}; got {seed}.")
        normalized: dict[str, Any] = {}
        for key in IDENTITY_COLUMNS:
            expected = expected_identity[key]
            actual = seed if key == "seed" else index if key == "task_index" else str(source[key]).strip()
            if actual != expected:
                raise ValueError(
                    f"Result task_index {index} {key} must be {expected!r}; got {actual!r}."
                )
            normalized[key] = expected
        for key, expected in FORMAL_RESULT_CONTRACT.items():
            normalized[key] = _contract_value(source[key], key, expected)
        for key in CANONICAL_METRICS:
            normalized[key] = _as_metric(source[key], key)
        by_index[index] = {key: normalized[key] for key in RAW_METRIC_COLUMNS}

    if set(by_index) != set(expected_tasks):
        missing = sorted(set(expected_tasks) - set(by_index))
        raise ValueError(
            f"Formal heterogeneous result table must contain exactly 20 registered rows; "
            f"found {len(by_index)}, missing task indices={missing}."
        )
    return [by_index[index] for index in range(len(expected_tasks))]


def write_metrics_csv(path: str | Path, records: Iterable[Mapping[str, Any]]) -> None:
    rows = validate_metric_records(records)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RAW_METRIC_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, destination)


def preregistered_direction_counts(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = validate_metric_records(records)
    by_role_seed = {(row["model_role"], int(row["seed"])): row for row in rows}
    definitions = (
        ("representation_global_saturation", "no_representation", "saturation_rel_l2", ">"),
        ("front_supervision_front_band", "no_front_supervision", "front_band_rmse", ">"),
        ("front_supervision_contour", "no_front_supervision", "contour_chamfer_s0175", ">"),
        ("local_fv_conservation", "no_local_fv", "local_fv_co2_rmse", ">"),
        ("local_fv_front_tradeoff", "no_local_fv", "front_band_rmse", "<"),
    )
    comparisons = []
    for label, role, metric, operator in definitions:
        per_seed = {}
        for seed in (0, 1, 2, 42):
            ablated = by_role_seed[(role, seed)][metric]
            complete = by_role_seed[("complete", seed)][metric]
            per_seed[str(seed)] = bool(ablated > complete if operator == ">" else ablated < complete)
        comparisons.append({
            "comparison": label,
            "ablation_role": role,
            "metric": metric,
            "expected_operator": f"ablation {operator} complete",
            "matching_seed_count": int(sum(per_seed.values())),
            "total_seed_count": 4,
            "per_seed": per_seed,
        })
    return {
        "is_success_criterion": False,
        "policy": "All 20 registered results remain valid regardless of directional agreement.",
        "comparisons": comparisons,
    }
