import copy
import csv
import json
from pathlib import Path
import subprocess
import sys

import pytest

from configuration_k import CANONICAL_METRICS, FORMAL_RESULT_CONTRACT, all_formal_tasks
from k_results_validation import (
    RAW_METRIC_COLUMNS,
    preregistered_direction_counts,
    validate_metric_records,
    write_metrics_csv,
)


ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = ROOT / "tools/validate_heterogeneous_results.py"


def _records():
    rows = []
    for task in all_formal_tasks():
        row = {
            "configuration": "K",
            "model_role": task.role,
            "exp_name": task.exp_name,
            "training_strategy": task.training_strategy,
            "leave_out": task.leave_out,
            "seed": task.seed,
            "task_index": task.array_index,
            "run_id": task.run_id,
        }
        row.update(FORMAL_RESULT_CONTRACT)
        row.update({metric: (task.array_index + 1) * (j + 1) / 1000 for j, metric in enumerate(CANONICAL_METRICS)})
        rows.append(row)
    return rows


def test_exact_twenty_registered_rows_validate_in_task_order():
    rows = list(reversed(_records()))

    validated = validate_metric_records(rows)

    assert len(validated) == 20
    assert [row["task_index"] for row in validated] == list(range(20))
    assert tuple(validated[0]) == RAW_METRIC_COLUMNS


@pytest.mark.parametrize(
    "mutation",
    ("missing", "duplicate", "smoke", "wrong_role", "nonfinite", "rar_global_7500"),
)
def test_result_validator_rejects_incomplete_duplicate_or_unregistered_rows(mutation):
    rows = _records()
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows[-1] = copy.deepcopy(rows[0])
    elif mutation == "smoke":
        rows.append({**copy.deepcopy(rows[0]), "run_kind": "smoke", "run_id": "smoke_complete_seed0"})
    elif mutation == "wrong_role":
        rows[0]["model_role"] = "complete"
    elif mutation == "nonfinite":
        rows[0]["saturation_rel_l2"] = float("nan")
    else:
        rows[0]["rar_global_select_count"] = 7500

    with pytest.raises(ValueError):
        validate_metric_records(rows)


def test_csv_writer_emits_exact_schema_and_twenty_rows(tmp_path):
    path = tmp_path / "heterogeneous_metrics_by_seed.csv"

    write_metrics_csv(path, _records())

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        assert tuple(reader.fieldnames) == RAW_METRIC_COLUMNS
    assert len(rows) == 20
    assert "rar_global_select_count" in reader.fieldnames


def test_direction_counts_are_diagnostic_and_never_filter_rows():
    rows = _records()
    # Force all preregistered directional comparisons to be false.
    for row in rows:
        if row["model_role"] == "complete":
            for metric in CANONICAL_METRICS:
                row[metric] = 100.0
        elif row["model_role"].startswith("no_"):
            for metric in CANONICAL_METRICS:
                row[metric] = 1.0

    report = preregistered_direction_counts(rows)

    assert report["is_success_criterion"] is False
    by_name = {item["comparison"]: item for item in report["comparisons"]}
    assert len(by_name) == 5
    assert by_name["local_fv_front_tradeoff"]["expected_operator"] == "ablation < complete"
    assert by_name["local_fv_front_tradeoff"]["matching_seed_count"] == 4
    assert all(
        item["matching_seed_count"] == 0
        for name, item in by_name.items()
        if name != "local_fv_front_tradeoff"
    )
    assert len(validate_metric_records(rows)) == 20


def test_standalone_validator_accepts_exact_csv_and_rejects_extra_row(tmp_path):
    path = tmp_path / "heterogeneous_metrics_by_seed.csv"
    write_metrics_csv(path, _records())

    valid = subprocess.run(
        [sys.executable, str(VALIDATOR), "--csv", str(path)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert valid.returncode == 0, valid.stdout + valid.stderr
    assert json.loads(valid.stdout)["row_count"] == 20

    with path.open("a", encoding="utf-8") as handle:
        row = _records()[0]
        handle.write(",".join(str(row[key]) for key in RAW_METRIC_COLUMNS) + "\n")
    invalid = subprocess.run(
        [sys.executable, str(VALIDATOR), "--csv", str(path)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert invalid.returncode != 0
