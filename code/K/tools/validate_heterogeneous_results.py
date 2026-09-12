#!/usr/bin/env python3
"""Validate that a Configuration K raw metric CSV is exactly the formal 20-row table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from k_results_validation import preregistered_direction_counts, validate_metric_records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    args = parser.parse_args()
    with args.csv.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    validated = validate_metric_records(rows)
    print(json.dumps({
        "status": "pass",
        "configuration": "K",
        "row_count": len(validated),
        "direction_diagnostics": preregistered_direction_counts(validated),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
