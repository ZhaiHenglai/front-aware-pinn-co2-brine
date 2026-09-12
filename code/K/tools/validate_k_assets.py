#!/usr/bin/env python3
"""Fail-closed asset preflight for Configuration K."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from permeability_field_k import validate_asset_bundle


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Reference dataset .pt")
    parser.add_argument("--field", required=True, help="Unique K_field_unique_grid.npz")
    parser.add_argument("--field-metadata", required=True, help="Unique-field metadata JSON")
    parser.add_argument("--reference-provenance", required=True, help="Reference simulator provenance JSON")
    parser.add_argument("--run-kind", choices=("formal", "smoke"), default="formal")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = validate_asset_bundle(
            args.dataset,
            args.field,
            args.field_metadata,
            args.reference_provenance,
            run_kind=args.run_kind,
        )
    except (OSError, ValueError) as exc:
        print(json.dumps({"status": "fail", "error": str(exc)}, indent=2, sort_keys=True))
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

