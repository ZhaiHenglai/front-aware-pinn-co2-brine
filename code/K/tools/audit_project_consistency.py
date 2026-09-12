#!/usr/bin/env python3
"""Static audit for stale paths and unsafe Slurm wrappers in the clean K project."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re


def _relative(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    root = args.root.resolve()
    shell_files = sorted(set(root.rglob("*.slurm")) | set(root.rglob("*.sh")))
    runtime_files = sorted(
        path
        for pattern in ("*.py", "*.slurm", "*.sh")
        for path in root.rglob(pattern)
        if not ({"tests", "docs", ".pytest_cache", "__pycache__"} & set(path.relative_to(root).parts))
        and path.resolve() != Path(__file__).resolve()
    )
    stale_workdirs = []
    masked_srun_failures = []
    fixed_nodes = []
    for path in shell_files:
        text = path.read_text(encoding="utf-8")
        relative = _relative(root, path)
        if re.search(r"\$HOME/PINNnew2/PINN12(?![_A-Za-z0-9])", text):
            stale_workdirs.append(relative)
        if re.search(r"^\s*#SBATCH\s+--nodelist(?:=|\s)", text, flags=re.MULTILINE):
            fixed_nodes.append(relative)
        has_srun = re.search(r"^\s*srun\s", text, flags=re.MULTILINE) is not None
        has_errexit = re.search(r"^\s*set\s+-[^\n#]*e", text, flags=re.MULTILINE) is not None
        propagates_status = "status=$?" in text and re.search(r"exit\s+\$\{?status\}?", text) is not None
        if has_srun and not has_errexit and not propagates_status:
            masked_srun_failures.append(relative)

    stale_dataset_references = []
    forbidden_dataset_tokens = (
        "tables_cache_0_265_step1.pt",
        "0e245470c42585faabc2198950123da2a90439275ce51e0271e35afdee0dfc5f",
        "tables_cache_0_723_step1.pt",
        "b34a14bee121740475f7dc492b0dd2ab532347273c011e1eeeb39f01266e25b7",
    )
    for path in runtime_files:
        text = path.read_text(encoding="utf-8")
        if any(token in text for token in forbidden_dataset_tokens):
            stale_dataset_references.append(_relative(root, path))

    required_assets = (
        root / "data/tables_cache_0_775_step1.pt",
        root / "permeability/K_field_unique_grid.npz",
        root / "permeability/K_field_unique_grid.json",
        root / "manifests/k_reference_provenance.json",
    )
    missing_required_assets = [
        _relative(root, path) for path in required_assets if not path.is_file()
    ]

    canonical = root / "slurm/submit_configuration_k.sh"
    report = {
        "root": str(root),
        "shell_file_count": len(shell_files),
        "stale_workdirs": stale_workdirs,
        "masked_srun_failures": masked_srun_failures,
        "fixed_nodes": fixed_nodes,
        "stale_dataset_references": stale_dataset_references,
        "missing_required_assets": missing_required_assets,
        "canonical_submitter": "slurm/submit_configuration_k.sh" if canonical.is_file() else None,
    }
    print(json.dumps(report, sort_keys=True))
    return 0 if (
        not stale_workdirs
        and not masked_srun_failures
        and not fixed_nodes
        and not stale_dataset_references
        and not missing_required_assets
        and canonical.is_file()
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
