#!/usr/bin/env python3
"""Inventory, hash, and classify existing IC-FERST reference evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path


GROUPS = {
    "capillary_pressure": re.compile(r"capillary|capillarypressure|\bpc\b", re.I),
    "mesh_adaptivity": re.compile(r"mesh|adapt|refine|coarsen|metric", re.I),
    "time_control": re.compile(r"time.?step|timestep|dt\b|adaptive.?time|max.?time", re.I),
    "solver_tolerance": re.compile(r"tolerance|nonlinear|newton|ksp|snes|relative.?error", re.I),
    "mass_balance": re.compile(r"mass.?balance|conservation|mass.?error|injected.?mass|produced.?mass", re.I),
    "permeability": re.compile(r"permeab|porosity|relative.?permeab|brooks|corey", re.I),
}
TEXT_SUFFIXES = {".mpml", ".flml", ".xml", ".yaml", ".yml", ".cfg", ".ini", ".json", ".log", ".out", ".err", ".txt", ".csv", ".geo"}
CANDIDATE_SUFFIXES = TEXT_SUFFIXES | {".msh", ".stat"}
PRUNE = {".git", "conda", "anaconda3", "evaluation_supplement_20260905T173006Z", "failed_attempts", "__pycache__", ".cache"}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def scan(root: Path):
    for current, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in PRUNE and not d.startswith("evaluation_supplement_")]
        for name in files:
            path = Path(current) / name
            if path.suffix.lower() in CANDIDATE_SUFFIXES or any(x in name.lower() for x in ("mass", "solver", "adapt", "icferst")):
                yield path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--search-root", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for root in args.search_root:
        root = root.resolve()
        if not root.exists():
            records.append({"path": str(root), "status": "search_root_missing", "size": 0, "sha256": "", "groups": []})
            continue
        for path in scan(root):
            try:
                stat = path.stat()
                groups = set()
                snippets = {}
                if path.suffix.lower() in TEXT_SUFFIXES and stat.st_size <= 64 * 1024 * 1024:
                    text = path.read_text(encoding="utf-8", errors="replace")
                    for group, pattern in GROUPS.items():
                        matches = []
                        for line_no, line in enumerate(text.splitlines(), 1):
                            if pattern.search(line):
                                matches.append(f"L{line_no}: {line.strip()[:300]}")
                                if len(matches) == 3: break
                        if matches:
                            groups.add(group); snippets[group] = matches
                records.append({
                    "path": str(path.resolve()), "status": "candidate", "size": stat.st_size,
                    "mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
                    "sha256": digest(path) if stat.st_size <= 2 * 1024**3 else "not_hashed_size_over_2GiB",
                    "groups": sorted(groups), "snippets": snippets,
                })
            except (OSError, PermissionError) as exc:
                records.append({"path": str(path), "status": f"unreadable:{exc}", "size": 0, "sha256": "", "groups": []})

    records.sort(key=lambda r: r["path"])
    (args.output / "reference_evidence_inventory.json").write_text(
        json.dumps({"generated_utc": datetime.now(timezone.utc).isoformat(), "records": records}, indent=2) + "\n",
        encoding="utf-8")
    with (args.output / "reference_evidence_inventory.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["path", "status", "size", "mtime_utc", "sha256", "groups"])
        writer.writeheader()
        for r in records:
            writer.writerow({k: (";".join(r[k]) if k == "groups" else r.get(k, "")) for k in writer.fieldnames})
    counts = {group: sum(group in r.get("groups", []) for r in records) for group in GROUPS}
    missing = [group for group, count in counts.items() if count == 0]
    lines = ["# Reference simulation evidence inventory", "", f"Candidate files: {sum(r['status']=='candidate' for r in records)}", ""]
    lines += [f"- {group}: {counts[group]} candidate files" for group in GROUPS]
    lines += ["", "## Unresolved evidence classes", ""] + ([f"- {x}" for x in missing] or ["- None detected; candidates still require scientific review."])
    lines += ["", "This inventory proves file presence and hashes only. It does not by itself prove H/K alignment, Pc=0, convergence, or mass conservation.", ""]
    (args.output / "REFERENCE_SIMULATION_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")
    summary = {"status": "completed", "candidate_files": sum(r["status"] == "candidate" for r in records), "group_counts": counts, "unresolved_groups": missing}
    (args.output / "inventory_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
