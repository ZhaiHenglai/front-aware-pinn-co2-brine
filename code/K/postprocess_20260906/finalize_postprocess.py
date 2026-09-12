#!/usr/bin/env python3
"""Combine post-processing statuses without overstating missing simulator evidence."""

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def load(path):
    with path.open(encoding="utf-8") as stream: return json.load(stream)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""): h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", required=True, type=Path)
    args = ap.parse_args()
    root = args.output_root.resolve()
    supplement = load(root / "completed_supplement_verification.json")
    common = load(root / "hk_common_time" / "HK_COMMON_TIME_REPORT.json")
    inventory = load(root / "reference_evidence" / "inventory_summary.json")
    if supplement.get("status") != "completed": raise SystemExit("supplement verification failed")
    if common.get("status") != "completed": raise SystemExit("H/K common-time extraction failed")
    evaluation_root = Path(supplement["evaluation_root"])
    manifests = {}
    for path in (evaluation_root / "by_run").rglob("evaluation_manifest.json"):
        record = load(path)
        manifests[record["task"]["run_id"]] = record
    with (evaluation_root / "heterogeneous_metrics_by_seed.csv").open(newline="", encoding="utf-8") as stream:
        metric_rows = list(csv.DictReader(stream))
    imported = []
    for row in metric_rows:
        manifest = manifests[row["run_id"]]
        imported.append({
            **row,
            "model_id": row["run_id"],
            "dataset_id": "sha256:" + manifest["dataset_sha256"],
            "permeability_field_id": manifest["permeability_field_id"],
            "permeability_field_sha256": manifest["permeability_field_sha256"],
            "checkpoint_id": "sha256:" + manifest["checkpoint_sha256"],
            "training_protocol_id": "sha256:" + manifest["resolved_config_sha256"],
            "evaluation_protocol_id": "sha256:" + manifest["code_manifest"]["sha256"],
        })
    with (root / "paper_import_k_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(imported[0])); writer.writeheader(); writer.writerows(imported)
    paper_contract = {
        "status": "completed", "rows": len(imported), "model_id_mapping": "model_id := run_id",
        "rar_global_select_count": 0,
        "basis": "All 20 resolved K configurations and the historical H protocol use zero global RAR selections. A value of 7500 is incompatible with the completed checkpoints.",
        "external_paper_validator_status": "not_checked; K_EXECUTION_FREEZE.md and DATA_SCHEMA.md named in the audit were not present in this project package",
    }
    (root / "PAPER_IMPORT_CONTRACT.json").write_text(json.dumps(paper_contract, indent=2) + "\n", encoding="utf-8")
    overall = {
        "status": "completed_with_reference_evidence_review_required" if inventory.get("unresolved_groups") else "completed_candidates_require_scientific_review",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "pinn_training_rerun": False, "k_supplement_rerun": False,
        "completed_supplement": supplement,
        "hk_common_time": common,
        "reference_evidence_inventory": inventory,
        "paper_import_contract": paper_contract,
        "important_limitation": "Candidate presence is not proof of reference convergence or reference-only mass conservation. Original IC-FERST inputs/logs must be reviewed; run finite verification simulations only if existing evidence is absent and the manuscript retains a convergence claim.",
    }
    (root / "POSTPROCESS_FINAL_STATUS.json").write_text(json.dumps(overall, indent=2) + "\n", encoding="utf-8")
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.name not in {"POSTPROCESS_SHA256SUMS"})
    (root / "POSTPROCESS_SHA256SUMS").write_text("".join(f"{sha(p)}  {p.relative_to(root)}\n" for p in files), encoding="utf-8")
    print(json.dumps({"status": overall["status"], "output_root": str(root)}, indent=2))


if __name__ == "__main__": main()
