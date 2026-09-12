#!/usr/bin/env python3
"""Re-evaluate existing K checkpoints and export supplemental diagnostics."""
from pathlib import Path
import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from configuration_k import task_from_index
from permeability_field_k import file_sha256

def commands(task, out, device="cuda", dpi=600, grid=500):
    run = ROOT / "runs" / task.role / task.run_id
    common = ["--data", str(ROOT / "data/tables_cache_0_775_step1.pt")]
    core = [sys.executable, str(ROOT / "evaluation/run_one_k_evaluation.py"),
            "--task-index", str(task.array_index), "--checkpoint", str(run / "final_checkpoint.pt"),
            *common, "--field", str(ROOT / "permeability/K_field_unique_grid.npz"),
            "--field-metadata", str(ROOT / "permeability/K_field_unique_grid.json"),
            "--reference-provenance", str(ROOT / "manifests/k_reference_provenance.json"),
            "--out-dir", str(out), "--device", device, "--publication-figures"]
    alias = out / "checkpoint_view" / (
        f"model_{task.exp_name}_{task.training_strategy}_LOO-{task.leave_out}_seed{task.seed}_v100_final.pt")
    snapshots = [sys.executable, str(ROOT / "Results/evaluate_snapshot_k.py"),
                 *common, "--ckpt", str(alias), "--run-label", task.role,
                 "--out-root", str(out / "snapshots"),
                 "--device", device, "--figure-dpi", str(dpi),
                 "--grid-nx", str(grid), "--grid-ny", str(grid),
                 "--export-pdf", "1", "--export-svg", "1", "--export-tiff", "0"]
    stability = [sys.executable, str(ROOT / "Results/evaluate_training_stability_k.py"),
                 "--log-dir", str(run), "--out-root", str(out / "stability"),
                 "--experiments", task.exp_name, "--training-strategy", task.training_strategy,
                 "--include-loo", "1", "--leave-outs", task.leave_out,
                 "--figure-dpi", str(dpi), "--export-pdf", "1", "--export-svg", "1",
                 "--export-tiff", "0", "--loo-labels", "0"]
    return {"core": core, "snapshots": snapshots, "stability": stability}

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task-index", required=True, type=int)
    ap.add_argument("--evaluation-root", required=True, type=Path)
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--figure-dpi", type=int, default=600)
    ap.add_argument("--grid-size", type=int, default=500)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--retry", action="store_true", help="Verify/skip completed tasks; archive and rerun failed tasks.")
    args = ap.parse_args()
    task = task_from_index(args.task_index)
    out = args.evaluation_root.resolve() / "by_run" / task.role / task.run_id
    cmds = commands(task, out, args.device, args.figure_dpi, args.grid_size)
    if args.dry_run:
        print(json.dumps(cmds, indent=2))
        return
    if out.exists():
        if not args.retry:
            raise ValueError(f"Output already exists: {out}. Use --retry for failed tasks.")
        summary = out / "supplement_summary.json"
        record = json.loads(summary.read_text()) if summary.is_file() else {}
        if record.get("status") == "completed":
            if record.get("task") != task.as_dict() or record.get("code_manifest_sha256") != file_sha256(ROOT / "manifests/CODE_SHA256SUMS"):
                raise ValueError("Completed task belongs to another code version or task.")
            for relative, sha in record["outputs"].items():
                if file_sha256(out / relative) != sha:
                    raise ValueError(f"Completed artifact changed: {relative}")
            print(f"Verified and skipped completed task: {task.run_id}")
            return
        archive = args.evaluation_root.resolve() / "failed_attempts"
        archive.mkdir(exist_ok=True)
        saved = archive / (task.run_id + "_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
        out.rename(saved)
        for name in cmds:
            driver_log = out.parent / f"{task.run_id}_{name}_driver.log"
            if driver_log.exists():
                driver_log.rename(saved / f"{name}_driver.log")
        print(f"Previous failed output preserved at {saved}", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    failures = {}
    for name, cmd in cmds.items():
        # Core creates and checks an empty task directory.
        log = out.parent / f"{task.run_id}_{name}_driver.log"
        with log.open("w") as f:
            f.write(json.dumps(cmd) + "\n")
            f.flush()
            result = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
        if result.returncode:
            failures[name] = {"returncode": result.returncode, "log": str(log)}
    out.mkdir(exist_ok=True)
    # Hash every supplemental output so missing/changed files can be checked at aggregation.
    outputs = {str(p.relative_to(out)): file_sha256(p) for p in sorted(out.rglob("*"))
               if p.is_file() and not p.is_symlink()}
    record = {"status": "failed" if failures else "completed", "task": task.as_dict(),
              "elapsed_seconds": time.monotonic() - started, "failures": failures,
              "outputs": outputs, "commands": cmds,
              "code_manifest_sha256": file_sha256(ROOT / "manifests/CODE_SHA256SUMS")}
    (out / "supplement_summary.json").write_text(json.dumps(record, indent=2) + "\n")
    if failures:
        raise SystemExit(json.dumps(failures))
    print(f"Completed {task.run_id}: {len(outputs)} artifacts")

if __name__ == "__main__":
    main()
