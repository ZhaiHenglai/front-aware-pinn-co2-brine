#!/bin/bash
# Maxwell interactive submission entrypoint. It submits evaluation only.
# Every failure returns a non-zero status instead of terminating a sourced shell.

pinn12_submit_supplement() {
  set +e
  set +u
  set +o pipefail

  local script_dir project_root py sbatch_bin output_root eval_job summary_job rc

  script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
  if [ -z "$script_dir" ] || [ ! -d "$script_dir" ]; then
    echo "SUPPLEMENT_CONTEXT_ERROR: cannot resolve slurm directory" >&2
    return 2
  fi
  project_root="$(cd -- "$script_dir/.." && pwd -P)"
  py="${PY:-$HOME/conda/envs/py311/bin/python}"

  if [ -r /etc/profile ]; then
    source /etc/profile
  fi
  module purge
  rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "SUPPLEMENT_MODULE_ERROR: module purge failed (rc=$rc)" >&2
    return "$rc"
  fi
  module load slurm
  rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "SUPPLEMENT_MODULE_ERROR: module load slurm failed (rc=$rc)" >&2
    return "$rc"
  fi

  sbatch_bin="$(command -v sbatch)"
  if [ ! -x "$py" ]; then
    echo "SUPPLEMENT_PREFLIGHT_ERROR: missing Python: $py" >&2
    return 2
  fi
  if [ -z "$sbatch_bin" ]; then
    echo "SUPPLEMENT_PREFLIGHT_ERROR: sbatch is unavailable after module load slurm" >&2
    return 2
  fi
  if ! cd "$project_root"; then
    echo "SUPPLEMENT_PREFLIGHT_ERROR: cannot enter $project_root" >&2
    return 2
  fi

  export PINN_PROJECT_ROOT="$project_root"
  export PY="$py"
  unset SBATCH_ARRAY_INX

  # A fresh evaluation must never inherit a preflight or prior retry root.
  if [ -n "${PINN_SUPPLEMENT_ROOT:-}" ]; then
    echo "SUPPLEMENT_PREFLIGHT_ERROR: PINN_SUPPLEMENT_ROOT is already set: $PINN_SUPPLEMENT_ROOT" >&2
    echo "For a fresh run execute: unset PINN_SUPPLEMENT_ROOT PINN_SUPPLEMENT_RETRY" >&2
    return 2
  fi
  export PINN_SUPPLEMENT_RETRY=0

  bash slurm/verify_code_manifest.sh "$project_root"
  rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "SUPPLEMENT_PREFLIGHT_ERROR: code-manifest verification failed (rc=$rc)" >&2
    return "$rc"
  fi
  "$py" -c 'import torch,numpy,scipy,pandas,matplotlib; print("Evaluation dependencies available")'
  rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "SUPPLEMENT_PREFLIGHT_ERROR: Python dependencies unavailable (rc=$rc)" >&2
    return "$rc"
  fi
  "$py" -c '
from pathlib import Path
import json
from configuration_k import all_formal_tasks
for t in all_formal_tasks():
    d=Path("runs")/t.role/t.run_id
    for name in ("final_checkpoint.pt","resolved_config.json","training_summary.json","training_log.out"):
        if not (d/name).is_file(): raise SystemExit(f"Missing {d/name}")
    if json.loads((d/"training_summary.json").read_text())["status"] != "completed":
        raise SystemExit(f"Training incomplete: {t.run_id}")
print("PASS: 20 trained checkpoints and logs")
'
  rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "SUPPLEMENT_PREFLIGHT_ERROR: completed-checkpoint verification failed (rc=$rc)" >&2
    return "$rc"
  fi

  output_root="$project_root/evaluation_supplement_$(date -u +%Y%m%dT%H%M%SZ)"
  if [ -e "$output_root" ]; then
    echo "SUPPLEMENT_PREFLIGHT_ERROR: output root already exists: $output_root" >&2
    return 2
  fi
  mkdir "$output_root"
  rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "SUPPLEMENT_PREFLIGHT_ERROR: cannot create $output_root (rc=$rc)" >&2
    return "$rc"
  fi
  export PINN_SUPPLEMENT_ROOT="$output_root"

  eval_job="$($sbatch_bin --parsable --export=ALL slurm/run_supplement_K.slurm)"
  rc=$?
  eval_job="${eval_job%%;*}"
  if [ "$rc" -ne 0 ] || [[ ! "$eval_job" =~ ^[0-9]+$ ]]; then
    echo "SUPPLEMENT_SUBMIT_ERROR: evaluation array was not submitted (rc=$rc; job=${eval_job:-none})" >&2
    return 2
  fi
  printf 'evaluation_job=%s\nevaluation_root=%s\n' "$eval_job" "$output_root" > "$output_root/submission.txt"

  summary_job="$($sbatch_bin --parsable --export=ALL --dependency="afterok:$eval_job" slurm/run_supplement_aggregate_K.slurm)"
  rc=$?
  summary_job="${summary_job%%;*}"
  if [ "$rc" -ne 0 ] || [[ ! "$summary_job" =~ ^[0-9]+$ ]]; then
    echo "SUPPLEMENT_SUBMIT_ERROR: evaluation job $eval_job is submitted, but aggregation was not (rc=$rc; job=${summary_job:-none})" >&2
    echo "evaluation_job=$eval_job" >&2
    echo "evaluation_root=$output_root" >&2
    return 2
  fi
  printf 'aggregation_job=%s\n' "$summary_job" >> "$output_root/submission.txt"

  printf 'evaluation_job=%s\nevaluation_root=%s\naggregation_job=%s\n' \
    "$eval_job" "$output_root" "$summary_job"
  squeue -j "$eval_job,$summary_job"
  rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "SUPPLEMENT_NOTE: jobs were submitted; squeue display failed (rc=$rc)" >&2
  fi
  return 0
}

pinn12_submit_supplement "$@"
pinn12_submit_supplement_rc=$?

# With `bash script`, only the child shell exits. With `source script`, return.
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  exit "$pinn12_submit_supplement_rc"
fi
return "$pinn12_submit_supplement_rc"
