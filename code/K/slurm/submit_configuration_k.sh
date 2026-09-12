#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"
export PINN_PROJECT_ROOT="$PROJECT_ROOT"
SBATCH_BIN="${SBATCH_BIN:-sbatch}"
PY="${PY:-$HOME/conda/envs/py311/bin/python}"
if [ ! -x "$PY" ]; then
  echo "Python executable not found or not executable: $PY" >&2
  exit 3
fi

"$SCRIPT_DIR/verify_code_manifest.sh" "$PROJECT_ROOT"

# Never let shell state from a smoke/debug/holdout session mutate a formal chain.
unset PINN_DRY_RUN PINN_CKPT_TAG PINN_DATA_HOLDOUT PINN_HOLDOUT_MODE PINN_HOLDOUT_SEED
unset PINN_HOLDOUT_FRACTION PINN_HOLDOUT_HOLD_EVERY PINN_HOLDOUT_HOLD_START
unset PINN_HOLDOUT_R_LO PINN_HOLDOUT_R_HI PINN_HOLDOUT_SPLIT_X PINN_HOLDOUT_SPLIT_Y
unset PINN_EVAL_ROOT PINN_MAX_ADAM_ITERS
export PINN_GPU_ID=0

# SBATCH_* variables override options embedded in a batch script. Clear resource,
# array, dependency, and launch controls whose inheritance could change this chain.
unset SBATCH_ARRAY_INX SBATCH_DEPENDENCY SBATCH_EXPORT SBATCH_EXPORT_FILE
unset SBATCH_GRES SBATCH_GPUS SBATCH_GPUS_PER_NODE SBATCH_GPUS_PER_SOCKET SBATCH_GPUS_PER_TASK
unset SBATCH_PARTITION SBATCH_CONSTRAINT SBATCH_EXCLUSIVE SBATCH_OVERSUBSCRIBE
unset SBATCH_NODELIST SBATCH_EXCLUDE
unset SBATCH_TIME SBATCH_TIME_MIN SBATCH_MEM SBATCH_MEM_PER_CPU SBATCH_MEM_PER_GPU
unset SBATCH_NODES SBATCH_NTASKS SBATCH_NTASKS_PER_CORE SBATCH_NTASKS_PER_GPU
unset SBATCH_NTASKS_PER_NODE SBATCH_NTASKS_PER_SOCKET SBATCH_CPUS_PER_TASK
unset SBATCH_BEGIN SBATCH_DEADLINE SBATCH_HOLD SBATCH_REQUEUE SBATCH_NO_REQUEUE
unset SBATCH_TEST_ONLY SBATCH_WAIT SBATCH_WAIT_ALL_NODES SBATCH_CHDIR

DATA_PATH="${PINN_DATA_PT:-$PROJECT_ROOT/data/tables_cache_0_775_step1.pt}"
FIELD_PATH="${PINN_K_FIELD_NPZ:-$PROJECT_ROOT/permeability/K_field_unique_grid.npz}"
FIELD_METADATA_PATH="${PINN_K_FIELD_METADATA:-$PROJECT_ROOT/permeability/K_field_unique_grid.json}"
PROVENANCE_PATH="${PINN_REFERENCE_PROVENANCE:-$PROJECT_ROOT/manifests/k_reference_provenance.json}"
OUTPUT_ROOT="${PINN_OUTPUT_ROOT:-$PROJECT_ROOT}"
export PINN_DATA_PT="$DATA_PATH"
export PINN_K_FIELD_NPZ="$FIELD_PATH"
export PINN_K_FIELD_METADATA="$FIELD_METADATA_PATH"
export PINN_REFERENCE_PROVENANCE="$PROVENANCE_PATH"
export PINN_OUTPUT_ROOT="$OUTPUT_ROOT"
for required in "$DATA_PATH" "$FIELD_PATH" "$FIELD_METADATA_PATH" "$PROVENANCE_PATH"; do
  if [ ! -f "$required" ]; then
    echo "Required formal Configuration K asset not found: $required" >&2
    exit 4
  fi
done

mkdir -p "$OUTPUT_ROOT/manifests"
SUBMISSION_LOCK="$OUTPUT_ROOT/manifests/formal_submission.lock"
if ! mkdir "$SUBMISSION_LOCK" 2>/dev/null; then
  echo "Refusing duplicate Configuration K formal submission; atomic claim already exists: $SUBMISSION_LOCK" >&2
  echo "Inspect $SUBMISSION_LOCK/submission_state.txt and confirm every recorded job is terminal with sacct." >&2
  echo "Before manual lock removal, preserve that state and archive all formal artifacts from the canonical output namespace; never mix a retry with prior formal outputs." >&2
  exit 6
fi

LOCK_COMMITTED=0
PREFLIGHT_TMP=""
cleanup() {
  local status=$?
  if [ -n "$PREFLIGHT_TMP" ]; then
    rm -f -- "$PREFLIGHT_TMP"
  fi
  if [ "$LOCK_COMMITTED" -eq 0 ]; then
    rmdir -- "$SUBMISSION_LOCK" 2>/dev/null || true
  fi
  trap - EXIT
  exit "$status"
}
trap cleanup EXIT

write_submission_state() {
  local status="$1"
  local training_id="$2"
  local evaluation_id="$3"
  local aggregation_id="$4"
  local state_tmp
  state_tmp="$(mktemp "$SUBMISSION_LOCK/.submission_state.XXXXXX")"
  {
    printf 'status=%s\n' "$status"
    printf 'project_root=%s\n' "$PROJECT_ROOT"
    printf 'output_root=%s\n' "$OUTPUT_ROOT"
    printf 'training_job_id=%s\n' "$training_id"
    printf 'evaluation_job_id=%s\n' "$evaluation_id"
    printf 'aggregation_job_id=%s\n' "$aggregation_id"
    printf '%s\n' 'recovery_policy=Confirm every recorded job is terminal with sacct; preserve this state and archive all formal artifacts before manual lock removal; never mix a retry with prior formal outputs.'
  } > "$state_tmp"
  mv "$state_tmp" "$SUBMISSION_LOCK/submission_state.txt"
}

validate_job_id() {
  local label="$1"
  local job_id="$2"
  case "$job_id" in
    ''|*[!0-9]*)
      echo "sbatch returned an invalid $label job ID: $job_id" >&2
      exit 7
      ;;
  esac
}

PREFLIGHT_PATH="$OUTPUT_ROOT/manifests/formal_submission_preflight.json"
PREFLIGHT_TMP="$(mktemp "$OUTPUT_ROOT/manifests/.formal_submission_preflight.XXXXXX")"
"$PY" "$PROJECT_ROOT/tools/validate_k_assets.py" \
  --dataset "$DATA_PATH" \
  --field "$FIELD_PATH" \
  --field-metadata "$FIELD_METADATA_PATH" \
  --reference-provenance "$PROVENANCE_PATH" \
  --run-kind formal > "$PREFLIGHT_TMP"
mv "$PREFLIGHT_TMP" "$PREFLIGHT_PATH"

SMOKE_SUMMARY="$OUTPUT_ROOT/smoke/complete_seed0/training_summary.json"
SMOKE_CONFIG="$OUTPUT_ROOT/smoke/complete_seed0/resolved_config.json"
for required in "$SMOKE_SUMMARY" "$SMOKE_CONFIG"; do
  if [ ! -f "$required" ]; then
    echo "Completed Configuration K smoke evidence not found: $required" >&2
    exit 5
  fi
done
"$PY" -c '
import json, pathlib, sys
summary = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
config = json.loads(pathlib.Path(sys.argv[2]).read_text(encoding="utf-8"))
assets = json.loads(pathlib.Path(sys.argv[3]).read_text(encoding="utf-8"))
status = summary.get("status")
if status != "completed":
    raise SystemExit(f"Smoke status must be completed; got {status!r}")
expected = {
    "configuration": "K", "run_kind": "smoke", "model_role": "complete",
    "exp_name": "M7", "training_strategy": "BASE", "leave_out": "NONE", "seed": 0,
    "pc_enabled": False, "pc_entry_pressure_pa": 0.0,
    "t_ref_s": 31544.99609375, "analysis_t_max_s": 31544.99609375,
    "rar_start_iteration": 3500, "rar_update_interval": 100,
    "rar_candidate_count": 30000, "rar_residual_select_count": 22500,
    "rar_global_select_count": 0,
}
for key, value in expected.items():
    if config.get(key) != value:
        raise SystemExit(f"Smoke resolved config mismatch for {key}: expected {value!r}, got {config.get(key)!r}")
if config.get("reference_dataset_sha256") != assets.get("dataset", {}).get("sha256"):
    raise SystemExit("Smoke and formal preflight reference dataset hashes differ")
if config.get("permeability_field_sha256") != assets.get("field", {}).get("sha256"):
    raise SystemExit("Smoke and formal preflight permeability field hashes differ")
iterations = int(config.get("max_adam_iters", -1))
if not 50 <= iterations <= 200:
    raise SystemExit(f"Smoke iteration budget must be 50..200; got {iterations}")
' "$SMOKE_SUMMARY" "$SMOKE_CONFIG" "$PREFLIGHT_PATH"

"$PY" -c '
from pathlib import Path
import sys
sys.path.insert(0, sys.argv[1])
from configuration_k import all_formal_tasks
root = Path(sys.argv[2])
occupied = []
for task in all_formal_tasks():
    run = root / "runs" / task.role / task.run_id
    evaluation = root / "evaluation" / "by_run" / task.role / task.run_id
    if run.exists() or evaluation.exists():
        occupied.append(str(run if run.exists() else evaluation))
for name in ("heterogeneous_metrics_by_seed.csv", "aggregation_summary.json"):
    path = root / "evaluation" / name
    if path.exists():
        occupied.append(str(path))
if occupied:
    raise SystemExit("Refusing mixed/overwrite submission; existing formal artifacts: " + ", ".join(occupied))
' "$PROJECT_ROOT" "$OUTPUT_ROOT"

# Persist the atomic claim before the first queue mutation. Any partial submission
# remains visible and cannot be accidentally doubled by a blind retry.
LOCK_COMMITTED=1
write_submission_state "submitting_training" "" "" ""

training_job_id="$($SBATCH_BIN --parsable --array=0-19 --chdir="$PROJECT_ROOT" --export=ALL "$SCRIPT_DIR/run_pinn_K_formal_array.slurm")"
training_job_id="${training_job_id%%;*}"
validate_job_id "training" "$training_job_id"
write_submission_state "training_submitted" "$training_job_id" "" ""

evaluation_job_id="$($SBATCH_BIN --parsable --array=0-19 --chdir="$PROJECT_ROOT" --export=ALL --kill-on-invalid-dep=yes --dependency=afterok:"$training_job_id" "$SCRIPT_DIR/run_evaluate_K_formal_array.slurm")"
evaluation_job_id="${evaluation_job_id%%;*}"
validate_job_id "evaluation" "$evaluation_job_id"
write_submission_state "evaluation_submitted" "$training_job_id" "$evaluation_job_id" ""

aggregation_job_id="$($SBATCH_BIN --parsable --chdir="$PROJECT_ROOT" --export=ALL --kill-on-invalid-dep=yes --dependency=afterok:"$evaluation_job_id" "$SCRIPT_DIR/run_aggregate_validate_K.slurm")"
aggregation_job_id="${aggregation_job_id%%;*}"
validate_job_id "aggregation" "$aggregation_job_id"
write_submission_state "submitted" "$training_job_id" "$evaluation_job_id" "$aggregation_job_id"

echo "training_job_id=$training_job_id"
echo "evaluation_job_id=$evaluation_job_id"
echo "aggregation_job_id=$aggregation_job_id"
