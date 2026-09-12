#!/bin/bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 TASK_INDEX {formal|smoke}" >&2
  exit 2
fi

TASK_INDEX="$1"
RUN_KIND="$2"
case "$RUN_KIND" in
  formal|smoke) ;;
  *) echo "RUN_KIND must be formal or smoke; got: $RUN_KIND" >&2; exit 2 ;;
esac

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"
PY="${PY:-$HOME/conda/envs/py311/bin/python}"
if [ ! -x "$PY" ]; then
  echo "Python executable not found or not executable: $PY" >&2
  exit 3
fi

TASK_TSV="$("$PY" -c '
import sys
sys.path.insert(0, sys.argv[1])
from configuration_k import task_from_index
try:
    task = task_from_index(int(sys.argv[2]))
except (TypeError, ValueError) as exc:
    print(str(exc), file=sys.stderr)
    raise SystemExit(2)
print("\t".join(map(str, (task.array_index, task.role, task.exp_name, task.training_strategy, task.leave_out, task.seed, task.run_id))))
' "$PROJECT_ROOT" "$TASK_INDEX")"
IFS=$'\t' read -r ARRAY_INDEX ROLE EXP_NAME TRAINING_STRATEGY LEAVE_OUT SEED RUN_ID <<< "$TASK_TSV"

if [ "$RUN_KIND" = "smoke" ] && [ "$ARRAY_INDEX" -ne 4 ]; then
  echo "Configuration K smoke is registered only for task index 4 (complete seed 0)." >&2
  exit 4
fi

DATA_PATH="${PINN_DATA_PT:-$PROJECT_ROOT/data/tables_cache_0_775_step1.pt}"
FIELD_PATH="${PINN_K_FIELD_NPZ:-$PROJECT_ROOT/permeability/K_field_unique_grid.npz}"
FIELD_METADATA_PATH="${PINN_K_FIELD_METADATA:-$PROJECT_ROOT/permeability/K_field_unique_grid.json}"
PROVENANCE_PATH="${PINN_REFERENCE_PROVENANCE:-$PROJECT_ROOT/manifests/k_reference_provenance.json}"
for required in "$DATA_PATH" "$FIELD_PATH" "$FIELD_METADATA_PATH" "$PROVENANCE_PATH"; do
  if [ ! -f "$required" ]; then
    echo "Required Configuration K asset not found: $required" >&2
    exit 5
  fi
done

OUTPUT_ROOT="${PINN_OUTPUT_ROOT:-$PROJECT_ROOT}"
if [ "$RUN_KIND" = "formal" ]; then
  RUN_DIR="$OUTPUT_ROOT/runs/$ROLE/$RUN_ID"
  MAX_ADAM_ITERS=20000
else
  RUN_DIR="$OUTPUT_ROOT/smoke/$RUN_ID"
  MAX_ADAM_ITERS="${PINN_MAX_ADAM_ITERS:-100}"
fi

if [ -e "$RUN_DIR" ]; then
  echo "Refusing to overwrite an existing Configuration K run: $RUN_DIR" >&2
  exit 6
fi

if [ "${PINN_DRY_RUN:-0}" = "1" ]; then
  "$PY" -c '
import json, sys
print(json.dumps({
    "array_index": int(sys.argv[1]),
    "role": sys.argv[2],
    "exp_name": sys.argv[3],
    "training_strategy": sys.argv[4],
    "leave_out": sys.argv[5],
    "seed": int(sys.argv[6]),
    "run_kind": sys.argv[7],
    "run_dir": sys.argv[8],
    "asset_content_validation": "deferred_until_execution",
}, sort_keys=True))
' "$ARRAY_INDEX" "$ROLE" "$EXP_NAME" "$TRAINING_STRATEGY" "$LEAVE_OUT" "$SEED" "$RUN_KIND" "$RUN_DIR"
  exit 0
fi

"$SCRIPT_DIR/verify_code_manifest.sh" "$PROJECT_ROOT"
"$SCRIPT_DIR/require_cuda_gpu.sh" "$PY"

mkdir -p "$(dirname -- "$RUN_DIR")"
if ! mkdir "$RUN_DIR" 2>/dev/null; then
  echo "Refusing to overwrite an existing or concurrently-created Configuration K run: $RUN_DIR" >&2
  exit 6
fi

write_failed_summary() {
  local message="$1"
  "$PY" -c '
import sys
sys.path.insert(0, sys.argv[1])
from k_training_runtime import finalize_training_summary, write_json_atomic
write_json_atomic(sys.argv[2], finalize_training_summary(
    status="failed",
    elapsed_seconds=0.0,
    peak_gpu_memory_allocated_bytes=0,
    peak_gpu_memory_reserved_bytes=0,
    checkpoint_path=None,
    error=sys.argv[3],
))
' "$PROJECT_ROOT" "$RUN_DIR/training_summary.json" "$message"
}

if "$PY" "$PROJECT_ROOT/tools/validate_k_assets.py" \
  --dataset "$DATA_PATH" \
  --field "$FIELD_PATH" \
  --field-metadata "$FIELD_METADATA_PATH" \
  --reference-provenance "$PROVENANCE_PATH" \
  --run-kind "$RUN_KIND" > "$RUN_DIR/asset_preflight.json"; then
  :
else
  preflight_status=$?
  write_failed_summary "asset preflight failed with exit code $preflight_status"
  exit "$preflight_status"
fi

export PINN_RUN_KIND="$RUN_KIND"
export PINN_MODEL_ROLE="$ROLE"
export PINN_EXP_NAME="$EXP_NAME"
export PINN_TRAINING_STRATEGY="$TRAINING_STRATEGY"
export PINN_LEAVE_OUT="$LEAVE_OUT"
export PINN_SEED="$SEED"
export PINN_GPU_ID=0
export PINN_DATA_HOLDOUT=0
unset PINN_CKPT_TAG PINN_HOLDOUT_MODE PINN_HOLDOUT_SEED PINN_HOLDOUT_FRACTION
unset PINN_HOLDOUT_HOLD_EVERY PINN_HOLDOUT_HOLD_START PINN_HOLDOUT_R_LO PINN_HOLDOUT_R_HI
unset PINN_HOLDOUT_SPLIT_X PINN_HOLDOUT_SPLIT_Y
export PINN_PC_ENABLE=0
export PINN_PC_ENTRY=0
export PINN_T_REF=31544.99609375
export PINN_ANALYSIS_T_MAX=31544.99609375
export PINN_MAX_ADAM_ITERS="$MAX_ADAM_ITERS"
export PINN_DATA_PT="$DATA_PATH"
export PINN_K_FIELD_NPZ="$FIELD_PATH"
export PINN_K_FIELD_METADATA="$FIELD_METADATA_PATH"
export PINN_REFERENCE_PROVENANCE="$PROVENANCE_PATH"
export PINN_RUN_DIR="$RUN_DIR"
export PINN_CKPT_DIR="$RUN_DIR"
export PINN_PHYS_TAG=K_nopc_Hwindow_frozenH
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"
unset PYTORCH_CUDA_ALLOC_CONF
export PYTORCH_ALLOC_CONF=expandable_segments:True

exec > "$RUN_DIR/training_log.out" 2> "$RUN_DIR/training_log.err"
echo "Configuration K task: index=$ARRAY_INDEX role=$ROLE seed=$SEED"
echo "Run kind: $RUN_KIND"
echo "Run dir: $RUN_DIR"
echo "Slurm job: ${SLURM_JOB_ID:-not-under-slurm}"
echo "Host: $(hostname)"
echo "Python: $($PY -V 2>&1)"
echo "Start: $(date --iso-8601=seconds)"
nvidia-smi -L

cd "$PROJECT_ROOT"
if srun "$PY" -u "$PROJECT_ROOT/pinn_experiment_M0_M7_BASE_leave_one_out.py"; then
  :
else
  training_status=$?
  if [ ! -f "$RUN_DIR/training_summary.json" ]; then
    write_failed_summary "training launch failed with exit code $training_status"
  fi
  echo "Training failed with exit code $training_status" >&2
  exit "$training_status"
fi

if [ ! -f "$RUN_DIR/final_checkpoint.pt" ] || [ ! -f "$RUN_DIR/training_summary.json" ]; then
  write_failed_summary "training process returned success without required final artifacts"
  exit 7
fi
echo "End: $(date --iso-8601=seconds)"
