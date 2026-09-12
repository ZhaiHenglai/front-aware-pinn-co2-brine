#!/bin/bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 TASK_INDEX" >&2
  exit 2
fi

TASK_INDEX="$1"
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
print("\t".join(map(str, (task.array_index, task.role, task.seed, task.run_id))))
' "$PROJECT_ROOT" "$TASK_INDEX")"
IFS=$'\t' read -r ARRAY_INDEX ROLE SEED RUN_ID <<< "$TASK_TSV"

DATA_PATH="${PINN_DATA_PT:-$PROJECT_ROOT/data/tables_cache_0_775_step1.pt}"
FIELD_PATH="${PINN_K_FIELD_NPZ:-$PROJECT_ROOT/permeability/K_field_unique_grid.npz}"
FIELD_METADATA_PATH="${PINN_K_FIELD_METADATA:-$PROJECT_ROOT/permeability/K_field_unique_grid.json}"
PROVENANCE_PATH="${PINN_REFERENCE_PROVENANCE:-$PROJECT_ROOT/manifests/k_reference_provenance.json}"
OUTPUT_ROOT="${PINN_OUTPUT_ROOT:-$PROJECT_ROOT}"
RUN_DIR="$OUTPUT_ROOT/runs/$ROLE/$RUN_ID"
CHECKPOINT="$RUN_DIR/final_checkpoint.pt"
EVAL_ROOT="${PINN_EVAL_ROOT:-$OUTPUT_ROOT/evaluation}"
EVAL_DIR="$EVAL_ROOT/by_run/$ROLE/$RUN_ID"

for required in "$DATA_PATH" "$FIELD_PATH" "$FIELD_METADATA_PATH" "$PROVENANCE_PATH" "$CHECKPOINT"; do
  if [ ! -f "$required" ]; then
    echo "Required Configuration K evaluation input not found: $required" >&2
    exit 4
  fi
done

RUNNER=(
  "$PY" -u "$PROJECT_ROOT/evaluation/run_one_k_evaluation.py"
  --task-index "$ARRAY_INDEX"
  --checkpoint "$CHECKPOINT"
  --data "$DATA_PATH"
  --field "$FIELD_PATH"
  --field-metadata "$FIELD_METADATA_PATH"
  --reference-provenance "$PROVENANCE_PATH"
  --out-dir "$EVAL_DIR"
  --python "$PY"
  --device cuda
  --gpu-id "${PINN_GPU_ID:-0}"
)

if [ "${PINN_DRY_RUN:-0}" = "1" ]; then
  "${RUNNER[@]}" --dry-run
  exit 0
fi

"$SCRIPT_DIR/verify_code_manifest.sh" "$PROJECT_ROOT"
"$SCRIPT_DIR/require_cuda_gpu.sh" "$PY"

TRAINING_SUMMARY="$RUN_DIR/training_summary.json"
RESOLVED_CONFIG="$RUN_DIR/resolved_config.json"
for required in "$TRAINING_SUMMARY" "$RESOLVED_CONFIG"; do
  if [ ! -f "$required" ]; then
    echo "Formal training evidence not found: $required" >&2
    exit 5
  fi
done
"$PY" -c '
import json, sys
record = json.load(open(sys.argv[1], encoding="utf-8"))
status = record.get("status")
if status != "completed":
    raise SystemExit(f"Training status is not completed: {status!r}")
' "$TRAINING_SUMMARY"

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_THREADING_LAYER=GNU
unset PYTORCH_CUDA_ALLOC_CONF
export PYTORCH_ALLOC_CONF=expandable_segments:True

cd "$PROJECT_ROOT"
srun "${RUNNER[@]}"
