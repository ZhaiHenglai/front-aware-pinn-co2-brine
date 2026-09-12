#!/bin/bash
#SBATCH --job-name=K_HK_common
#SBATCH --partition=uoa-compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=0-12:00:00
#SBATCH --output=K_HK_common_%j.out
#SBATCH --error=K_HK_common_%j.err

set -euo pipefail
: "${PINN_PROJECT_ROOT:?}"
: "${PINN_EVALUATION_ROOT:?}"
: "${PINN_H_DATA:?}"
: "${PINN_POSTPROCESS_ROOT:?}"
: "${PY:?}"
cd "$PINN_PROJECT_ROOT"
mkdir -p "$PINN_POSTPROCESS_ROOT"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/pinn12_matplotlib_${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR"

"$PY" postprocess_20260906/verify_completed_supplement.py \
  --evaluation-root "$PINN_EVALUATION_ROOT" \
  --report "$PINN_POSTPROCESS_ROOT/completed_supplement_verification.json"

K_META=$(find "$PINN_EVALUATION_ROOT/by_run/complete/complete_seed0" \
  -type f -name snapshot_metadata.json -print -quit)
[ -n "$K_META" ]

"$PY" postprocess_20260906/build_hk_common_time_reference.py \
  --h-data "$PINN_H_DATA" \
  --k-data "$PINN_PROJECT_ROOT/data/tables_cache_0_775_step1.pt" \
  --k-field "$PINN_PROJECT_ROOT/permeability/K_field_unique_grid.npz" \
  --k-snapshot-metadata "$K_META" \
  --output "$PINN_POSTPROCESS_ROOT/hk_common_time"

echo HK_COMMON_TIME_JOB_COMPLETED
