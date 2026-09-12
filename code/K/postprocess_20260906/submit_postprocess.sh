#!/bin/bash
set -euo pipefail
: "${PINN_PROJECT_ROOT:?}"
: "${PINN_EVALUATION_ROOT:?}"
: "${PINN_H_DATA:?}"
: "${PINN_REFERENCE_SEARCH_ROOT:?}"
: "${PINN_POSTPROCESS_ROOT:?}"
: "${PY:?}"

cd "$PINN_PROJECT_ROOT"
for p in "$PINN_EVALUATION_ROOT" "$PINN_H_DATA" "$PINN_PROJECT_ROOT/data/tables_cache_0_775_step1.pt"; do
  [ -e "$p" ] || { echo "MISSING_REQUIRED_INPUT=$p" >&2; exit 2; }
done

mkdir -p "$PINN_POSTPROCESS_ROOT"
common_job=$(sbatch --parsable --export=ALL postprocess_20260906/slurm/run_verify_and_common_time.slurm)
common_job=${common_job%%;*}
inventory_job=$(sbatch --parsable --export=ALL postprocess_20260906/slurm/run_reference_inventory.slurm)
inventory_job=${inventory_job%%;*}
final_job=$(sbatch --parsable --export=ALL --dependency="afterok:${common_job}:${inventory_job}" postprocess_20260906/slurm/run_finalize.slurm)
final_job=${final_job%%;*}

{
  printf 'common_time_job=%s\n' "$common_job"
  printf 'reference_inventory_job=%s\n' "$inventory_job"
  printf 'finalization_job=%s\n' "$final_job"
  printf 'postprocess_root=%s\n' "$PINN_POSTPROCESS_ROOT"
} | tee "$PINN_POSTPROCESS_ROOT/submission.txt"
squeue -j "$common_job,$inventory_job,$final_job" -o '%.20i %.32j %.10T %.12M %R'
