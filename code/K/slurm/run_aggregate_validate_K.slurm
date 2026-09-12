#!/bin/bash
#SBATCH --job-name=K_aggregate
#SBATCH --partition=uoa-compute
#SBATCH --output=K_aggregate_%j.out
#SBATCH --error=K_aggregate_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -eo pipefail
source /etc/profile
module purge
module load slurm
set -u

PROJECT_ROOT="${PINN_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [ -z "$PROJECT_ROOT" ] || [ ! -f "$PROJECT_ROOT/configuration_k.py" ]; then
  echo "Cannot resolve Configuration K project root; export PINN_PROJECT_ROOT or submit from the project root." >&2
  exit 3
fi
PROJECT_ROOT="$(cd -- "$PROJECT_ROOT" && pwd -P)"
PY="${PY:-$HOME/conda/envs/py311/bin/python}"
if [ ! -x "$PY" ]; then
  echo "Python executable not found or not executable: $PY" >&2
  exit 3
fi

"$PROJECT_ROOT/slurm/verify_code_manifest.sh" "$PROJECT_ROOT"

DATA_PATH="${PINN_DATA_PT:-$PROJECT_ROOT/data/tables_cache_0_775_step1.pt}"
FIELD_PATH="${PINN_K_FIELD_NPZ:-$PROJECT_ROOT/permeability/K_field_unique_grid.npz}"
FIELD_METADATA_PATH="${PINN_K_FIELD_METADATA:-$PROJECT_ROOT/permeability/K_field_unique_grid.json}"
PROVENANCE_PATH="${PINN_REFERENCE_PROVENANCE:-$PROJECT_ROOT/manifests/k_reference_provenance.json}"
OUTPUT_ROOT="${PINN_OUTPUT_ROOT:-$PROJECT_ROOT}"
EVAL_ROOT="${PINN_EVAL_ROOT:-$OUTPUT_ROOT/evaluation}"

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_THREADING_LAYER=GNU

cd "$PROJECT_ROOT"
srun "$PY" -u "$PROJECT_ROOT/evaluation/aggregate_k_results.py" \
  --output-root "$OUTPUT_ROOT" \
  --evaluation-root "$EVAL_ROOT" \
  --data "$DATA_PATH" \
  --field "$FIELD_PATH" \
  --field-metadata "$FIELD_METADATA_PATH" \
  --reference-provenance "$PROVENANCE_PATH"

srun "$PY" -u "$PROJECT_ROOT/tools/validate_heterogeneous_results.py" \
  --csv "$EVAL_ROOT/heterogeneous_metrics_by_seed.csv" \
  > "$EVAL_ROOT/heterogeneous_results_validation.json"
