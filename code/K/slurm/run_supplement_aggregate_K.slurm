#!/bin/bash
#SBATCH --job-name=K_supp_summary
#SBATCH --partition=uoa-compute
#SBATCH --output=K_supp_summary_%j.out
#SBATCH --error=K_supp_summary_%j.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
set -eo pipefail
source /etc/profile
module purge
module load slurm
set -u
: "${PINN_PROJECT_ROOT:?Export PINN_PROJECT_ROOT}"
: "${PINN_SUPPLEMENT_ROOT:?Export PINN_SUPPLEMENT_ROOT}"
PY="${PY:-$HOME/conda/envs/py311/bin/python}"
cd "$PINN_PROJECT_ROOT"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 MKL_THREADING_LAYER=GNU MPLBACKEND=Agg
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS" OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
bash slurm/verify_code_manifest.sh "$PINN_PROJECT_ROOT"
srun "$PY" -u evaluation/assemble_k_supplement.py --evaluation-root "$PINN_SUPPLEMENT_ROOT"
srun "$PY" tools/validate_heterogeneous_results.py \
  --csv "$PINN_SUPPLEMENT_ROOT/heterogeneous_metrics_by_seed.csv"

