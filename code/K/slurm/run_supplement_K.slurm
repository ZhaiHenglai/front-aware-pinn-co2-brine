#!/bin/bash
#SBATCH --job-name=K_supplement
#SBATCH --partition=spot-gpu
#SBATCH --output=K_supplement_%A_%a.out
#SBATCH --error=K_supplement_%A_%a.err
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --gres=gpu:Tesla_V100-PCIE-32GB:1
#SBATCH --array=0-19%4
set -eo pipefail
source /etc/profile
module purge
module load slurm
set -u
: "${PINN_PROJECT_ROOT:?Export PINN_PROJECT_ROOT before sbatch}"
: "${PINN_SUPPLEMENT_ROOT:?Export a new PINN_SUPPLEMENT_ROOT before sbatch}"
PY="${PY:-$HOME/conda/envs/py311/bin/python}"
cd "$PINN_PROJECT_ROOT"
bash slurm/verify_code_manifest.sh "$PINN_PROJECT_ROOT"
bash slurm/require_cuda_gpu.sh "$PY"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 MKL_THREADING_LAYER=GNU MPLBACKEND=Agg
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS" OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS" NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"
unset PINN_DATA_HOLDOUT PINN_HOLDOUT_MODE PINN_HOLDOUT_SEED PINN_HOLDOUT_FRACTION
RETRY_ARGS=()
if [ "${PINN_SUPPLEMENT_RETRY:-0}" = 1 ]; then RETRY_ARGS=(--retry); fi
srun "$PY" -u evaluation/run_k_supplement.py --task-index "$SLURM_ARRAY_TASK_ID" \
  --evaluation-root "$PINN_SUPPLEMENT_ROOT" --device cuda "${RETRY_ARGS[@]}"
