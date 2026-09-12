#!/bin/bash
#SBATCH --job-name=K_evaluate
#SBATCH --partition=spot-gpu
#SBATCH --output=K_evaluate_%A_%a.out
#SBATCH --error=K_evaluate_%A_%a.err
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --gres=gpu:Tesla_V100-PCIE-32GB:1
#SBATCH --array=0-19

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
unset PINN_DRY_RUN PINN_CKPT_TAG PINN_DATA_HOLDOUT PINN_HOLDOUT_MODE PINN_HOLDOUT_SEED
unset PINN_HOLDOUT_FRACTION PINN_HOLDOUT_HOLD_EVERY PINN_HOLDOUT_HOLD_START
unset PINN_HOLDOUT_R_LO PINN_HOLDOUT_R_HI PINN_HOLDOUT_SPLIT_X PINN_HOLDOUT_SPLIT_Y PINN_EVAL_ROOT
export PINN_GPU_ID=0
"$PROJECT_ROOT/slurm/run_evaluate_K_common.sh" "$SLURM_ARRAY_TASK_ID"
