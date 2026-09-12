#!/bin/bash
#SBATCH --job-name=K_post_final
#SBATCH --partition=uoa-compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0-02:00:00
#SBATCH --output=K_post_final_%j.out
#SBATCH --error=K_post_final_%j.err

set -euo pipefail
: "${PINN_PROJECT_ROOT:?}"
: "${PINN_POSTPROCESS_ROOT:?}"
: "${PY:?}"
cd "$PINN_PROJECT_ROOT"
"$PY" postprocess_20260906/finalize_postprocess.py --output-root "$PINN_POSTPROCESS_ROOT"
echo POSTPROCESS_FINALIZATION_COMPLETED
