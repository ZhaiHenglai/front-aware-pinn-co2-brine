#!/bin/bash
#SBATCH --job-name=K_ref_audit
#SBATCH --partition=uoa-compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=0-08:00:00
#SBATCH --output=K_ref_audit_%j.out
#SBATCH --error=K_ref_audit_%j.err

set -euo pipefail
: "${PINN_PROJECT_ROOT:?}"
: "${PINN_REFERENCE_SEARCH_ROOT:?}"
: "${PINN_POSTPROCESS_ROOT:?}"
: "${PY:?}"
cd "$PINN_PROJECT_ROOT"
mkdir -p "$PINN_POSTPROCESS_ROOT/reference_evidence"
"$PY" postprocess_20260906/reference_evidence_inventory.py \
  --search-root "$PINN_REFERENCE_SEARCH_ROOT" \
  --output "$PINN_POSTPROCESS_ROOT/reference_evidence"
echo REFERENCE_EVIDENCE_INVENTORY_COMPLETED
