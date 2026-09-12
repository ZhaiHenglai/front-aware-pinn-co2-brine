#!/usr/bin/env python3
from pathlib import Path
import json
import sys

root = Path(__file__).resolve().parents[1]
k_root = root / "code" / "K"
sys.path.insert(0, str(k_root))

import configuration_k
import eos_k
import k_evaluation_runtime
import k_physics
import k_results_validation
import k_training_runtime
from permeability_field_k import validate_field_artifacts

tasks = configuration_k.all_formal_tasks()
configs = list((root / "configurations" / "K").glob("*/resolved_config.json"))
assert len(tasks) == 20
assert len(configs) == 20
field = validate_field_artifacts(
    k_root / "permeability" / "K_field_unique_grid.npz",
    k_root / "permeability" / "K_field_unique_grid.json",
)
print(json.dumps({"status": "PASS", "formal_tasks": len(tasks), "resolved_configs": len(configs), "field_id": field.field_id, "field_shape": field.shape}, indent=2))
