import ast
from pathlib import Path


TRAINER = Path(__file__).resolve().parents[1] / "pinn_experiment_M0_M7_BASE_leave_one_out.py"


def _tree():
    return ast.parse(TRAINER.read_text(encoding="utf-8"))


def _function(tree, name):
    return next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name)


def _call_names(node):
    names = []
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            if isinstance(child.func, ast.Name):
                names.append(child.func.id)
            elif isinstance(child.func, ast.Attribute):
                names.append(child.func.attr)
    return names


def test_dataset_is_cropped_before_nondimensional_training_tensors_are_created():
    tree = _tree()
    crop_line = min(
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        and node.func.id == "crop_row_aligned_arrays"
    )
    t_all_line = min(
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        and any(isinstance(target, ast.Name) and target.id == "t_all"
                for target in (node.targets if isinstance(node, ast.Assign) else [node.target]))
    )

    assert crop_line < t_all_line


def test_every_physics_flux_path_uses_shared_permeability_aware_darcy_helper():
    tree = _tree()

    for name in ("pde_residual", "total_velocity", "phase_mass_fluxes"):
        assert "darcy_velocity_components" in _call_names(_function(tree, name)), name


def test_checkpoint_config_merges_resolved_k_evidence():
    tree = _tree()
    function = _function(tree, "build_ckpt_config")

    assert "resolved_checkpoint_config" in _call_names(function)


def test_trainer_fits_shared_eos_without_passing_the_model_seed():
    tree = _tree()
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "fit_eos_record"
    ]

    assert len(calls) == 1
    assert all(
        not (isinstance(node, ast.Name) and node.id == "SEED")
        for node in ast.walk(calls[0])
    )


def test_checkpoint_and_resolved_config_embed_the_sealed_eos_record():
    function = _function(_tree(), "build_ckpt_config")
    eos_entries = []
    for node in ast.walk(function):
        if not isinstance(node, ast.Dict):
            continue
        for key, value in zip(node.keys, node.values):
            if isinstance(key, ast.Constant) and key.value == "eos_record":
                eos_entries.append(value)

    assert len(eos_entries) == 1
    assert isinstance(eos_entries[0], ast.Name)
    assert eos_entries[0].id == "EOS_RECORD"


def test_trainer_has_no_data_max_time_reference_assignment():
    tree = _tree()
    bad_assignments = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == "USE_DATA_MAX_TIME_AS_T_REF" for target in node.targets):
            if isinstance(node.value, ast.Constant) and node.value.value is True:
                bad_assignments.append(node.lineno)

    assert bad_assignments == []


def test_no_runtime_profile_retains_a_nonzero_global_rar_selection():
    tree = _tree()
    bad_values = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        for key, value in zip(node.keys, node.values):
            if isinstance(key, ast.Constant) and key.value == "N_SEL_G":
                if isinstance(value, ast.Constant) and value.value != 0:
                    bad_values.append((value.lineno, value.value))

    assert bad_values == []


def test_resolved_config_file_uses_the_full_checkpoint_configuration():
    source = TRAINER.read_text(encoding="utf-8")

    assert 'write_json_atomic(RUN_DIR / "resolved_config.json", build_ckpt_config())' in source


def test_training_loop_fails_closed_on_nonfinite_loss_gradients_and_parameters():
    train = _function(_tree(), "train")
    calls = []
    for node in ast.walk(train):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            name = node.func.attr
        else:
            continue
        calls.append((node.lineno, name))

    first = lambda name: min(line for line, call_name in calls if call_name == name)
    assert first("ensure_finite_training_loss") < first("backward")
    assert first("backward") < first("ensure_finite_gradients") < first("step")
    assert first("step") < first("ensure_finite_parameters")
