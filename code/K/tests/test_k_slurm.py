import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
COMMON = ROOT / "slurm/run_pinn_K_common.sh"
SMOKE = ROOT / "slurm/run_pinn_K_smoke.slurm"
FORMAL = ROOT / "slurm/run_pinn_K_formal_array.slurm"
EVAL_COMMON = ROOT / "slurm/run_evaluate_K_common.sh"
EVAL_FORMAL = ROOT / "slurm/run_evaluate_K_formal_array.slurm"
AGGREGATE = ROOT / "slurm/run_aggregate_validate_K.slurm"
SUBMIT = ROOT / "slurm/submit_configuration_k.sh"
GPU_PREFLIGHT = ROOT / "slurm/require_cuda_gpu.sh"
CODE_PREFLIGHT = ROOT / "slurm/verify_code_manifest.sh"


def _dummy_assets(tmp_path):
    paths = {}
    for key, name in (
        ("PINN_DATA_PT", "dataset.pt"),
        ("PINN_K_FIELD_NPZ", "field.npz"),
        ("PINN_K_FIELD_METADATA", "field.json"),
        ("PINN_REFERENCE_PROVENANCE", "provenance.json"),
    ):
        path = tmp_path / name
        path.write_text("dry-run placeholder", encoding="utf-8")
        paths[key] = str(path)
    return paths


def _fake_nvidia_smi(tmp_path, *, exit_code):
    bin_dir = tmp_path / "fake-bin"
    bin_dir.mkdir(exist_ok=True)
    executable = bin_dir / "nvidia-smi"
    executable.write_text(
        "#!/bin/bash\n"
        f"exit {int(exit_code)}\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    return bin_dir


def _fake_torch_modules(
    tmp_path,
    *,
    cuda_available,
    gpu_name="Tesla V100-PCIE-32GB",
    total_memory_bytes=32 * 1024**3,
):
    module_dir = tmp_path / "fake-python-modules"
    torch_dir = module_dir / "torch"
    torch_dir.mkdir(parents=True, exist_ok=True)
    torch_dir.joinpath("__init__.py").write_text(
        "class Tensor: pass\n"
        "class cuda:\n"
        "    @staticmethod\n"
        f"    def is_available(): return {bool(cuda_available)!r}\n"
        "    @staticmethod\n"
        f"    def device_count(): return {1 if cuda_available else 0}\n"
        "    @staticmethod\n"
        f"    def get_device_name(index): return {gpu_name!r}\n"
        "    @staticmethod\n"
        f"    def get_device_properties(index): return type('Props', (), {{'total_memory': {int(total_memory_bytes)}}})()\n",
        encoding="utf-8",
    )
    torch_dir.joinpath("nn.py").write_text("class Module: pass\n", encoding="utf-8")
    return module_dir


def _run_common(tmp_path, task_index, run_kind):
    env = os.environ.copy()
    env.update(_dummy_assets(tmp_path))
    env.update(
        {
            "PY": sys.executable,
            "PINN_DRY_RUN": "1",
            "PINN_OUTPUT_ROOT": str(tmp_path / "outputs"),
        }
    )
    return subprocess.run(
        ["bash", str(COMMON), str(task_index), run_kind],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def _run_evaluation_common(tmp_path, task_index):
    assets = _dummy_assets(tmp_path)
    output_root = tmp_path / "outputs"
    role = "no_local_fv" if task_index == 19 else "complete"
    seed = 42 if task_index == 19 else 0
    run_id = f"{role}_seed{seed}"
    run_dir = output_root / "runs" / role / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "final_checkpoint.pt").write_text("dry-run checkpoint", encoding="utf-8")
    module_dir = _fake_torch_modules(tmp_path, cuda_available=False)
    env = os.environ.copy()
    env.update(assets)
    env.update(
        {
            "PY": sys.executable,
            "PYTHONPATH": str(module_dir),
            "PINN_DRY_RUN": "1",
            "PINN_OUTPUT_ROOT": str(output_root),
        }
    )
    return subprocess.run(
        ["bash", str(EVAL_COMMON), str(task_index)],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
def test_common_launcher_dry_run_resolves_unique_formal_task(tmp_path):
    result = _run_common(tmp_path, 19, "formal")

    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    assert payload["array_index"] == 19
    assert payload["role"] == "no_local_fv"
    assert payload["exp_name"] == "M7"
    assert payload["leave_out"] == "FV"
    assert payload["seed"] == 42
    assert payload["run_dir"].endswith("runs/no_local_fv/no_local_fv_seed42")
    assert payload["asset_content_validation"] == "deferred_until_execution"


def test_common_launcher_rejects_task_outside_registered_matrix(tmp_path):
    result = _run_common(tmp_path, 20, "formal")

    assert result.returncode != 0
    assert "0..19" in result.stderr


def test_common_launcher_refuses_even_a_partial_existing_run_directory(tmp_path):
    output_root = tmp_path / "outputs"
    existing = output_root / "runs/no_local_fv/no_local_fv_seed42"
    existing.mkdir(parents=True)
    (existing / "partial.log").write_text("failed prior attempt", encoding="utf-8")
    env = os.environ.copy()
    env.update(_dummy_assets(tmp_path))
    env.update({
        "PY": sys.executable,
        "PINN_DRY_RUN": "1",
        "PINN_OUTPUT_ROOT": str(output_root),
    })

    result = subprocess.run(
        ["bash", str(COMMON), "19", "formal"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "Refusing to overwrite" in result.stderr


def test_asset_preflight_failure_writes_failed_training_summary(tmp_path):
    output_root = tmp_path / "outputs"
    bin_dir = _fake_nvidia_smi(tmp_path, exit_code=0)
    module_dir = _fake_torch_modules(tmp_path, cuda_available=True)
    env = os.environ.copy()
    env.update(_dummy_assets(tmp_path))
    env.update(
        {
            "PATH": f"{bin_dir}:{env['PATH']}",
            "PY": sys.executable,
            "PYTHONPATH": str(module_dir),
            "PINN_OUTPUT_ROOT": str(output_root),
        }
    )

    result = subprocess.run(
        ["bash", str(COMMON), "4", "formal"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    summary_path = output_root / "runs/complete/complete_seed0/training_summary.json"
    assert summary_path.is_file()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["status"] == "failed"
    assert "asset preflight" in summary["error"]


def test_smoke_launcher_resolves_complete_seed_zero(tmp_path):
    result = _run_common(tmp_path, 4, "smoke")

    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    assert payload["role"] == "complete"
    assert payload["seed"] == 0
    assert payload["run_dir"].endswith("smoke/complete_seed0")


def test_evaluation_launcher_dry_run_resolves_matching_formal_checkpoint(tmp_path):
    result = _run_evaluation_common(tmp_path, 19)

    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    assert payload["task"]["role"] == "no_local_fv"
    assert payload["task"]["seed"] == 42
    assert payload["commands"]["global"][payload["commands"]["global"].index("--ckpt-dir") + 1].endswith(
        "runs/no_local_fv/no_local_fv_seed42"
    )
    assert payload["content_validation"] == "deferred_until_execution"


def test_gpu_jobs_fail_closed_when_nvidia_smi_is_unusable(tmp_path):
    bin_dir = _fake_nvidia_smi(tmp_path, exit_code=9)
    assets = _dummy_assets(tmp_path)

    for task_index, run_kind in ((4, "smoke"), (4, "formal")):
        env = os.environ.copy()
        env.update(assets)
        env.update(
            {
                "PATH": f"{bin_dir}:{env['PATH']}",
                "PY": sys.executable,
                "PINN_OUTPUT_ROOT": str(tmp_path / f"{run_kind}-outputs"),
            }
        )
        result = subprocess.run(
            ["bash", str(COMMON), str(task_index), run_kind],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode != 0
        assert "nvidia-smi" in result.stderr, result.stdout + result.stderr

    output_root = tmp_path / "evaluation-outputs"
    checkpoint = output_root / "runs/complete/complete_seed0/final_checkpoint.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text("placeholder", encoding="utf-8")
    eval_env = os.environ.copy()
    eval_env.update(assets)
    eval_env.update(
        {
            "PATH": f"{bin_dir}:{eval_env['PATH']}",
            "PY": sys.executable,
            "PINN_OUTPUT_ROOT": str(output_root),
        }
    )
    result = subprocess.run(
        ["bash", str(EVAL_COMMON), "4"],
        cwd=ROOT,
        env=eval_env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "nvidia-smi" in result.stderr, result.stdout + result.stderr


def test_gpu_preflight_fails_closed_when_pytorch_cuda_is_unavailable(tmp_path):
    bin_dir = _fake_nvidia_smi(tmp_path, exit_code=0)
    module_dir = _fake_torch_modules(tmp_path, cuda_available=False)
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{bin_dir}:{env['PATH']}",
            "PYTHONPATH": str(module_dir),
        }
    )

    result = subprocess.run(
        ["bash", str(GPU_PREFLIGHT), sys.executable],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "CUDA is unavailable" in result.stderr, result.stdout + result.stderr


def test_gpu_preflight_requires_the_h_case_v100_32gb_profile(tmp_path):
    bin_dir = _fake_nvidia_smi(tmp_path, exit_code=0)
    for gpu_name, total_memory_bytes, expected_error in (
        ("Tesla T4", 16 * 1024**3, "V100"),
        ("Tesla V100-PCIE-16GB", 16 * 1024**3, "memory"),
    ):
        module_dir = _fake_torch_modules(
            tmp_path,
            cuda_available=True,
            gpu_name=gpu_name,
            total_memory_bytes=total_memory_bytes,
        )
        env = os.environ.copy()
        env.update(
            {
                "PATH": f"{bin_dir}:{env['PATH']}",
                "PYTHONPATH": str(module_dir),
            }
        )

        result = subprocess.run(
            ["bash", str(GPU_PREFLIGHT), sys.executable],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )

        assert result.returncode != 0
        assert expected_error in result.stderr, result.stdout + result.stderr


def test_submitter_builds_afterok_dependency_chain(tmp_path):
    log = tmp_path / "sbatch.log"
    counter = tmp_path / "counter"
    counter.write_text("1000", encoding="utf-8")
    fake = tmp_path / "fake_sbatch.sh"
    fake.write_text(
        "#!/bin/bash\n"
        "set -euo pipefail\n"
        "n=$(cat \"$SBATCH_COUNTER_FILE\")\n"
        "n=$((n + 1))\n"
        "printf '%s' \"$n\" > \"$SBATCH_COUNTER_FILE\"\n"
        "printf '%s\\n' \"$*\" >> \"$SBATCH_LOG\"\n"
        "printf '%s\\n' \"$n\"\n",
        encoding="utf-8",
    )
    fake.chmod(0o755)
    fake_python = tmp_path / "fake_python.sh"
    fake_python.write_text(
        "#!/bin/bash\n"
        "set -euo pipefail\n"
        "if [[ ${1:-} == *validate_k_assets.py ]]; then printf '%s\\n' '{\"status\":\"pass\"}'; fi\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    assets = _dummy_assets(tmp_path)
    output_root = tmp_path / "outputs"
    smoke_dir = output_root / "smoke/complete_seed0"
    smoke_dir.mkdir(parents=True)
    (smoke_dir / "training_summary.json").write_text('{"status":"completed"}', encoding="utf-8")
    (smoke_dir / "resolved_config.json").write_text(
        '{"configuration":"K","run_kind":"smoke","pc_enabled":false}', encoding="utf-8"
    )
    env = os.environ.copy()
    env.update(assets)
    env.update(
        {
            "SBATCH_BIN": str(fake),
            "SBATCH_COUNTER_FILE": str(counter),
            "SBATCH_LOG": str(log),
            "PY": str(fake_python),
            "PINN_OUTPUT_ROOT": str(output_root),
        }
    )

    result = subprocess.run(
        ["bash", str(SUBMIT)],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "training_job_id=1001" in result.stdout
    assert "evaluation_job_id=1002" in result.stdout
    assert "aggregation_job_id=1003" in result.stdout
    calls = log.read_text(encoding="utf-8").splitlines()
    assert all("--export=ALL" in call for call in calls)
    assert "--array=0-19" in calls[0]
    assert "--array=0-19" in calls[1]
    assert "--array=0-19" not in calls[2]
    assert "--dependency=afterok:1001" in calls[1]
    assert "--dependency=afterok:1002" in calls[2]


def test_submitter_atomically_allows_only_one_sanitized_formal_chain(tmp_path):
    log = tmp_path / "sbatch.log"
    fake = tmp_path / "fake_sbatch.sh"
    fake.write_text(
        "#!/bin/bash\n"
        "set -euo pipefail\n"
        "printf '%s|dry=%s|tag=%s|holdout=%s|gpu=%s|array_env=%s|eval_root=%s\\n' "
        '"$*" "${PINN_DRY_RUN:-}" "${PINN_CKPT_TAG:-}" "${PINN_DATA_HOLDOUT:-}" '
        '"${PINN_GPU_ID:-}" "${SBATCH_ARRAY_INX:-}" "${PINN_EVAL_ROOT:-}" >> "$SBATCH_LOG"\n'
        "sleep 0.1\n"
        "case \"${*: -1}\" in\n"
        "  *run_pinn_K_formal_array.slurm) printf '%s\\n' 2001 ;;\n"
        "  *run_evaluate_K_formal_array.slurm) printf '%s\\n' 2002 ;;\n"
        "  *run_aggregate_validate_K.slurm) printf '%s\\n' 2003 ;;\n"
        "  *) exit 8 ;;\n"
        "esac\n",
        encoding="utf-8",
    )
    fake.chmod(0o755)
    fake_python = tmp_path / "fake_python.sh"
    fake_python.write_text(
        "#!/bin/bash\n"
        "set -euo pipefail\n"
        "if [[ ${1:-} == *validate_k_assets.py ]]; then printf '%s\\n' '{\"status\":\"pass\"}'; fi\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    assets = _dummy_assets(tmp_path)
    output_root = tmp_path / "outputs"
    smoke_dir = output_root / "smoke/complete_seed0"
    smoke_dir.mkdir(parents=True)
    (smoke_dir / "training_summary.json").write_text('{"status":"completed"}', encoding="utf-8")
    (smoke_dir / "resolved_config.json").write_text(
        '{"configuration":"K","run_kind":"smoke","pc_enabled":false}', encoding="utf-8"
    )
    env = os.environ.copy()
    env.update(assets)
    env.update(
        {
            "SBATCH_BIN": str(fake),
            "SBATCH_LOG": str(log),
            "SBATCH_ARRAY_INX": "4",
            "PY": str(fake_python),
            "PINN_OUTPUT_ROOT": str(output_root),
            "PINN_DRY_RUN": "1",
            "PINN_CKPT_TAG": "inherited-tag",
            "PINN_DATA_HOLDOUT": "1",
            "PINN_GPU_ID": "7",
            "PINN_EVAL_ROOT": str(tmp_path / "wrong-evaluation-root"),
        }
    )

    processes = [
        subprocess.Popen(
            ["bash", str(SUBMIT)],
            cwd=ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        for _ in range(2)
    ]
    results = [process.communicate(timeout=20) + (process.returncode,) for process in processes]

    assert sorted(result[2] for result in results) == [0, 6], results
    refused = next(result for result in results if result[2] == 6)
    assert "submission_state.txt" in refused[1]
    assert "archive all formal artifacts" in refused[1]
    calls = log.read_text(encoding="utf-8").splitlines()
    assert len(calls) == 3, calls
    training_call = next(call for call in calls if "run_pinn_K_formal_array.slurm" in call)
    evaluation_call = next(call for call in calls if "run_evaluate_K_formal_array.slurm" in call)
    aggregation_call = next(call for call in calls if "run_aggregate_validate_K.slurm" in call)
    assert "--array=0-19" in training_call
    assert "--array=0-19" in evaluation_call
    assert "--array=0-19" not in aggregation_call
    for call in calls:
        assert "|dry=|tag=|holdout=|gpu=0|array_env=|eval_root=" in call, call

    state = output_root / "manifests/formal_submission.lock/submission_state.txt"
    assert state.is_file()
    state_text = state.read_text(encoding="utf-8")
    assert "status=submitted" in state_text
    assert "training_job_id=2001" in state_text
    assert "evaluation_job_id=2002" in state_text
    assert "aggregation_job_id=2003" in state_text
    assert "recovery_policy=" in state_text
    assert "sacct" in state_text


def test_submitter_requires_formal_asset_preflight_and_completed_smoke():
    text = SUBMIT.read_text(encoding="utf-8")

    assert "tools/validate_k_assets.py" in text
    assert "--run-kind" in text and "formal" in text
    assert "smoke/complete_seed0/training_summary.json" in text
    assert "smoke/complete_seed0/resolved_config.json" in text
    assert "unset SBATCH_NODELIST SBATCH_EXCLUDE" in text


def test_aggregation_job_is_cpu_only_and_runs_standalone_validator():
    text = AGGREGATE.read_text(encoding="utf-8")

    assert "--gres=gpu" not in text
    assert "evaluation/aggregate_k_results.py" in text
    assert "tools/validate_heterogeneous_results.py" in text


def test_new_slurm_entrypoints_have_valid_bash_syntax():
    for path in (COMMON, SMOKE, FORMAL, EVAL_COMMON, EVAL_FORMAL, AGGREGATE, SUBMIT, GPU_PREFLIGHT, CODE_PREFLIGHT):
        result = subprocess.run(["bash", "-n", str(path)], text=True, capture_output=True, check=False)
        assert result.returncode == 0, f"{path}: {result.stderr}"


def test_maxwell_batch_wrappers_source_profile_before_enabling_nounset():
    for path in (SMOKE, FORMAL, EVAL_FORMAL, AGGREGATE):
        source = path.read_text(encoding="utf-8")
        assert source.index("source /etc/profile") < source.index("set -u"), path


def test_training_uses_the_current_pytorch_allocator_variable():
    source = "\n".join(
        path.read_text(encoding="utf-8") for path in (COMMON, EVAL_COMMON)
    )

    assert "PYTORCH_ALLOC_CONF=expandable_segments:True" in source
    assert "export PYTORCH_CUDA_ALLOC_CONF" not in source
    assert "unset PYTORCH_CUDA_ALLOC_CONF" in source


def test_every_executing_stage_checks_the_frozen_code_manifest():
    for path in (COMMON, EVAL_COMMON, AGGREGATE, SUBMIT):
        source = path.read_text(encoding="utf-8")
        assert "verify_code_manifest.sh" in source, path


def test_spooled_batch_wrappers_resolve_the_exported_project_root():
    # Slurm executes a copied spool script, so BASH_SOURCE inside a .slurm file
    # is not a stable way to recover the submitted project's directory.
    for path in (SMOKE, FORMAL, EVAL_FORMAL, AGGREGATE):
        source = path.read_text(encoding="utf-8")
        assert "PINN_PROJECT_ROOT" in source, path
        assert "SLURM_SUBMIT_DIR" in source, path

    submitter = SUBMIT.read_text(encoding="utf-8")
    assert 'PINN_PROJECT_ROOT="$PROJECT_ROOT"' in submitter


def test_gpu_batch_wrappers_clear_inherited_dry_run_and_holdout_controls():
    for path in (SMOKE, FORMAL, EVAL_FORMAL):
        source = path.read_text(encoding="utf-8")
        assert "unset PINN_DRY_RUN" in source, path
        assert "PINN_CKPT_TAG" in source, path
        assert "PINN_DATA_HOLDOUT" in source, path
        assert "export PINN_GPU_ID=0" in source, path
