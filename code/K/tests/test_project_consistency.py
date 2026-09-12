import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
AUDITOR = ROOT / "tools/audit_project_consistency.py"


def test_consistency_auditor_reports_no_stale_masking_or_fixed_node_wrappers():
    result = subprocess.run(
        [sys.executable, str(AUDITOR), "--root", str(ROOT)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout)
    assert report["stale_workdirs"] == []
    assert report["masked_srun_failures"] == []
    assert report["fixed_nodes"] == []
    assert report["stale_dataset_references"] == []
    assert report["missing_required_assets"] == []
    assert report["canonical_submitter"] == "slurm/submit_configuration_k.sh"


def test_operator_readme_names_only_the_k_workflow_as_formal_entrypoint():
    readme = (ROOT / "README_CONFIGURATION_K.md").read_text(encoding="utf-8")

    assert "slurm/run_pinn_K_smoke.slurm" in readme
    assert "slurm/submit_configuration_k.sh" in readme
    assert "exactly 20" in readme
    assert "k_reference_provenance.json" in readme
    assert "K_field_unique_grid.npz" in readme
    assert "sinfo" in readme
    assert "scontrol show partition spot-gpu" in readme
    assert "do not submit" in readme.lower()


def test_environment_and_checksum_templates_cover_required_runtime_assets():
    environment = (ROOT / "environment.yml").read_text(encoding="utf-8")
    checksums = (ROOT / "manifests/SHA256SUMS.template").read_text(encoding="utf-8")

    for dependency in ("python=3.11", "pytorch", "numpy", "pandas", "scipy", "matplotlib", "pytest"):
        assert dependency in environment
    assert "2132a9be710e2f257ea397cd080654e689093d9c561be3edb5be1e471283e761" in checksums
    assert "data/tables_cache_0_775_step1.pt" in checksums
    assert "K_field_unique_grid.npz" in checksums


def test_clean_project_contains_no_training_or_evaluation_result_directories():
    for name in ("Results1", "Results2", "logs", "runs", "smoke", "failed_or_exploratory"):
        assert not (ROOT / name).exists(), f"unexpected retained result directory: {name}"
    checkpoint_files = [
        path for path in ROOT.rglob("*.pt")
        if path.relative_to(ROOT).as_posix() != "data/tables_cache_0_775_step1.pt"
    ]
    assert checkpoint_files == []


def test_runtime_and_operator_files_use_only_the_frozen_775_dataset_identity():
    checked = [
        ROOT / "configuration_k.py",
        ROOT / "README_CONFIGURATION_K.md",
        ROOT / "pinn_experiment_M0_M7_BASE_leave_one_out.py",
        ROOT / "slurm/run_pinn_K_common.sh",
        ROOT / "slurm/run_evaluate_K_common.sh",
        ROOT / "slurm/run_aggregate_validate_K.slurm",
        ROOT / "slurm/submit_configuration_k.sh",
        ROOT / "manifests/SHA256SUMS.template",
        ROOT / "manifests/k_reference_provenance.template.json",
    ]
    combined = "\n".join(path.read_text(encoding="utf-8") for path in checked)

    assert "tables_cache_0_775_step1.pt" in combined
    assert "2132a9be710e2f257ea397cd080654e689093d9c561be3edb5be1e471283e761" in combined
    assert "tables_cache_0_265_step1.pt" not in combined
    assert "0e245470c42585faabc2198950123da2a90439275ce51e0271e35afdee0dfc5f" not in combined
    assert "tables_cache_0_723_step1.pt" not in combined
    assert "b34a14bee121740475f7dc492b0dd2ab532347273c011e1eeeb39f01266e25b7" not in combined


def test_only_k_required_evaluator_implementations_and_no_legacy_entrypoints_remain():
    retained_results_files = {
        path.name for path in (ROOT / "Results").iterdir()
        if path.is_file()
    }
    assert retained_results_files == {
        "evaluate_snapshot_k.py",
        "evaluate_training_stability_k.py",
        "k_spatial_diagnostics.py",
        "_nature_plot_style_M0_M7.py",
        "_time_series_scaling.py",
        "evaluate_global_accuracy_M0_M7_L6_dS_shortlabels.py",
        "evaluate_front_plume_M0_M7_L6_dS.py",
        "evaluate_physics_M0_M7_L6_dS.py",
    }
    assert not (ROOT / "pinn_1d_buckley_leverett.py").exists()
    assert not (ROOT / "run_pinn_v100_3.slurm").exists()
    assert not (ROOT / "hosts.txt").exists()


def test_runtime_sources_do_not_fall_back_to_legacy_account_or_h_result_paths():
    runtime_sources = [
        ROOT / "pinn_experiment_M0_M7_BASE_leave_one_out.py",
        ROOT / "Results/evaluate_global_accuracy_M0_M7_L6_dS_shortlabels.py",
        ROOT / "Results/evaluate_front_plume_M0_M7_L6_dS.py",
        ROOT / "Results/evaluate_physics_M0_M7_L6_dS.py",
    ]
    combined = "\n".join(path.read_text(encoding="utf-8") for path in runtime_sources)

    for forbidden in (
        "<LOCAL_PATH>",
        "<LOCAL_PATH>sst/PINN3",
        '"tables_cache.pt"',
        'DEFAULT_CKPT_DIR = "./Results1"',
        'DEFAULT_OUT_DIR = "./eval_M0_M7/global"',
        'DEFAULT_OUT_ROOT = "./eval_M0_M7"',
    ):
        assert forbidden not in combined


def test_low_level_evaluators_require_explicit_k_input_and_output_locations():
    global_source = (ROOT / "Results/evaluate_global_accuracy_M0_M7_L6_dS_shortlabels.py").read_text(
        encoding="utf-8"
    )
    front_source = (ROOT / "Results/evaluate_front_plume_M0_M7_L6_dS.py").read_text(encoding="utf-8")
    physics_source = (ROOT / "Results/evaluate_physics_M0_M7_L6_dS.py").read_text(encoding="utf-8")

    assert 'ap.add_argument("--ckpt-dir", type=str, required=True)' in global_source
    assert 'ap.add_argument("--data", type=str, required=True)' in global_source
    assert 'ap.add_argument("--out-dir", type=str, required=True)' in global_source
    assert 'ap.add_argument("--ckpt-dir", type=str, required=True)' in front_source
    assert 'ap.add_argument("--data", type=str, required=True)' in front_source
    assert 'ap.add_argument("--out-root", type=str, required=True' in front_source
    assert 'ap.add_argument("--ckpt-dir", type=str, required=True)' in physics_source
    assert 'ap.add_argument("--data", type=str, required=True)' in physics_source
    assert 'ap.add_argument("--out-root", type=str, required=True)' in physics_source


def test_slurm_defaults_to_the_python_environment_proven_by_the_723_h_case():
    runtime_shell = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((ROOT / "slurm").iterdir())
        if path.suffix in {".sh", ".slurm"}
    )

    assert "$HOME/conda/envs/py311/bin/python" in runtime_shell
    assert "$HOME/conda/envs/pinn-k-py311/bin/python" not in runtime_shell


def test_maxwell_resources_follow_the_successful_723_h_case_profile():
    training = (ROOT / "slurm/run_pinn_K_formal_array.slurm").read_text(encoding="utf-8")
    smoke = (ROOT / "slurm/run_pinn_K_smoke.slurm").read_text(encoding="utf-8")
    evaluation = (ROOT / "slurm/run_evaluate_K_formal_array.slurm").read_text(encoding="utf-8")
    aggregation = (ROOT / "slurm/run_aggregate_validate_K.slurm").read_text(encoding="utf-8")

    for source in (training, smoke):
        assert "#SBATCH --partition=spot-gpu" in source
        assert "#SBATCH --cpus-per-task=4" in source
        assert "#SBATCH --mem=80G" in source
        assert "#SBATCH --gres=gpu:Tesla_V100-PCIE-32GB:1" in source
    assert "#SBATCH --time=7-00:00:00" in training

    assert "#SBATCH --partition=spot-gpu" in evaluation
    assert "#SBATCH --time=2-00:00:00" in evaluation
    assert "#SBATCH --gres=gpu:Tesla_V100-PCIE-32GB:1" in evaluation

    assert "#SBATCH --partition=uoa-compute" in aggregation
    assert "#SBATCH --cpus-per-task=4" in aggregation
    assert "#SBATCH --gres=gpu" not in aggregation


def test_operator_guide_records_the_723_h_maxwell_evidence_source():
    readme = (ROOT / "README_CONFIGURATION_K.md").read_text(encoding="utf-8")
    baseline = ROOT / "manifests/MAXWELL_H_CASE_BASELINE.md"

    assert baseline.is_file()
    baseline_text = baseline.read_text(encoding="utf-8")
    assert "PINN12_723.pt/run_pinn_v100_3.slurm" in baseline_text
    assert "egpu001.int.maxwell.abdn.ac.uk" in baseline_text
    assert "Tesla V100-PCIE-32GB" in baseline_text
    assert "Python 3.11.7" in baseline_text
    assert "MAXWELL_H_CASE_BASELINE.md" in readme
