from pathlib import Path
import sys
import json
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "Results"), str(ROOT / "evaluation")]
import evaluate_snapshot_k as snapshot
import k_spatial_diagnostics as diagnostics
import run_k_supplement as runner
import assemble_k_supplement as assembly
from configuration_k import all_formal_tasks


def test_snapshot_duplicate_average_and_well_mask():
    x = np.array([0., 0., 1., 0., 1.])
    y = np.array([0., 0., 0., 1., 1.])
    v = np.array([0.2, 0.6, 0.4, 0.4, 0.4])
    X, Y = np.meshgrid(np.linspace(0, 1, 3), np.linspace(0, 1, 3))
    mask = np.zeros_like(X, dtype=bool)
    mask[-1, -1] = True
    z = snapshot.interpolate_to_grid(x, y, v, X, Y, mask)
    np.testing.assert_allclose(z[~mask], .4)
    assert np.isnan(z[-1, -1])


def test_breakthrough_censoring_and_first_sample():
    assert diagnostics.first_arrival([0, 1, 2], [0, 0.04, 0.049]) is None
    assert diagnostics.first_arrival([0, 1, 2], [0, 0.05, 0.06]) == 1.


def test_all_task_commands_use_exact_checkpoints_and_labels(tmp_path):
    for task in all_formal_tasks():
        cmds = runner.commands(task, tmp_path / task.run_id, device="cpu")
        core = cmds["core"]
        assert core[core.index("--task-index")+1] == str(task.array_index)
        assert str(ROOT / "runs" / task.role / task.run_id / "final_checkpoint.pt") in core
        snap = cmds["snapshots"]
        assert snap[snap.index("--run-label")+1] == task.role
        assert f"LOO-{task.leave_out}_seed{task.seed}_" in snap[snap.index("--ckpt")+1]
        assert "--publication-figures" in core


def test_interactive_submission_entrypoint_returns_to_login_shell():
    source = (ROOT / "slurm" / "submit_supplement_K.sh").read_text(encoding="utf-8")
    assert "pinn12_submit_supplement()" in source
    assert "set +e" in source
    assert "set +u" in source
    assert "set +o pipefail" in source
    assert "set -euo pipefail" not in source
    assert "return \"$pinn12_submit_supplement_rc\"" in source


def test_verify_rejects_incomplete_task(tmp_path):
    task = all_formal_tasks()[0]
    out = tmp_path / "by_run" / task.role / task.run_id
    out.mkdir(parents=True)
    (out / "supplement_summary.json").write_text(json.dumps(
        {"status": "failed", "task": task.as_dict()}))
    with pytest.raises(ValueError, match="Incomplete"):
        assembly.verify(tmp_path)


def test_mass_changed_coordinates_not_just_row_order():
    from test_evaluator_spatial_metrics import (
        _single_layer_mass_fixture, _ConstantDensity, _ZeroSaturationModel, physics, torch)
    arrays, plan, cfg = _single_layer_mass_fixture()
    # A different second-layer support with uniform S=0.4 has exact mass 200 kg.
    second = {"x": np.array([1.,4.,1.,4.]), "y": np.array([1.,1.,4.,4.]),
              "t": np.full(4,10.), "p": np.full(4,10.), "Sco2":np.full(4,.4),
              "rho_c": np.full(4,1000.)}
    arrays = {k:np.r_[v,second[k]] for k,v in arrays.items()}
    plan["time_seconds"] = np.array([0.,10.])
    rows=physics.evaluate_mass(_ZeroSaturationModel(),cfg,arrays,plan,torch.device("cpu"),
                              _ConstantDensity(),100.,sample_per_time=1,batch_size=16,seed=7)
    assert [r["CO2_mass_true"] for r in rows] == pytest.approx([100.,200.])

def test_assembly_with_synthetic_twenty_task_fixture(tmp_path, monkeypatch):
    # Synthetic numbers solely for integration testing, never research results.
    import pandas as pd
    tasks = all_formal_tasks()
    grids = {}
    X, Y = np.meshgrid(np.arange(3.), np.arange(3.))
    for task in tasks:
        out = tmp_path / "by_run" / task.role / task.run_id
        paths = []
        for tag in ("t0", "early", "middle", "late", "final"):
            d = out / "snapshots" / task.role / tag
            d.mkdir(parents=True)
            p = d / "snapshot_grid.npz"
            np.savez(p, X=X, Y=Y, S_true=X*.1, S_pred=X*.09, S_error=-X*.01,
                     p_true=X+1e7, p_pred=X+1e7+1, p_error=np.ones_like(X))
            paths.append(p)
        grids[task.run_id] = paths
        pd.DataFrame([{"S_rmse": float(task.seed+1)}]).to_csv(out / "summary_test.csv", index=False)
        pd.DataFrame([{"x": x, "kind": "adam", "loss": float(task.seed+1)}
                      for x in (200,400)]).to_csv(out / "loss_points.csv", index=False)
    def fake_aggregate(_args):
        rows = []
        for task in tasks:
            rows.append(dict(model_role=task.role, seed=task.seed,
                **{m:float(task.seed+1+(task.role != "complete")) for m in assembly.CANONICAL_METRICS}))
        pd.DataFrame(rows).to_csv(tmp_path / "heterogeneous_metrics_by_seed.csv", index=False)
    monkeypatch.setattr(assembly, "verify", lambda _root: grids)
    monkeypatch.setattr(assembly, "aggregate_formal", fake_aggregate)
    monkeypatch.setattr(assembly, "save", lambda fig, _base: assembly.plt.close(fig))
    monkeypatch.setattr(assembly.plt.Figure, "colorbar", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(sys, "argv", ["assemble", "--evaluation-root", str(tmp_path)])
    assembly.main()
    paired = pd.read_csv(tmp_path / "comparison/paired_ablation_by_seed.csv")
    assert len(paired) == 80
    assert (paired.deterioration_absolute == 1).all()
    summary = pd.read_csv(tmp_path / "comparison/summary_test_multiseed_summary.csv")
    assert len(summary) == 5
    assert (summary.S_rmse_count == 4).all()
    loss = pd.read_csv(tmp_path / "comparison/loss_points_multiseed_summary.csv")
    assert len(loss) == 10
    assert (loss.loss_count == 4).all()
    assert json.loads((tmp_path / "supplement_aggregation_summary.json").read_text())["status"] == "completed"
