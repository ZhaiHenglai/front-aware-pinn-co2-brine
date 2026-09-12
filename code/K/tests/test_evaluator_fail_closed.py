from pathlib import Path
import sys

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "Results"
if str(RESULTS) not in sys.path:
    sys.path.insert(0, str(RESULTS))

from evaluate_front_plume_M0_M7_L6_dS import (  # noqa: E402
    contour_points,
    contour_distance_metrics,
    error_metrics,
    regular_grid_from_points,
)
from evaluate_global_accuracy_M0_M7_L6_dS_shortlabels import (  # noqa: E402
    load_dataset_pack,
    predict_model,
)
from evaluate_physics_M0_M7_L6_dS import (  # noqa: E402
    _require_finite_array,
    metric_values,
    physics_cfg_update_from_ckpt,
    sample_fv_cells_np,
    sample_injection_arc_np,
    sample_outer_boundary_np,
    sample_outlet_arc_np,
)


class _NonfiniteModel(torch.nn.Module):
    def forward(self, x, y, t):
        value = torch.full_like(x, float("nan"))
        return value, value


def test_prediction_fails_closed_on_nonfinite_model_output():
    points = np.array([0.0, 1.0], dtype=np.float64)
    cfg = {"L_ref": 5.0, "T_ref": 31544.99609375, "dtype": "float64"}

    with pytest.raises(FloatingPointError, match="Nonfinite model prediction"):
        predict_model(
            _NonfiniteModel(), points, points, points, cfg, torch.device("cpu"), batch_size=2
        )


def test_dataset_loader_rejects_nonfinite_required_reference_values(tmp_path):
    path = tmp_path / "bad.pt"
    torch.save(
        {
            "arrays": {
                "x": np.array([0.0, np.nan]),
                "y": np.array([0.0, 1.0]),
                "t": np.array([0.0, 1.0]),
                "p": np.array([1.0, 2.0]),
                "Sco2": np.array([0.0, 0.1]),
            }
        },
        path,
    )

    with pytest.raises(ValueError, match="finite.*x|x.*finite"):
        load_dataset_pack(path)


def test_front_and_physics_metrics_reject_nonfinite_values():
    with pytest.raises(FloatingPointError, match="Nonfinite front metric"):
        error_metrics(np.array([0.1, np.nan]), np.array([0.1, 0.2]))

    with pytest.raises(FloatingPointError, match="Nonfinite physics metric"):
        metric_values(np.array([0.1, np.inf]), "test")

    with pytest.raises(FloatingPointError, match="CO2 density"):
        _require_finite_array(np.array([600.0, np.nan]), "CO2 density")


def test_contour_one_sided_absence_gets_domain_diagonal_penalty():
    gx = np.linspace(0.0, 5.0, 8)
    gy = np.linspace(0.0, 5.0, 8)
    xg, yg = np.meshgrid(gx, gy)
    true_grid = xg / 5.0
    pred_grid = np.zeros_like(true_grid)

    metrics = contour_distance_metrics(xg, yg, true_grid, pred_grid, 0.175)

    assert metrics["contour_missing_status_S0175"] == "predicted_missing"
    assert metrics["contour_chamfer_S0175"] == pytest.approx(np.sqrt(50.0))
    assert metrics["contour_hausdorff_S0175"] == pytest.approx(np.sqrt(50.0))
    assert metrics["contour_p95dist_S0175"] == pytest.approx(np.sqrt(50.0))


def test_contour_both_absent_is_explicitly_noninformative():
    gx = np.linspace(0.0, 5.0, 8)
    gy = np.linspace(0.0, 5.0, 8)
    xg, yg = np.meshgrid(gx, gy)
    metrics = contour_distance_metrics(
        xg, yg, np.zeros_like(xg), np.zeros_like(xg), 0.175
    )

    assert metrics["contour_missing_status_S0175"] == "both_missing"
    assert np.isnan(metrics["contour_chamfer_S0175"])


def test_contour_computation_errors_are_not_relabelled_as_missing(monkeypatch):
    class _BrokenAxes:
        def contour(self, *_args, **_kwargs):
            raise RuntimeError("synthetic contour failure")

    monkeypatch.setattr(
        sys.modules[contour_points.__module__].plt,
        "subplots",
        lambda **_kwargs: (object(), _BrokenAxes()),
    )
    monkeypatch.setattr(sys.modules[contour_points.__module__].plt, "close", lambda _fig: None)
    gx = np.linspace(0.0, 5.0, 8)
    gy = np.linspace(0.0, 5.0, 8)
    xg, yg = np.meshgrid(gx, gy)

    with pytest.raises(RuntimeError, match="synthetic contour failure"):
        contour_points(xg, yg, xg / 5.0, 0.175)


def test_contour_grid_uses_frozen_domain_not_sample_extrema():
    gx, gy, _, _ = regular_grid_from_points(
        np.array([1.0, 4.0]), np.array([2.0, 3.0]), nx=4, ny=5
    )

    assert gx[[0, -1]].tolist() == [0.0, 5.0]
    assert gy[[0, -1]].tolist() == [0.0, 5.0]


def test_physics_sampling_is_bounded_by_last_retained_reference_time():
    retained_max = 30986.943359375
    cfg = {
        "L_ref": 5.0,
        "T_ref": 31544.99609375,
        "U_ref": 5.4e-5,
        "p0": 1.0e7,
        "P_ref": 3.375e6,
        "r_well": 0.5,
        "inj_center": (0.0, 0.0),
        "out_center": (5.0, 5.0),
        "inj_bc_t_start_tilde": 0.0,
    }
    physics_cfg_update_from_ckpt(
        cfg,
        {
            "data_time_crop": {
                "first_retained_time_s": 0.0,
                "last_retained_time_s": retained_max,
            }
        },
    )
    rng = np.random.default_rng(7)

    for sampler, args in (
        (sample_injection_arc_np, (128, cfg, rng)),
        (sample_outlet_arc_np, (128, cfg, rng)),
        (sample_outer_boundary_np, (32, cfg, rng)),
    ):
        values = sampler(*args)
        time_s = values[2]
        assert np.min(time_s) >= 0.0
        assert np.max(time_s) <= retained_max

    _, _, t0, t1 = sample_fv_cells_np(128, cfg, 1.0 / 64.0, 0.001, rng)
    assert np.min(t0) >= 0.0
    assert np.max(t1) * cfg["T_ref"] <= retained_max


def test_fv_evaluation_cells_use_the_training_square_clearance_rule():
    cfg = {
        "L_ref": 5.0,
        "T_ref": 31544.99609375,
        "r_well": 0.5,
        "inj_center": (0.0, 0.0),
        "out_center": (5.0, 5.0),
        "evaluation_time_min_s": 0.0,
        "evaluation_time_max_s": 30986.943359375,
    }
    h_tilde = 1.0 / 64.0
    x_tilde, y_tilde, _, _ = sample_fv_cells_np(
        5000, cfg, h_tilde, 0.001, np.random.default_rng(19)
    )
    x = x_tilde * cfg["L_ref"]
    y = y_tilde * cfg["L_ref"]
    h_phys = h_tilde * cfg["L_ref"]
    half = 0.5 * h_phys
    well_clearance = cfg["r_well"] + 0.5 * np.sqrt(2.0) * h_phys

    assert np.all((x > half) & (x < cfg["L_ref"] - half))
    assert np.all((y > half) & (y < cfg["L_ref"] - half))
    assert np.all(np.hypot(x, y) > well_clearance)
    assert np.all(np.hypot(x - 5.0, y - 5.0) > well_clearance)
