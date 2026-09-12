from pathlib import Path
import sys

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "Results"
if str(RESULTS) not in sys.path:
    sys.path.insert(0, str(RESULTS))

import evaluate_front_plume_M0_M7_L6_dS as front  # noqa: E402
import evaluate_physics_M0_M7_L6_dS as physics  # noqa: E402


class _ZeroSaturationModel(torch.nn.Module):
    def forward(self, x, y, t):
        return torch.zeros_like(x), torch.zeros_like(x)


class _ConstantDensity(torch.nn.Module):
    def forward(self, pressure):
        return torch.full_like(pressure, 100.0)


def _single_layer_mass_fixture():
    # Four 2x2 cell centres, with the high-saturation point duplicated.  A
    # native-row mean therefore gives the wrong spatial weight to that cell.
    x = np.array([1.25, 1.25, 3.75, 1.25, 3.75], dtype=np.float64)
    y = np.array([1.25, 1.25, 1.25, 3.75, 3.75], dtype=np.float64)
    saturation = np.array([0.8, 0.8, 0.0, 0.0, 0.0], dtype=np.float64)
    pressure = np.full(x.size, 10.0, dtype=np.float64)
    arrays = {
        "x": x,
        "y": y,
        "t": np.zeros(x.size, dtype=np.float64),
        "p": pressure,
        "Sco2": saturation,
        # Canonical true mass must use the same checkpoint EOS as predicted
        # mass; this deliberately incompatible raw density is bias telemetry.
        "rho_c": np.full(x.size, 1000.0, dtype=np.float64),
    }
    # Deliberately expose only a zero-saturation sampled point.  The mass
    # evaluator must use the complete native layer, not this sampled plan.
    eval_plan = {
        "x": x,
        "y": y,
        "t": arrays["t"],
        "p": pressure,
        "s": saturation,
        "indices_per_time": [np.array([2], dtype=np.int64)],
        "time_seconds": np.array([0.0], dtype=np.float64),
    }
    cfg = {
        "L_ref": 5.0,
        "T_ref": 31544.99609375,
        "p0": 0.0,
        "P_ref": 1.0,
        "phi": 0.2,
        "rho_w_const": 1027.61,
        "r_well": 0.5,
        "inj_center": (0.0, 0.0),
        "out_center": (5.0, 5.0),
        "dtype": "float64",
        "mass_grid_nx": 2,
        "mass_grid_ny": 2,
    }
    return arrays, eval_plan, cfg


def test_mass_uses_full_unique_layer_and_same_eos_on_fixed_grid():
    arrays, eval_plan, cfg = _single_layer_mass_fixture()

    rows = physics.evaluate_mass(
        _ZeroSaturationModel(),
        cfg,
        arrays,
        eval_plan,
        torch.device("cpu"),
        _ConstantDensity(),
        100.0,
        sample_per_time=1,
        batch_size=16,
        seed=7,
    )

    assert len(rows) == 1
    row = rows[0]
    # One of four equal 6.25 m^2 cells has S=0.8:
    # 6.25 * phi(0.2) * rho(100) * S(0.8) = 100 kg.
    assert row["CO2_mass_true"] == pytest.approx(100.0)
    assert row["CO2_mass_true_raw_density"] == pytest.approx(1000.0)
    assert row["mass_quadrature_unique_native_points"] == 4
    assert row["mass_quadrature_valid_cells"] == 4
    assert row["mass_quadrature_cell_area_m2"] == pytest.approx(6.25)
    assert row["mass_density_semantics"] == "checkpoint_eos_for_true_and_predicted_pressure"


def test_mass_adaptive_layers_rebuild_mapping():
    arrays, plan, cfg = _single_layer_mass_fixture()
    # Reordered support and changed duplicate multiplicities.
    idx = np.array([4, 2, 0, 3, 0, 0])
    for key, values in list(arrays.items()):
        extra = values[idx]
        if key == "t":
            extra = np.ones(idx.size) * 10.0
        arrays[key] = np.concatenate([values, extra])
    plan["time_seconds"] = np.array([0.0, 10.0])
    rows = physics.evaluate_mass(
        _ZeroSaturationModel(), cfg, arrays, plan, torch.device("cpu"),
        _ConstantDensity(), 100.0, sample_per_time=1, batch_size=16, seed=7,
    )
    assert [row["CO2_mass_true"] for row in rows] == pytest.approx([100.0, 100.0])
    assert [row["mass_quadrature_unique_native_points"] for row in rows] == [4, 4]


def test_mass_fixed_cell_quadrature_excludes_both_quarter_wells():
    arrays, eval_plan, cfg = _single_layer_mass_fixture()
    cfg["mass_grid_nx"] = 10
    cfg["mass_grid_ny"] = 10

    row = physics.evaluate_mass(
        _ZeroSaturationModel(),
        cfg,
        arrays,
        eval_plan,
        torch.device("cpu"),
        _ConstantDensity(),
        100.0,
        sample_per_time=1,
        batch_size=32,
        seed=7,
    )[0]

    # On the 0.5 m cell-centred grid, (0.25,0.25) and (4.75,4.75)
    # lie inside the radius-0.5 quarter wells and are excluded.
    assert row["mass_quadrature_valid_cells"] == 98
    assert row["mass_quadrature_effective_area_m2"] == pytest.approx(24.5)


def test_selected_contour_layer_uses_all_rows_and_collapses_duplicates():
    x_one = np.array([0.0, 0.0, 5.0, 0.0, 5.0], dtype=np.float64)
    y_one = np.array([0.0, 0.0, 0.0, 5.0, 5.0], dtype=np.float64)
    arrays = {
        "x": np.tile(x_one, 2),
        "y": np.tile(y_one, 2),
        "t": np.repeat(np.array([0.0, 10.0]), x_one.size),
        "Sco2": np.array(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            dtype=np.float64,
        ),
    }

    layers = front.build_full_reference_layers(
        arrays,
        np.array([0.0, 10.0], dtype=np.float64),
        selected_indices={1},
    )

    x_unique, y_unique, s_unique = layers[1]
    assert x_unique.size == y_unique.size == s_unique.size == 4
    values = {
        (float(x), float(y)): float(s)
        for x, y, s in zip(x_unique, y_unique, s_unique)
    }
    assert values[(0.0, 0.0)] == pytest.approx(0.3)
    assert values[(5.0, 0.0)] == pytest.approx(0.6)
    assert values[(0.0, 5.0)] == pytest.approx(0.8)
    assert values[(5.0, 5.0)] == pytest.approx(1.0)


def test_contour_grid_mapping_does_not_resample_full_unique_support():
    x = np.array([0.0, 5.0, 0.0, 5.0, 2.5], dtype=np.float64)
    y = np.array([0.0, 0.0, 5.0, 5.0, 2.5], dtype=np.float64)
    values = np.array([0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    gx = np.array([0.0, 2.5, 5.0], dtype=np.float64)
    Xg, Yg = np.meshgrid(gx, gx)

    class _WouldDropCentre:
        def choice(self, *_args, **_kwargs):
            return np.array([0, 1, 2, 3], dtype=np.int64)

    mapped = front.interpolate_true_to_grid(
        x,
        y,
        values,
        Xg,
        Yg,
        max_points=4,
        rng=_WouldDropCentre(),
    )

    assert mapped[1, 1] == pytest.approx(1.0)


def test_bracketed_contour_without_segments_fails_closed(monkeypatch):
    class _EmptyContour:
        allsegs = [[]]

    class _Axes:
        def contour(self, *_args, **_kwargs):
            return _EmptyContour()

    monkeypatch.setattr(front.plt, "subplots", lambda **_kwargs: (object(), _Axes()))
    monkeypatch.setattr(front.plt, "close", lambda _fig: None)
    gx = np.linspace(0.0, 5.0, 8)
    Xg, Yg = np.meshgrid(gx, gx)

    with pytest.raises(RuntimeError, match="brackets.*no contour segments"):
        front.contour_points(Xg, Yg, Xg / 5.0, 0.175)


def test_canonical_contour_requires_an_informative_selected_time():
    rows = [
        {
            "run_id": "complete_seed0",
            "contour_missing_status_S0175": "both_missing",
        },
        {
            "run_id": "complete_seed0",
            "contour_missing_status_S0175": "both_missing",
        },
    ]

    with pytest.raises(ValueError, match="informative S=0.175 contour"):
        front.validate_canonical_contour_support(rows)

    front.validate_canonical_contour_support(
        rows
        + [
            {
                "run_id": "complete_seed0",
                "contour_missing_status_S0175": "present",
            }
        ]
    )


def test_front_grid_masks_both_well_holes_for_truth_and_prediction():
    centres = (np.arange(10, dtype=np.float64) + 0.5) * 0.5
    Xg, Yg = np.meshgrid(centres, centres)
    true_grid = np.ones_like(Xg)
    pred_grid = np.ones_like(Xg)
    cfg = {
        "r_well": 0.5,
        "inj_center": (0.0, 0.0),
        "out_center": (5.0, 5.0),
    }

    true_masked, pred_masked, valid = front.apply_effective_domain_mask(
        Xg, Yg, true_grid, pred_grid, cfg
    )

    assert int(np.sum(~valid)) == 2
    assert np.isnan(true_masked[0, 0]) and np.isnan(pred_masked[0, 0])
    assert np.isnan(true_masked[-1, -1]) and np.isnan(pred_masked[-1, -1])
    assert np.isfinite(true_masked[valid]).all()
    metrics = front.plume_metrics(
        pred_masked.reshape(-1), true_masked.reshape(-1), [0.5]
    )
    assert metrics["plume_area_true_S050"] == pytest.approx(1.0)
    assert metrics["plume_area_pred_S050"] == pytest.approx(1.0)
    excess = front.plume_metrics_excess(
        pred_masked.reshape(-1), true_masked.reshape(-1), 0.0, [0.5]
    )
    assert excess["plume_area_true_dS050"] == pytest.approx(1.0)
    assert excess["plume_area_pred_dS050"] == pytest.approx(1.0)
