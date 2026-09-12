import numpy as np
import pytest
import torch

from k_physics import darcy_velocity_components, permeability_tilde
from permeability_field_k import PermeabilityFieldK


def _field():
    return PermeabilityFieldK.from_arrays(
        np.array([0.0, 5.0]),
        np.array([0.0, 5.0]),
        np.array([[1e-14, 2e-14], [4e-14, 8e-14]]),
    )


def test_permeability_scale_changes_with_position():
    field = _field()
    x = torch.tensor([[0.0], [1.0]], dtype=torch.float64, requires_grad=True)
    y = torch.tensor([[0.0], [1.0]], dtype=torch.float64, requires_grad=True)

    scaled = permeability_tilde(field, x, y, k_ref=1e-14)

    assert scaled[:, 0].tolist() == pytest.approx([1.0, 8.0], rel=1e-12)


def test_darcy_velocity_uses_local_k_and_remains_differentiable():
    field = _field()
    x = torch.tensor([[0.25], [0.75]], dtype=torch.float64, requires_grad=True)
    y = torch.tensor([[0.25], [0.75]], dtype=torch.float64, requires_grad=True)
    rel_perm = torch.ones_like(x)
    dpdx = torch.full_like(x, 2.0)
    dpdy = torch.full_like(y, -3.0)

    vx, vy = darcy_velocity_components(
        field,
        x,
        y,
        relative_permeability=rel_perm,
        dpdx=dpdx,
        dpdy=dpdy,
        viscosity_ratio=4.0,
        k_ref=1e-14,
    )
    gradient = torch.autograd.grad(vx.sum(), x, create_graph=True)[0]

    assert torch.all(vx < 0)
    assert torch.all(vy > 0)
    assert not torch.allclose(vx[0], vx[1])
    assert torch.isfinite(gradient).all()
    assert torch.any(gradient.abs() > 0)


def test_permeability_helper_does_not_force_a_host_sync_on_every_gpu_call(monkeypatch):
    field = _field()
    x = torch.tensor([[0.5]], dtype=torch.float64)
    y = torch.tensor([[0.5]], dtype=torch.float64)

    def forbidden_runtime_scan(*_args, **_kwargs):
        raise AssertionError("validated immutable K must not be rescanned in every flux call")

    monkeypatch.setattr(torch, "isfinite", forbidden_runtime_scan)
    scaled = permeability_tilde(field, x, y, k_ref=1e-14)

    assert scaled.item() > 0.0
