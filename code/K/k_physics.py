"""Shared permeability-aware Darcy helpers for Configuration K."""

from __future__ import annotations

import torch


def permeability_tilde(permeability_field, x_t: torch.Tensor, y_t: torch.Tensor, k_ref: float) -> torch.Tensor:
    reference = float(k_ref)
    if not reference > 0.0:
        raise ValueError(f"k_ref must be positive; got {k_ref!r}.")
    # The immutable field is exhaustively validated once by asset preflight.
    # Re-running reductions followed by Python truth conversion here would
    # synchronize the GPU twice for every phase/flux evaluation. Periodic
    # training loss/gradient/parameter health checks remain fail-closed.
    permeability = permeability_field(x_t, y_t)
    return permeability / reference


def darcy_velocity_components(
    permeability_field,
    x_t: torch.Tensor,
    y_t: torch.Tensor,
    *,
    relative_permeability: torch.Tensor,
    dpdx: torch.Tensor,
    dpdy: torch.Tensor,
    viscosity_ratio: float,
    k_ref: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    k_tilde = permeability_tilde(permeability_field, x_t, y_t, k_ref)
    coefficient = -k_tilde * float(viscosity_ratio) * relative_permeability
    return coefficient * dpdx, coefficient * dpdy
