import copy
import importlib
import importlib.util
from pathlib import Path
import re
import sys

import numpy as np
import pytest
import torch

from configuration_k import DATASET_SHA256


def _module():
    assert importlib.util.find_spec("eos_k") is not None, "shared frozen EOS module is missing"
    return importlib.import_module("eos_k")


def _synthetic_eos_samples():
    pressure = np.linspace(9.0e6, 12.0e6, 256, dtype=np.float64)
    centered = (pressure - 10.5e6) / 1.5e6
    density = 610.0 + 35.0 * centered + 4.0 * centered**2
    return pressure, density


def test_frozen_eos_record_is_independent_of_model_random_seed():
    eos = _module()
    pressure, density = _synthetic_eos_samples()

    np.random.seed(0)
    torch.manual_seed(0)
    seed_zero_record = eos.fit_eos_record(
        pressure,
        density,
        reference_dataset_sha256=DATASET_SHA256,
    )

    np.random.seed(42)
    torch.manual_seed(42)
    seed_42_record = eos.fit_eos_record(
        pressure,
        density,
        reference_dataset_sha256=DATASET_SHA256,
    )

    assert seed_zero_record == seed_42_record
    assert seed_zero_record["fit_protocol"]["sample_seed"] == 0
    assert re.fullmatch(r"[0-9a-f]{64}", seed_zero_record["record_sha256"])


def test_frozen_eos_record_rejects_coefficient_tampering():
    eos = _module()
    pressure, density = _synthetic_eos_samples()
    record = eos.fit_eos_record(
        pressure,
        density,
        reference_dataset_sha256=DATASET_SHA256,
    )
    tampered = copy.deepcopy(record)
    tampered["model"]["coefficients"][0] += 0.01

    with pytest.raises(ValueError, match="SHA-256"):
        eos.validate_eos_record(
            tampered,
            expected_dataset_sha256=DATASET_SHA256,
        )


def test_frozen_eos_record_rejects_unacceptable_fit_or_nonpositive_density():
    eos = _module()
    pressure, density = _synthetic_eos_samples()
    record = eos.fit_eos_record(
        pressure,
        density,
        reference_dataset_sha256=DATASET_SHA256,
    )

    poor_fit = copy.deepcopy(record)
    poor_fit.pop("record_sha256")
    poor_fit["fit_diagnostics"]["max_relative_error"] = 0.5
    poor_fit["fit_diagnostics"]["rms_relative_error"] = 0.4
    poor_fit = eos.seal_eos_record(poor_fit)
    with pytest.raises(ValueError, match="fit error"):
        eos.validate_eos_record(poor_fit, expected_dataset_sha256=DATASET_SHA256)

    nonpositive = copy.deepcopy(record)
    nonpositive.pop("record_sha256")
    nonpositive["model"]["coefficients"] = [
        -abs(value) if index == 0 else 0.0
        for index, value in enumerate(nonpositive["model"]["coefficients"])
    ]
    nonpositive = eos.seal_eos_record(nonpositive)
    with pytest.raises(ValueError, match="positive density"):
        eos.validate_eos_record(nonpositive, expected_dataset_sha256=DATASET_SHA256)


def test_torch_eos_model_is_built_only_from_the_validated_record():
    eos = _module()
    pressure, density = _synthetic_eos_samples()
    record = eos.fit_eos_record(
        pressure,
        density,
        reference_dataset_sha256=DATASET_SHA256,
    )
    model, rho_ref = eos.torch_eos_from_record(
        record,
        dtype=torch.float64,
        device=torch.device("cpu"),
        expected_dataset_sha256=DATASET_SHA256,
    )
    p = torch.tensor([[10.5e6]], dtype=torch.float64, requires_grad=True)

    rho = model(p)
    gradient = torch.autograd.grad(rho.sum(), p)[0]

    assert rho_ref == pytest.approx(record["model"]["rho_ref_kg_m3"])
    assert rho.item() == pytest.approx(rho_ref, rel=2.0e-3)
    assert torch.isfinite(gradient).all()
    assert torch.any(gradient.abs() > 0.0)


def test_physics_evaluator_reuses_checkpoint_eos_without_dataset_refitting():
    results_dir = Path(__file__).resolve().parents[1] / "Results"
    if str(results_dir) not in sys.path:
        sys.path.insert(0, str(results_dir))
    evaluator = importlib.import_module("evaluate_physics_M0_M7_L6_dS")
    eos = _module()
    pressure, density = _synthetic_eos_samples()
    record = eos.fit_eos_record(
        pressure,
        density,
        reference_dataset_sha256=DATASET_SHA256,
    )

    model, rho_ref = evaluator.eos_model_from_checkpoint(
        {"eos_record": record},
        dtype=torch.float64,
        device=torch.device("cpu"),
    )
    predicted = model(torch.tensor([[10.5e6]], dtype=torch.float64))

    assert rho_ref == pytest.approx(record["model"]["rho_ref_kg_m3"])
    assert torch.isfinite(predicted).all()
