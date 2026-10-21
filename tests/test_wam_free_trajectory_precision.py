"""Observe the original conversion's actual product, including BF16 promotion."""

import importlib
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("probe_wam_free_trajectory_precision")


@pytest.mark.parametrize("dtype", ["bfloat16", "float32"])
@pytest.mark.parametrize("payload", ["signed_normal", "boundaries_subnormal"])
def test_observer_records_real_multiply_once_and_preserves_conversion(module, dtype, payload):
    import torch

    class Scheduler:
        def __init__(self):
            self.calls = 0

        def convert_model_output(self, model_output, *, sample):
            self.calls += 1
            self.result = sample - torch.tensor(0.8077949285507202, dtype=torch.float32) * model_output
            return self.result

    s, v = module.precision_payload(payload)
    sample, velocity = torch.tensor(s), torch.tensor(v).to(getattr(torch, dtype))
    before = sample.clone(), velocity.clone()
    scheduler = Scheduler()
    result, (product, product_dtype) = module.observe_conversion(scheduler, sample, velocity)
    assert result is scheduler.result and scheduler.calls == 1
    assert product_dtype == f"torch.{dtype}" and product.dtype == np.float32
    expected = (sample - torch.from_numpy(product)).numpy()
    np.testing.assert_array_equal(result.numpy().view(np.uint32), expected.view(np.uint32))
    assert torch.equal(before[0], sample) and torch.equal(before[1], velocity)


def test_observer_rejects_a_repeated_product_in_real_method(module):
    import torch

    class Repeated:
        def convert_model_output(self, model_output, *, sample):
            return sample - 0.5 * model_output - 0.25 * model_output

    with pytest.raises(ValueError, match="one observed"):
        module.observe_conversion(Repeated(), torch.zeros(2, 128), torch.ones(2, 128))


def test_cpu_scalar_promotion_quantizes_sigma_before_bfloat16_product(module):
    import torch
    from verify_wam_readout import quantize_bfloat16

    _, value = module.precision_payload("signed_normal")
    velocity = torch.tensor(value).to(torch.bfloat16)
    sigma = np.float32(0.8077949285507202)
    actual = (torch.tensor(sigma) * velocity).float().numpy()
    v = velocity.float().numpy().astype(np.float64)
    sigma_bf16 = float(quantize_bfloat16(np.asarray(sigma, np.float64)))
    cpu_gold = quantize_bfloat16((v * sigma_bf16).astype(np.float32))
    different_order = quantize_bfloat16((v * float(sigma)).astype(np.float32))
    np.testing.assert_array_equal(actual.view(np.uint32), cpu_gold.view(np.uint32))
    assert np.max(np.abs(actual - different_order)) == 0.015625
