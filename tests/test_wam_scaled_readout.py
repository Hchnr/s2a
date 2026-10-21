"""027 train-only normalization, one synthetic solve, and actual Torch paths."""

import numpy as np
import pytest

from s2a.wam_flow_readout import FlowReadoutHook, FlowResidualReadout
from s2a.wam_readout import ACTIVE_MASK, LIBERO_DOMAIN, DomainContext
from s2a.wam_scaled_readout import ScaledFlowReadout, training_normalization


def test_global_vector_l2_rms_not_column_whitening_or_extra_sqrt_hidden_width():
    hidden = np.asarray([[-1.0, -1e-6], [1.0, 1e-6]])
    weights = np.asarray([0.5, 0.5])
    center, scale = training_normalization(hidden, weights)
    normalized = (hidden - center) / scale
    np.testing.assert_array_equal(center, [0, 0])
    assert np.isclose(scale, np.sqrt(1 + 1e-12), atol=0, rtol=1e-15)
    assert np.isclose(np.sum(weights * np.sum(normalized**2, axis=1)), 1, atol=1e-15)
    assert normalized[:, 1].std() / normalized[:, 0].std() == 1e-6
    with pytest.raises(ValueError, match="positive scale"):
        training_normalization(np.ones((2, 3)), weights)


def test_one_fixed_ridge_solve_matches_independent_augmented_ls_and_uses_only_training(monkeypatch):
    rng = np.random.default_rng(42)
    hidden, target = rng.normal(size=(12, 4)), rng.normal(size=(12, 10))
    original_hidden, original_target = hidden.copy(), target.copy()
    weights = np.arange(1, 13, dtype=np.float64)
    weights /= weights.sum()
    sigma = np.tile(np.asarray([0.2, 0.5, 0.8], np.float32), 4)
    solve, calls = np.linalg.solve, []

    def observed(a, b):
        calls.append((a.shape, b.shape))
        return solve(a, b)

    monkeypatch.setattr(np.linalg, "solve", observed)
    model = ScaledFlowReadout.fit(hidden, sigma, target, weights)
    assert len(calls) == 1
    mean = np.sum(weights[:, None] * original_hidden, axis=0)
    scale = np.sqrt(np.sum(weights[:, None] * (original_hidden - mean) ** 2))
    design = -np.column_stack(((original_hidden - mean) / scale, np.ones(12))) / sigma.astype(np.float64)[:, None]
    expected = np.linalg.lstsq(
        np.vstack((np.sqrt(weights)[:, None] * design, np.sqrt(0.1) * np.eye(5))),
        np.vstack((np.sqrt(weights)[:, None] * original_target, np.zeros((5, 10)))),
        rcond=None,
    )[0]
    np.testing.assert_allclose(model.center, mean, atol=1e-15, rtol=1e-15)
    np.testing.assert_allclose(model.scale, scale, atol=1e-15, rtol=1e-15)
    np.testing.assert_allclose(model.weights, expected, atol=1e-14, rtol=1e-13)
    np.testing.assert_array_equal(hidden, original_hidden)
    np.testing.assert_array_equal(target, original_target)
    # Deployment's very different validation inputs never refit preprocessing.
    saved = model.center.copy(), model.scale
    torch = pytest.importorskip("torch")
    validation = torch.full((5, 4), 1e9, dtype=torch.float64)
    model.apply_torch(
        validation,
        torch.zeros(5, 128, dtype=torch.float64),
        DomainContext(LIBERO_DOMAIN, 31, 0, ACTIVE_MASK),
        torch.zeros(5, dtype=torch.long),
        0.5,
    )
    np.testing.assert_array_equal(model.center, saved[0])
    assert model.scale == saved[1]
    assert len(calls) == 1
    with pytest.raises(ValueError, match="control refit is forbidden"):
        model.fit_bias(hidden, sigma, target, weights)


def test_actual_torch_zero_domain_bypass_and_mathematical_single_bfloat16_cast():
    torch = pytest.importorskip("torch")
    context = DomainContext(LIBERO_DOMAIN, 31, 0, ACTIVE_MASK)
    hidden, base = torch.zeros(1, 2, dtype=torch.bfloat16), torch.ones(1, 128, dtype=torch.bfloat16)
    base[:, 10] = -0.0
    ids = torch.zeros(1, dtype=torch.long)
    coefficients = np.zeros((3, 10))
    coefficients[-1] = -0.5 * (2**-8 + 2**-30)
    model = ScaledFlowReadout(coefficients, np.asarray([0.5, -0.25]), 2.0)
    result = model.apply_torch(hidden, base, context, ids, np.float32(0.5))
    assert torch.equal(result[:, :10], torch.full((1, 10), 1 + 2**-7, dtype=torch.bfloat16))
    assert torch.equal(result[:, 10:].view(torch.int16), base[:, 10:].view(torch.int16))
    assert (
        ScaledFlowReadout.zero(2, center=model.center, scale=model.scale).apply_torch(hidden, base, context, ids, 0.5)
        is base
    )
    other = DomainContext("other/full/domain", 2, 0, ACTIVE_MASK)
    assert model.apply_torch(hidden, base, other, ids, 0.5) is base


def test_actual_torch_normalization_order_and_original_hook_compatibility():
    torch = pytest.importorskip("torch")
    coefficients = np.zeros((1281, 10))
    coefficients[:2, :2] = [[0.125, -0.25], [0.5, 0.25]]
    center = np.zeros(1280)
    center[:2] = [0.5, -0.25]
    model = ScaledFlowReadout(coefficients, center, 2.0)
    assert isinstance(model, FlowResidualReadout)
    context = DomainContext(LIBERO_DOMAIN, 31, 0, ACTIVE_MASK)
    hidden = torch.ones(2, 1280, dtype=torch.float64)
    base = torch.zeros(2, 128, dtype=torch.float64)
    ids = torch.zeros(2, dtype=torch.long)
    expected = (
        -np.column_stack(((hidden.numpy() - center) / 2, np.ones(2))) / np.float64(np.float32(0.2)) @ coefficients
    )

    class Head(torch.nn.Module):
        def forward(self, hidden, projection_ids):
            return base.clone()

    head = Head()
    hook = FlowReadoutHook(head, model)
    with hook.request(context), hook._velocity_step(float(np.float32(0.2)), {}):
        output = head(hidden, ids)
    np.testing.assert_array_equal(output.numpy()[:, :10], expected)
    assert torch.count_nonzero(output[:, 10:]) == 0
    hook.remove()


def test_preprocessing_parameters_are_copied_and_bad_scale_rejected():
    coefficients, center = np.zeros((3, 10)), np.asarray([1.0, 2.0])
    model = ScaledFlowReadout(coefficients, center, 1)
    center[0] = 20
    coefficients[0, 0] = 20
    np.testing.assert_array_equal(model.center, [1, 2])
    assert model.is_zero and not model.center.flags.writeable
    for scale in (0, -1, np.inf, np.nan):
        with pytest.raises(ValueError, match="positive scalar"):
            ScaledFlowReadout(coefficients, center, scale)
