"""025 synthetic fit and real Torch deployment boundaries; no model/data reads."""

import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from s2a.wam_readout import (
    ACTIVE_MASK,
    HIDDEN_SIZE,
    LIBERO_DOMAIN,
    RIDGE,
    DomainContext,
    ReadoutHook,
    ResidualReadout,
    features,
)


@pytest.fixture
def torch():
    return pytest.importorskip("torch")


@pytest.fixture
def libero():
    return DomainContext(LIBERO_DOMAIN, 31, 0, ACTIVE_MASK)


@pytest.mark.parametrize("rows,width", [(7, 3), (4, 9)])
def test_weighted_ridge_matches_independent_augmented_and_dual_solves(rows, width):
    rng = np.random.default_rng(42)
    hidden, target = rng.normal(size=(rows, width)), rng.normal(size=(rows, 10))
    weights = rng.uniform(size=rows)
    weights /= weights.sum()
    original = [a.copy() for a in (hidden, target, weights)]
    fitted = ResidualReadout.fit(hidden, target, weights)
    phi = np.concatenate((hidden / np.sqrt(width), np.ones((rows, 1))), axis=1)
    x, y = np.sqrt(weights)[:, None] * phi, np.sqrt(weights)[:, None] * target
    augmented = np.linalg.lstsq(
        np.concatenate((x, np.sqrt(RIDGE) * np.eye(width + 1))),
        np.concatenate((y, np.zeros((width + 1, 10)))),
        rcond=None,
    )[0]
    dual = x.T @ np.linalg.solve(x @ x.T + RIDGE * np.eye(rows), y)
    np.testing.assert_allclose(fitted.weights, augmented, rtol=1e-13, atol=1e-14)
    np.testing.assert_allclose(fitted.weights, dual, rtol=1e-13, atol=1e-14)
    for actual, before in zip((hidden, target, weights), original, strict=True):
        np.testing.assert_array_equal(actual, before)
    # The model owns its copied coefficients, not a mutable view of inputs.
    hidden[:] = 0
    target[:] = 0
    weights[:] = 0
    np.testing.assert_allclose(fitted.weights, augmented, rtol=1e-13, atol=1e-14)


def test_bias_is_regularized_and_uses_identical_weighted_targets():
    hidden = np.arange(12).reshape(4, 3)
    target = np.arange(40).reshape(4, 10)
    weights = np.asarray([0.1, 0.2, 0.3, 0.4])
    fitted = ResidualReadout.fit_bias(hidden, target, weights)
    np.testing.assert_array_equal(fitted.weights[:-1], np.zeros((3, 10)))
    np.testing.assert_allclose(fitted.weights[-1], (weights @ target) / 1.1, rtol=1e-15)
    with pytest.raises(ValueError, match="summing to one"):
        ResidualReadout.fit(hidden, target, weights * 2)
    with pytest.raises(ValueError, match="positive weights"):
        ResidualReadout.fit(hidden, target, [0, 0.3, 0.3, 0.4])


def test_features_and_coefficients_validate_shape_finiteness_and_input_isolation():
    hidden = np.asarray([[2.0, 4.0]])
    phi = features(hidden)
    np.testing.assert_array_equal(phi[0, :2], hidden[0] / np.sqrt(2))
    assert phi[0, 2] == 1 and phi.dtype == np.float64
    coefficients = np.zeros((3, 10))
    model = ResidualReadout(coefficients)
    coefficients[0, 0] = 5
    assert model.is_zero
    assert not model.weights.flags.writeable
    for invalid in ([], np.zeros(2), np.asarray([[np.nan]])):
        with pytest.raises(ValueError, match="finite nonempty"):
            features(invalid)


@pytest.mark.parametrize(
    "domain,norm,projection,mask",
    [
        (LIBERO_DOMAIN + "_wrong", 31, 0, ACTIVE_MASK),
        (LIBERO_DOMAIN, 2, 0, ACTIVE_MASK),
        (LIBERO_DOMAIN, 31, 1, ACTIVE_MASK),
        (LIBERO_DOMAIN, 31, 0, (True,) * 128),
    ],
)
def test_full_domain_gate_rejects_every_inconsistent_identity(domain, norm, projection, mask):
    with pytest.raises(ValueError, match="must agree"):
        DomainContext(domain, norm, projection, mask)


def test_zero_and_shared_projection_other_domain_return_original_object(torch, libero):
    hidden = torch.arange(12, dtype=torch.bfloat16).reshape(4, 3)
    base = torch.arange(512, dtype=torch.bfloat16).reshape(4, 128)
    ids = torch.zeros(4, dtype=torch.long)
    zero = ResidualReadout.zero(3)
    assert zero.apply_torch(hidden, base, libero, ids) is base
    nonzero = ResidualReadout(np.ones((4, 10)))
    other = DomainContext("another/full/domain", 2, 0, ACTIVE_MASK)
    assert nonzero.apply_torch(hidden, base, other, ids) is base
    with pytest.raises(ValueError, match="mixed or inconsistent"):
        nonzero.apply_torch(hidden, base, libero, torch.tensor([0, 0, 1, 0]))


def test_real_bfloat16_ties_and_inactive_slots_preserve_bytes(torch, libero):
    hidden = torch.tensor([[-1.0, 0.0], [1.0, 0.0]], dtype=torch.bfloat16)
    base = torch.ones(2, 128, dtype=torch.bfloat16)
    base[:, 10:] = torch.arange(118, dtype=torch.bfloat16)
    original_hidden, original_base = hidden.clone(), base.clone()
    weights = np.zeros((3, 10))
    weights[0, :] = np.sqrt(2) * 2**-8
    weights[-1, :] = 2 * 2**-8
    adapted = ResidualReadout(weights).apply_torch(hidden, base, libero, torch.zeros(2, dtype=torch.long))
    expected = torch.tensor([[1.0] * 10, [1.015625] * 10], dtype=torch.bfloat16)
    assert torch.equal(adapted[:, :10].view(torch.int16), expected.view(torch.int16))
    assert torch.equal(adapted[:, 10:].view(torch.int16), original_base[:, 10:].view(torch.int16))
    assert torch.equal(base.view(torch.int16), original_base.view(torch.int16))
    assert torch.equal(hidden.view(torch.int16), original_hidden.view(torch.int16))
    assert adapted is not base and adapted.dtype == base.dtype


def test_fp64_addition_casts_once_and_rejects_nonfinite_quantized_result(torch, libero):
    hidden = torch.zeros((1, 2), dtype=torch.float32)
    base = torch.zeros((1, 128), dtype=torch.float32)
    weights = np.zeros((3, 10))
    weights[-1] = np.linspace(-0.1, 0.1, 10)
    out = ResidualReadout(weights).apply_torch(hidden, base, libero, torch.zeros(1, dtype=torch.long))
    np.testing.assert_array_equal(out[0, :10].numpy(), weights[-1].astype(np.float32))
    weights[-1] = 1e100
    with pytest.raises(ValueError, match="nonfinite output"):
        ResidualReadout(weights).apply_torch(hidden, base, libero, torch.zeros(1, dtype=torch.long))


def test_request_hook_uses_actual_head_hidden_and_restores_context_on_exception(torch, libero):
    class Head(torch.nn.Module):
        def forward(self, hidden, projection_ids):
            return hidden[:, :128] * 2

    head = Head().eval()
    weights = np.zeros((HIDDEN_SIZE + 1, 10))
    weights[-1] = 0.25
    hidden = torch.zeros((2, HIDDEN_SIZE), dtype=torch.bfloat16)
    ids = torch.zeros(2, dtype=torch.long)
    before_parameters = tuple(head.parameters())
    hook = ReadoutHook(head, ResidualReadout(weights))
    with pytest.raises(ValueError, match="outside its request"):
        head(hidden, ids)
    try:
        with hook.request(libero):
            actual = head(hidden, ids)
            assert torch.equal(actual[:, :10], torch.full((2, 10), 0.25, dtype=torch.bfloat16))
            assert torch.count_nonzero(actual[:, 10:]) == 0
            with pytest.raises(ValueError, match="non-nested"):
                with hook.request(libero):
                    pass
            raise RuntimeError("request failed")
    except RuntimeError:
        pass
    with pytest.raises(ValueError, match="outside its request"):
        head(hidden, ids)
    assert tuple(head.parameters()) == before_parameters
    hook.remove()
    assert torch.count_nonzero(head(hidden, ids)) == 0
    with pytest.raises(ValueError, match="width1280"):
        ReadoutHook(head, ResidualReadout.zero(3))


def test_request_context_isolated_between_real_cpu_threads(torch, libero):
    class Head(torch.nn.Module):
        def forward(self, hidden, projection_ids):
            return torch.zeros((len(hidden), 128))

    weights = np.zeros((HIDDEN_SIZE + 1, 10))
    weights[-1] = 1
    head = Head()
    hook = ReadoutHook(head, ResidualReadout(weights))
    other = DomainContext("another/full/domain", 2, 0, ACTIVE_MASK)

    def evaluate(context):
        with hook.request(context):
            return float(head(torch.zeros((1, HIDDEN_SIZE)), torch.zeros(1, dtype=torch.long)).sum())

    with ThreadPoolExecutor(2) as executor:
        results = list(executor.map(evaluate, [libero, other] * 4))
    assert results == [10, 0] * 4
    hook.remove()


def test_numpy_algorithm_does_not_require_a_global_torch_import():
    # Inspect only the imported module's own namespace, not other tests' imports.
    assert "torch" not in vars(sys.modules[ResidualReadout.__module__])
