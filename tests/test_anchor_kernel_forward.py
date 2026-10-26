import numpy as np
import pytest

from s2a.anchor_kernel_forward import BLOCKS, condition_features


def test_scalar_group_distance_parity_and_training_center_identity():
    rng = np.random.default_rng(42)
    centers = rng.normal(size=(66, 89))
    parameters = dict(state_mean=rng.normal(size=89), state_scale=rng.uniform(0.5, 2, size=89), anchor_centers=centers)
    raw = centers[:2].copy()
    phi = condition_features(raw, parameters)
    assert phi.shape == (2, 156)
    np.testing.assert_array_equal(phi[:, 0], 1)
    np.testing.assert_allclose(phi[[0, 1], [90, 91]], 1, atol=1e-14)
    for i in range(2):
        for j in range(66):
            distance = 0.0
            for block in BLOCKS:
                distance += sum(((raw[i, k] - centers[j, k]) / parameters["state_scale"][k]) ** 2 for k in block) / len(
                    block
                )
            expected = (1 + distance / 7) ** -2
            np.testing.assert_allclose(phi[i, 90 + j], expected, rtol=1e-14)
    # Duplicate q sine/cosine fields remain explicit linear inputs but do not alter distance.
    changed = raw.copy()
    changed[:, 7:21] += 100
    np.testing.assert_array_equal(condition_features(changed, parameters)[:, 90:], phi[:, 90:])
    with pytest.raises(ValueError):
        condition_features(raw, dict(parameters, state_scale=np.zeros(89)))


def test_action_affinity_with_fixed_measured_condition(monkeypatch):
    import s2a.anchor_kernel_forward as model

    rng = np.random.default_rng(43)
    raw = rng.normal(size=(2, 89))
    monkeypatch.setattr(model, "raw_condition", lambda data: raw)
    p = dict(
        state_mean=np.zeros(89),
        state_scale=np.ones(89),
        anchor_centers=rng.normal(size=(66, 89)),
        translation_mean=np.array([0.01, -0.02, 0.03]),
        translation_scale=np.array([0.03, 0.04, 0.05]),
    )
    a, b = rng.normal(size=(2, 6)), rng.normal(size=(2, 6))
    x = model.design(dict(action_goal_delta=a), p)
    y = model.design(dict(action_goal_delta=b), p)
    mid = model.design(dict(action_goal_delta=0.3 * a + 0.7 * b), p)
    assert x.shape == (2, 624)
    np.testing.assert_allclose(mid, 0.3 * x + 0.7 * y, rtol=1e-12, atol=1e-12)
