import copy

import numpy as np
from test_state_conditioned_arm_forward import sample

from s2a.history_conditioned_arm_forward import design as linear_design
from s2a.nonlinear_conditioned_arm_forward import design, predict, with_history


def setup():
    p = dict(
        dt=0.05,
        kp=np.ones(6) * 150,
        kd=np.ones(6) * 2 * np.sqrt(150),
        state_mean=np.zeros(83),
        state_scale=np.ones(83),
        translation_mean=np.zeros(3),
        translation_scale=np.ones(3) * 0.05,
        frequencies=np.random.default_rng(42).normal(size=(64, 83)) / np.sqrt(83),
        coefficients=np.random.default_rng(42).normal(size=(848, 12)) * 0.001,
    )
    return with_history(sample(), p), p


def test_original_linear_subspace_is_exactly_preserved():
    data, p = setup()
    columns = np.concatenate([np.arange(j * 212, j * 212 + 84) for j in range(4)])
    np.testing.assert_array_equal(design(data, p)[:, columns], linear_design(data, p))


def test_translation_affinity_preserves_explicit_history():
    a, p = setup()
    a = {k: v[1:2].copy() for k, v in a.items()}
    b = copy.deepcopy(a)
    a["action_goal_delta"][:, :3] = [-0.02, 0.01, 0.03]
    b["action_goal_delta"][:, :3] = [0.01, -0.01, -0.02]
    middle = copy.deepcopy(a)
    middle["action_goal_delta"][:, :3] = 0.4 * a["action_goal_delta"][:, :3] + 0.6 * b["action_goal_delta"][:, :3]
    np.testing.assert_allclose(predict(middle, p), 0.4 * predict(a, p) + 0.6 * predict(b, p), rtol=0, atol=1e-12)
    assert np.linalg.norm(middle["x_response_residual"]) > 0


def test_future_labels_cannot_affect_predictions_and_state_basis_is_nonlinear():
    data, p = setup()
    before = predict(data, p)
    changed = copy.deepcopy(data)
    changed["y_motion"] = np.full((2, 3), np.nan)
    changed["y_twist"] = np.full((2, 6), 1e9)
    np.testing.assert_array_equal(predict(changed, p), before)
    low = {k: v[:1].copy() for k, v in data.items()}
    high, mid = copy.deepcopy(low), copy.deepcopy(low)
    high["x_response_residual"][0, 0] += 2
    mid["x_response_residual"][0, 0] += 1
    assert np.linalg.norm(design(low, p)[:, 84:212] + design(high, p)[:, 84:212] - 2 * design(mid, p)[:, 84:212]) > 1e-4
