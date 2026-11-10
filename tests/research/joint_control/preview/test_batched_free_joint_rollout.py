"""Batched free dynamics must match every sequential endpoint and derivative."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from s2a.research.joint_control.preview.batched_free_joint_rollout import BatchedFreeJointRollout
from s2a.research.joint_control.preview.interior_joint_preview import build_joint_preview, rollout_joint_preview
from s2a.research.response_models.forward.joint_limit_response import build_joint_response, predict_joint_response


def example(horizon, *, near_limit=False):
    rng = np.random.default_rng(42 + horizon)
    a = rng.normal(size=(7, 7))
    mass = np.eye(7) + a.T @ a * 0.04
    jacobian = np.c_[np.eye(6), rng.normal(size=6) * 0.1]
    q = rng.normal(size=7) * 0.1
    if near_limit:
        q[0], q[1] = 1.999999999, -1.999999999
    initial = rng.normal(size=7) * 0.1
    model = build_joint_preview(mass, jacobian, np.ones(6) * 100, np.ones(6) * 20, q, initial, null_kp=10, dt=0.002)
    velocity, deltas = rng.normal(size=7) * 0.05, rng.normal(size=(horizon, 6)) * 0.02
    site = np.eye(4)
    site[:3, :3] = Rotation.from_rotvec([0.4, -0.2, 0.3]).as_matrix()
    site[:3, 3] = [0.2, -0.1, 0.4]
    return model, velocity, deltas, site, -np.ones(7) * 2, np.ones(7) * 2


@pytest.mark.parametrize("horizon", [1, 4, 8, 16])
@pytest.mark.parametrize("near_limit", [False, True])
def test_all_free_rollout_outputs_match_sequential_model(horizon, near_limit):
    model, velocity, deltas, site, lower, upper = example(horizon, near_limit=near_limit)
    candidate = BatchedFreeJointRollout(model)
    original = rollout_joint_preview(model, velocity, deltas, site, lower, upper, enforce_limits=False)
    actual = candidate.rollout(velocity, deltas, site, lower, upper, enforce_limits=False)
    assert set(actual) == set(original)
    for key in original:
        np.testing.assert_allclose(actual[key], original[key], atol=1e-10, rtol=0, err_msg=key)
    if horizon > 1:
        np.testing.assert_array_equal(actual["joint_position_jacobian"][:-1, :, -6:], 0)


def test_candidate_copies_model_and_does_not_modify_inputs():
    model, velocity, deltas, site, lower, upper = example(8)
    candidate = BatchedFreeJointRollout(model)
    before = [x.copy() for x in (velocity, deltas, site, lower, upper)]
    expected = candidate.rollout(velocity, deltas, site, lower, upper, enforce_limits=False)
    model["q"][0] += 1
    model["transition"][0, 0] += 1
    actual = candidate.rollout(velocity, deltas, site, lower, upper, enforce_limits=False)
    for key in expected:
        np.testing.assert_array_equal(actual[key], expected[key])
    for item, saved in zip((velocity, deltas, site, lower, upper), before, strict=True):
        np.testing.assert_array_equal(item, saved)


def test_rejects_projected_mode_and_unsupported_rotation_latch():
    model, velocity, deltas, site, lower, upper = example(4)
    candidate = BatchedFreeJointRollout(model)
    with pytest.raises(ValueError, match="unprojected"):
        candidate.rollout(velocity, deltas, site, lower, upper, enforce_limits=True)
    deltas[1, 3:] = 0
    with pytest.raises(ValueError, match="rotation"):
        candidate.rollout(velocity, deltas, site, lower, upper, enforce_limits=False)


def test_independent_scalar_transition_reproduces_each_inner_joint_position():
    rng = np.random.default_rng(42)
    a = rng.normal(size=(7, 7))
    mass = np.eye(7) + a.T @ a * 0.04
    jacobian = np.c_[np.eye(6), rng.normal(size=6) * 0.1]
    q, initial = rng.normal(size=(2, 7)) * 0.1
    velocity, deltas = rng.normal(size=7) * 0.03, rng.normal(size=(4, 6)) * 0.02
    kp, kd = np.ones(6) * 100, np.ones(6) * 20
    lower, upper = -np.ones(7) * 2, np.ones(7) * 2
    model = build_joint_preview(mass, jacobian, kp, kd, q, initial, null_kp=10, dt=0.002)
    result = BatchedFreeJointRollout(model).rollout(velocity, deltas, np.eye(4), lower, upper, enforce_limits=False)
    expected = []
    for delta in deltas:
        current = build_joint_response(mass, jacobian, kp, kd, q, initial, null_kp=10, dt=0.002)
        for substeps in range(1, 26):
            sample = predict_joint_response(
                current, velocity, delta, lower, upper, substeps=substeps, enforce_limits=False
            )
            expected.append(sample["joint_position"])
        q, velocity = sample["joint_position"], sample["joint_velocity"]
    np.testing.assert_allclose(result["inner_joint_position"], expected, atol=1e-10, rtol=0)


def test_inner_position_jacobian_matches_centered_action_perturbation():
    model, velocity, deltas, site, lower, upper = example(4)
    candidate = BatchedFreeJointRollout(model)
    original = candidate.rollout(velocity, deltas, site, lower, upper, enforce_limits=False)
    plus, minus = deltas.copy(), deltas.copy()
    plus[1, 0] += 1e-6
    minus[1, 0] -= 1e-6
    a = candidate.rollout(velocity, plus, site, lower, upper, enforce_limits=False)
    b = candidate.rollout(velocity, minus, site, lower, upper, enforce_limits=False)
    numerical = (a["inner_joint_position"] - b["inner_joint_position"]) / 2e-6
    np.testing.assert_allclose(numerical, original["inner_joint_position_jacobian"][:, :, 6], atol=1e-8, rtol=1e-5)
