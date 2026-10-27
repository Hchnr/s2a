import numpy as np
import pytest
from current_coupled_independent import response
from scipy.spatial.transform import Rotation
from test_current_coupled_response import plant
from test_rate_preview_active_set import contract

from s2a.coupled_response_history import CoupledResponseHistory
from s2a.current_coupled_response import build_response


def fixture():
    c = contract()
    mass, j = plant()
    relative = Rotation.from_rotvec([0.2, -0.1, 0.05]).as_matrix()
    model = build_response(mass, j, c["kp"], c["kd"], relative)
    independent = response(mass, j, c["kp"], c["kd"], relative)
    pose = np.eye(4)
    pose[:3, 3] = [0.4, 0.1, 0.8]
    twist = np.array([0.02, -0.01, 0.03, 0.1, -0.2, 0.05])
    return c, model, independent, pose, twist


def test_history_uses_actual_command_and_world_rotation_composition_after_completion():
    c, model, independent, pose, twist = fixture()
    history = CoupledResponseHistory()
    np.testing.assert_array_equal(history.current(pose, twist), np.zeros(12))
    actual = np.array([-0.1, 0.05, 0.2])
    commit = history.commit_executed(pose, twist, model, actual, c)
    nominal = (
        independent["initial_velocity_response"] @ twist
        + independent["goal_forcing_response"] @ np.r_[0.05 * actual, independent["rotation_forcing"]]
    )
    np.testing.assert_allclose(commit["nominal_response"], nominal, atol=1e-9, rtol=0)
    # A consumer changing its receipt must not change the committed prediction.
    commit["nominal_response"][:] = 999
    with pytest.raises(ValueError, match="observed"):
        history.current(pose, twist)
    with pytest.raises(ValueError, match="observed"):
        history.commit_executed(pose, twist, model, actual, c)
    dp = np.array([0.0002, -0.0001, 0.0003])
    rotation = Rotation.from_rotvec([0.001, -0.002, 0.0005])
    dv = np.array([0.003, -0.002, 0.001, 0.004, 0.001, -0.003])
    next_pose = pose.copy()
    next_pose[:3, 3] += nominal[:3] + dp
    next_pose[:3, :3] = (rotation * Rotation.from_rotvec(nominal[3:6])).as_matrix() @ pose[:3, :3]
    next_twist = nominal[6:] + dv
    observed = history.observe_completed(next_pose, next_twist)
    expected = np.r_[dp, rotation.as_rotvec(), 0.05 * dv]
    np.testing.assert_allclose(observed, expected, atol=1e-9, rtol=0)
    np.testing.assert_allclose(history.current(next_pose, next_twist), expected, atol=1e-9, rtol=0)
    assert history.completed_periods == 1
    wrong = next_twist.copy()
    wrong[0] += 0.001
    with pytest.raises(ValueError, match="differs"):
        history.current(next_pose, wrong)


def test_independent_intervals_reset_pending_prediction_and_innovation():
    c, model, _, pose, twist = fixture()
    history = CoupledResponseHistory()
    with pytest.raises(ValueError, match="committed"):
        history.observe_completed(pose, twist)
    history.commit_executed(pose, twist, model, np.zeros(3), c)
    history.reset()
    np.testing.assert_array_equal(history.current(pose, twist), np.zeros(12))
    assert history.completed_periods == 0 and history.pending is None
    bad = dict(model, dt=0.1)
    with pytest.raises(ValueError, match="20Hz"):
        history.commit_executed(pose, twist, bad, np.zeros(3), c)
    assert history.pending is None
