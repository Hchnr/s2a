import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation

from s2a.control import PIDConfig, PosePID
from s2a.geometry import advance, pose_error, transform
from s2a.trajectory import ReferenceTrajectory


def test_pid_tracking_error_not_delta_first_sample_and_independent_state():
    config = PIDConfig(kp=(1.0,) * 6, ki=(2.0,) * 6, kd=(0.1,) * 6)
    pid, other = PosePID(config), PosePID(config)
    target = advance(np.eye(4), [0.01, 0, 0, 0, 0, 0.01], 1)
    first = pid.update(np.eye(4), target, np.zeros(6), 0)
    assert_allclose(first["integral"], 0)
    second = pid.update(np.eye(4), target, np.zeros(6), 0.1)
    assert_allclose(second["integral"], second["error"] * 0.1)
    assert_allclose(second["derivative"], 0)
    assert_allclose(other.state.integral, 0)
    cached = pid.update(np.eye(4), target, np.ones(6), 0.1)
    assert cached is second
    pid.reset()
    assert_allclose(pid.state.integral, 0)


def test_final_downstream_saturation_freezes_integral_and_releases():
    pid = PosePID(PIDConfig(ki=(1.0,) * 6, max_translation_speed=10, max_translation_acceleration=100))
    target = advance(np.eye(4), [0.1, 0, 0, 0, 0, 0], 1)
    pid.update(np.eye(4), target, np.zeros(6), 0)
    for step in range(1, 11):
        result = pid.update(
            np.eye(4), target, np.zeros(6), step * 0.05, apply=lambda v: (np.zeros(6), "downstream_blocked")
        )
        assert result["command"] == "downstream_blocked"
    assert_allclose(pid.state.integral, 0)
    result = pid.update(np.eye(4), target, np.zeros(6), 0.55)
    assert result["integral"][0] > 0
    assert result["velocity"][0] > 0


def test_measurement_derivative_avoids_replan_kick_and_filters_noise():
    pid = PosePID(PIDConfig(kd=(1.0,) * 6, tau_d=0.1))
    pid.update(np.eye(4), np.eye(4), np.zeros(6), 0)
    moved_target = advance(np.eye(4), [0.1, 0, 0, 0, 0, 0], 1)
    result = pid.update(np.eye(4), moved_target, np.zeros(6), 0.05)
    assert_allclose(result["derivative"], 0)
    measured = advance(np.eye(4), [0.01, 0, 0, 0, 0, 0], 0.05)
    result = pid.update(measured, moved_target, np.zeros(6), 0.1)
    assert_allclose(result["derivative"][0], -(1 - np.exp(-0.5)) * 0.01)


def test_stale_gap_time_reversal_and_pi_policy():
    pid = PosePID(PIDConfig(ki=(1.0,) * 6))
    target = advance(np.eye(4), [0.1, 0, 0, 0, 0, 0], 1)
    pid.update(np.eye(4), target, np.zeros(6), 1.0)
    pid.update(np.eye(4), target, np.zeros(6), 1.1)
    before = pid.state.integral.copy()
    with pytest.raises(ValueError, match="stale"):
        pid.update(np.eye(4), target, np.zeros(6), 1.2, now=2.0)
    assert_allclose(pid.state.integral, before)
    for time, reason in [(2.0, "feedback_gap"), (0.0, "time_reversal")]:
        result = pid.update(np.eye(4), target, np.zeros(6), time)
        assert result["reset_reason"] == reason
        assert_allclose(result["integral"], 0)
        assert_allclose(result["derivative"], 0)
    pi_target = transform(np.zeros(3), Rotation.from_rotvec([0, 0, np.pi]).as_matrix())
    with pytest.raises(ValueError, match="pi"):
        pid.update(np.eye(4), pi_target, np.zeros(6), 0.1)


def test_rate_limit_and_no_state_sharing():
    pid = PosePID(PIDConfig(max_translation_acceleration=0.1))
    pid.update(np.eye(4), np.eye(4), np.zeros(6), 0)
    result = pid.update(np.eye(4), np.eye(4), [1, 0, 0, 0, 0, 0], 0.1)
    assert_allclose(result["velocity"], [0.01, 0, 0, 0, 0, 0])


def test_reference_anchor_velocity_and_chunk_splice_continuity():
    times = np.array([0.0, 1.0, 2.0, 3.0])
    poses = np.array([advance(np.eye(4), [0.01, 0, 0, 0, 0, 0.1], max(t, 1e-12)) for t in times])
    reference = ReferenceTrajectory(times, poses)
    t, v = reference.sample(0.7)
    eps = 1e-6
    a, _ = reference.sample(0.7 + eps)
    b, _ = reference.sample(0.7 - eps)
    assert_allclose(v, pose_error(a, b) / (2 * eps), atol=1e-9)
    assert_allclose(t[:3, 3], [0.007, 0, 0], atol=1e-12)
    deltas = np.tile([0.02, 0, 0, 1, 0, 0, 0, 1, 0], (3, 1))
    switch = reference.splice(1.0, np.array([1.0, 2.0, 3.0, 4.0]), deltas, 0.5)
    pose0, velocity0 = switch.sample(1.0)
    assert_allclose(pose0, reference.sample(1.0)[0], atol=1e-12)
    assert_allclose(velocity0, reference.sample(1.0)[1], atol=1e-12)
    assert_allclose(switch.sample(1.0 + eps)[1], velocity0, atol=1e-5)
    assert_allclose(switch.sample(1.5 - eps)[1], switch.sample(1.5)[1], atol=1e-5)
    assert_allclose(reference.sample(10)[1], 0)
    with pytest.raises(ValueError):
        reference.sample(-1)
