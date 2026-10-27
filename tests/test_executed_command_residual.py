import numpy as np
import pytest

from s2a.causal_arm_residual import CausalArmResidualOSC
from s2a.executed_command_residual import ExecutedCommandResidualOSC
from s2a.osc import OSCInterface


def make(cls):
    return cls(
        OSCInterface(-np.ones(6), np.ones(6), -np.ones(6) * 0.05, np.ones(6) * 0.05),
        np.ones(6) * 150,
        np.ones(6) * 25,
        0.05,
        max_command_rate=5,
    )


def test_actual_mixed_command_not_provisional_is_used_for_next_innovation():
    converter = make(ExecutedCommandResidualOSC)
    desired = np.eye(4)
    desired[:3, 3] = 0.02
    candidate = converter.convert(np.eye(4), desired, np.zeros(6), desired_linear_velocity=np.zeros(3))
    actual = -candidate["command"]
    committed = converter.commit_executed(np.eye(4), np.zeros(6), actual)
    disturbance = np.array([0.001, -0.002, 0.003])
    velocity_disturbance = np.array([0.01, -0.02, 0.03])
    observed = np.eye(4)
    observed[:3, 3] = committed["nominal_motion"] + disturbance
    velocity = np.r_[committed["nominal_linear_velocity"] + velocity_disturbance, np.zeros(3)]
    next_result = converter.convert(observed, observed, velocity, desired_linear_velocity=np.zeros(3))
    np.testing.assert_allclose(next_result["response_residual"]["position"], disturbance, atol=1e-15)
    np.testing.assert_allclose(next_result["response_residual"]["linear_velocity"], velocity_disturbance, atol=1e-15)
    assert next_result["response_residual"]["previous_command_cycle"] == 0
    assert np.max(np.abs(next_result["command"] - actual)) <= 0.25 + 1e-12


def test_committing_same_command_preserves085_and_reset_clears_state():
    converter, original = make(ExecutedCommandResidualOSC), make(CausalArmResidualOSC)
    desired = np.eye(4)
    desired[:3, 3] = 0.01
    for i in range(3):
        pose = np.eye(4)
        pose[:3, 3] = i * 0.001
        velocity = np.ones(6) * 0.01
        a = converter.convert(pose, desired, velocity, desired_linear_velocity=np.zeros(3))
        b = original.convert(pose, desired, velocity, desired_linear_velocity=np.zeros(3))
        np.testing.assert_allclose(a["command"], b["command"], atol=1e-14)
        converter.commit_executed(pose, velocity, a["command"])
    converter.reset()
    with pytest.raises(ValueError, match="converted"):
        converter.commit_executed(np.eye(4), np.zeros(6), np.zeros(6))


def test_stale_or_different_state_commit_rejected_without_corrupting_history():
    converter = make(ExecutedCommandResidualOSC)
    candidate = converter.convert(np.eye(4), np.eye(4), np.zeros(6), desired_linear_velocity=np.zeros(3))
    wrong = np.eye(4)
    wrong[0, 3] = 0.001
    with pytest.raises(ValueError, match="same current"):
        converter.commit_executed(wrong, np.zeros(6), candidate["command"])
    converter.commit_executed(np.eye(4), np.zeros(6), candidate["command"])
    with pytest.raises(ValueError, match="converted"):
        converter.commit_executed(np.eye(4), np.zeros(6), candidate["command"])
