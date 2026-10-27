import numpy as np
import pytest

from s2a.causal_arm_residual import CausalArmResidualOSC
from s2a.osc import OSCInterface
from s2a.pose_velocity_inverse_osc import PoseVelocityInverseOSC


def make(cls, rate=5):
    interface = OSCInterface(-np.ones(6), np.ones(6), -np.ones(6) * 0.05, np.ones(6) * 0.05)
    return cls(interface, np.ones(6) * 150, np.ones(6) * 25, 0.05, max_command_rate=rate)


def test_first_command_matches083_and_has_no_unobserved_residual():
    a, b = make(CausalArmResidualOSC), make(PoseVelocityInverseOSC)
    measured, desired = np.eye(4), np.eye(4)
    desired[:3, 3] = [0.01, -0.02, 0.005]
    x = a.convert(measured, desired, np.zeros(6), desired_linear_velocity=np.ones(3) * 0.01)
    y = b.convert(measured, desired, np.zeros(6), desired_linear_velocity=np.ones(3) * 0.01)
    np.testing.assert_array_equal(x["command"], y["command"])
    assert x["response_residual"]["observed_previous_endpoint"] is False
    assert x["response_residual"]["previous_command_cycle"] is None


def test_known_disturbance_recovered_from_constrained_sent_command_and_current_endpoint():
    a = make(CausalArmResidualOSC)
    measured, desired = np.eye(4), np.eye(4)
    desired[:3, 3] = 1
    first = a.convert(measured, desired, np.zeros(6), desired_linear_velocity=np.ones(3))
    assert first["saturated"] and first["rate_limited"]
    dp, dv = np.array([0.001, -0.002, 0.003]), np.array([0.01, -0.02, 0.03])
    current = np.eye(4)
    current[:3, 3] = first["nominal_predicted_motion"][:3] + dp
    velocity = np.r_[first["nominal_predicted_linear_velocity"] + dv, np.zeros(3)]
    target = np.eye(4)
    target[:3, 3] = current[:3, 3] + 0.001
    result = a.convert(current, target, velocity, desired_linear_velocity=np.zeros(3))
    np.testing.assert_allclose(result["response_residual"]["position"], dp, atol=1e-16)
    np.testing.assert_allclose(result["response_residual"]["linear_velocity"], dv, atol=1e-16)
    reference = make(PoseVelocityInverseOSC)
    reference.previous_command = first["command"].copy()
    adjusted = target.copy()
    adjusted[:3, 3] -= dp
    expected = reference.convert(current, adjusted, velocity, desired_linear_velocity=-dv)
    np.testing.assert_allclose(result["command"], expected["command"], atol=1e-14)
    np.testing.assert_allclose(result["predicted_linear_velocity"], result["nominal_predicted_linear_velocity"] + dv)
    assert result["response_residual"]["previous_command_cycle"] == 0


def test_reset_clears_pending_measurement_and_rate_history():
    a = make(CausalArmResidualOSC)
    target = np.eye(4)
    target[0, 3] = 0.02
    a.convert(np.eye(4), target, np.zeros(6), desired_linear_velocity=np.zeros(3))
    a.reset()
    x = a.convert(np.eye(4), target, np.zeros(6), desired_linear_velocity=np.zeros(3))
    y = make(CausalArmResidualOSC).convert(np.eye(4), target, np.zeros(6), desired_linear_velocity=np.zeros(3))
    np.testing.assert_array_equal(x["command"], y["command"])
    assert x["response_residual"]["cycle"] == 0 and not x["response_residual"]["observed_previous_endpoint"]


def test_bad_observation_does_not_advance_pending_history():
    a = make(CausalArmResidualOSC)
    with pytest.raises(ValueError):
        a.convert(np.eye(4), np.eye(4), np.ones(6) * np.nan, desired_linear_velocity=np.zeros(3))
    result = a.convert(np.eye(4), np.eye(4), np.zeros(6), desired_linear_velocity=np.zeros(3))
    assert result["response_residual"]["cycle"] == 0
