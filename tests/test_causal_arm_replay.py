import numpy as np
import pytest
from causal_arm_independent_replay import CausalArmReplay
from scipy.spatial.transform import Rotation

from s2a.causal_arm_residual import CausalArmResidualOSC
from s2a.osc import OSCInterface


@pytest.mark.parametrize("damping", [0.5, 1.0, 2.0])
def test_independent_closed_form_and_two_row_lstsq_match_sequential_control(damping):
    interface = OSCInterface(
        -np.ones(6), np.ones(6), np.array([-0.05] * 3 + [-0.5] * 3), np.array([0.05] * 3 + [0.5] * 3)
    )
    kp = np.full(6, 150.0)
    kd = 2 * np.sqrt(kp) * damping
    model = CausalArmResidualOSC(interface, kp, kd, 0.05, max_command_rate=5)
    replay = CausalArmReplay(interface, kp, kd, 0.05, max_command_rate=5)
    for index in range(10):
        desired = np.eye(4)
        desired[:3, 3] = [0.001 * index, -0.002, 0.003]
        desired[:3, :3] = Rotation.from_rotvec([0.1 if index < 5 else 0.4, 0, 0]).as_matrix()
        velocity = np.array([0.01, -0.02, 0.005, 0.02, -0.01, 0.03])
        target = np.array([0.03, -0.01, 0.02])
        actual = model.convert(np.eye(4), desired, velocity, desired_linear_velocity=target)
        independent = replay.convert(np.eye(4), desired, velocity, desired_linear_velocity=target)
        for key in (
            "command",
            "requested_delta",
            "predicted_linear_velocity",
            "nominal_predicted_linear_velocity",
            "nominal_predicted_motion",
        ):
            np.testing.assert_allclose(actual[key], independent[key], rtol=0, atol=1e-10)
        for key in ("position", "linear_velocity"):
            np.testing.assert_allclose(
                actual["response_residual"][key], independent["response_residual"][key], atol=1e-10, rtol=0
            )
        for key in ("cycle", "previous_command_cycle", "observed_previous_endpoint"):
            assert actual["response_residual"][key] == independent["response_residual"][key]
        for key in ("saturated", "rate_limited", "rotation_inverse_enabled"):
            assert actual[key] == independent[key]
