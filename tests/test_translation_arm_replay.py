import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from translation_arm_independent_replay import TranslationArmReplay

from s2a.osc import OSCInterface
from s2a.translation_inverse_osc import TranslationInverseOSC


@pytest.mark.parametrize("damping", [0.5, 1.0, 2.0])
def test_closed_form_replay_matches_matrix_exponential_with_sequential_constraints(damping):
    interface = OSCInterface(
        -np.ones(6), np.ones(6), np.array([-0.05] * 3 + [-0.5] * 3), np.array([0.05] * 3 + [0.5] * 3)
    )
    kp = np.full(6, 150.0)
    kd = 2 * np.sqrt(kp) * damping
    producer = TranslationInverseOSC(interface, kp, kd, 0.05, max_command_rate=5)
    replay = TranslationArmReplay(interface, kp, kd, 0.05, max_command_rate=5)
    np.testing.assert_allclose(producer.response, replay.response, rtol=0, atol=1e-12)
    np.testing.assert_allclose(producer.momentum, replay.momentum, rtol=0, atol=1e-12)
    for step in range(8):
        measured = np.eye(4)
        desired = np.eye(4)
        desired[:3, 3] = [0.001 * step, -0.003, 0.002]
        desired[:3, :3] = Rotation.from_rotvec([0.1 if step < 4 else 0.4, 0.0, 0.0]).as_matrix()
        v = np.array([0.01, -0.02, 0.005, 0.02, -0.01, 0.03])
        actual = producer.convert(measured, desired, v)
        independent = replay.convert(measured, desired, v)
        for key in ("command", "requested_delta"):
            np.testing.assert_allclose(actual[key], independent[key], rtol=0, atol=1e-10)
        for key in ("saturated", "rate_limited", "rotation_inverse_enabled"):
            assert actual[key] == independent[key]
