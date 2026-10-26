import numpy as np
from scipy.spatial.transform import Rotation
from test_rate_preview_active_set import contract

from s2a.current_coupled_state import rotation_goal_from_command
from s2a.osc import OSCInterface


def interface():
    c = contract()
    return OSCInterface(*(c[key] for key in ("input_min", "input_max", "output_min", "output_max")))


def test_zero_rotation_command_retains_actual_previous_goal():
    current = Rotation.from_rotvec([0.2, -0.1, 0.3]).as_matrix()
    previous = Rotation.from_rotvec([-0.1, 0.2, 0.05]).as_matrix()

    def forbidden(_):
        raise AssertionError("Zero command must not decode a fresh rotation")

    goal, relative = rotation_goal_from_command(np.zeros(6), interface(), current, previous, matrix_decoder=forbidden)
    np.testing.assert_array_equal(goal, previous)
    np.testing.assert_allclose(relative, previous @ current.T, atol=1e-15)


def test_small_nonzero_command_uses_decoder_even_when_vendor_rounds_to_identity():
    current = Rotation.from_rotvec([0.2, -0.1, 0.3]).as_matrix()
    previous = Rotation.from_rotvec([-0.1, 0.2, 0.05]).as_matrix()
    command = np.zeros(6)
    command[3] = 1e-7
    calls = []

    def tiny_decoder(delta):
        calls.append(delta.copy())
        return np.eye(3)

    goal, relative = rotation_goal_from_command(command, interface(), current, previous, matrix_decoder=tiny_decoder)
    assert len(calls) == 1 and calls[0][0] != 0
    np.testing.assert_array_equal(goal, current)
    np.testing.assert_allclose(relative, np.eye(3), atol=1e-15)
