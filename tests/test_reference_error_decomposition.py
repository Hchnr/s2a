"""Check actual right-local frame coupling and coherent translation bias."""

import numpy as np
from scipy.spatial.transform import Rotation

from s2a.geometry import matrix_to_rot6d
from s2a.reference_error_decomposition import decompose, position_summary


def arrays(n):
    states, actions = np.zeros((n + 1, 128)), np.zeros((n, 128))
    states[:, 3:9] = actions[:, 3:9] = matrix_to_rot6d(np.eye(3))
    return states, actions


def test_rotation_error_alone_rotates_a_correct_local_translation():
    states, actions = arrays(2)
    states[2, 0] = actions[1, 0] = 0.1
    actions[0, 3:9] = matrix_to_rot6d(Rotation.from_euler("z", 90, degrees=True).as_matrix())
    parts = decompose(states, actions)
    np.testing.assert_allclose(parts["local_cumulative"], 0, atol=1e-15)
    np.testing.assert_allclose(parts["frame_cumulative"][-1], [-0.1, 0.1, 0], atol=1e-15)
    summary = position_summary(parts)
    assert abs(summary["total_mse_m2"] - summary["frame_cumulative_mse_m2"]) < 1e-15


def test_small_coherent_bias_accumulates_without_orientation_error():
    states, actions = arrays(4)
    states[:, 0] = np.arange(5) * 0.01
    actions[:, 0] = 0.02
    parts = decompose(states, actions)
    np.testing.assert_allclose(parts["frame_cumulative"], 0, atol=1e-15)
    np.testing.assert_allclose(parts["local_cumulative"][:, 0], np.arange(1, 5) * 0.01, atol=1e-15)


def test_initial_tcp_orientation_rotates_local_bias_into_base_frame():
    states, actions = arrays(2)
    states[:, 3:9] = matrix_to_rot6d(Rotation.from_euler("z", 90, degrees=True).as_matrix())
    actions[:, 0] = 0.01
    parts = decompose(states, actions)
    np.testing.assert_allclose(parts["total_cumulative"][:, 1], [0.01, 0.02], atol=1e-15)
    np.testing.assert_allclose(parts["frame_cumulative"], 0, atol=1e-15)
