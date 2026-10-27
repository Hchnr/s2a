import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from s2a.current_wrench import world_wrench_at_point


def test_lever_arm_and_rotated_force_have_correct_moment_origin():
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_rotvec([0, 0, np.pi / 2]).as_matrix()
    pose[:3, 3] = [0, 0, 1]
    value = world_wrench_at_point([2, 0, 0, 0, 0, 3], pose, np.zeros(3))
    np.testing.assert_allclose(value, [0, 2, 0, -2, 0, 3], atol=1e-14)


def test_shifting_reference_point_back_recovers_sensor_world_moment():
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_rotvec([0.3, -0.4, 0.2]).as_matrix()
    pose[:3, 3] = [0.1, 0.2, -0.3]
    local = np.arange(6) * 0.4
    point = np.array([-0.2, 0.4, 0.1])
    value = world_wrench_at_point(local, pose, point)
    np.testing.assert_allclose(
        value[3:] + np.cross(point - pose[:3, 3], value[:3]), pose[:3, :3] @ local[3:], atol=1e-14
    )


def test_nonfinite_or_invalid_frame_is_rejected():
    with pytest.raises(ValueError):
        world_wrench_at_point([np.nan] * 6, np.eye(4), np.zeros(3))
    pose = np.eye(4)
    pose[0, 0] = 2
    with pytest.raises(ValueError):
        world_wrench_at_point(np.zeros(6), pose, np.zeros(3))
