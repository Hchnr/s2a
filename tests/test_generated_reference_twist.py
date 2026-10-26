import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from s2a.generated_reference_twist import reference_world_twist


def test_generated_world_screw_velocity_uses_anchor_center_and_tail():
    anchor = np.eye(4)
    anchor[:3, :3] = Rotation.from_rotvec([0.4, -0.2, 0.3]).as_matrix()
    velocity, omega, dt = np.array([0.03, -0.01, 0.02]), np.array([0.2, -0.1, 0.3]), 0.05
    poses = np.repeat(anchor[None], 3, axis=0)
    for index, pose in enumerate(poses):
        pose[:3, 3] = velocity * dt * (index + 1)
        pose[:3, :3] = Rotation.from_rotvec(omega * dt * (index + 1)).as_matrix() @ anchor[:3, :3]
    for index in range(3):
        result = reference_world_twist(poses, index, anchor, dt)
        np.testing.assert_allclose(result["twist"], np.r_[velocity, omega], atol=1e-14, rtol=0)
        assert result["angular_frame"] == "MuJoCo world"
    single = reference_world_twist(poses[:1], 0, anchor, dt)
    assert single["mode"] == "single_anchor"
    np.testing.assert_allclose(single["twist"], np.r_[velocity, omega], atol=1e-14, rtol=0)


def test_unused_generated_future_does_not_change_current_velocity_and_pi_is_rejected():
    poses = np.repeat(np.eye(4)[None], 4, axis=0)
    first = reference_world_twist(poses, 0, np.eye(4), 0.05)
    poses[3, :3, :3] = Rotation.from_rotvec([1, 0.3, -0.2]).as_matrix()
    np.testing.assert_array_equal(reference_world_twist(poses, 0, np.eye(4), 0.05)["twist"], first["twist"])
    poses[1, :3, :3] = Rotation.from_rotvec([np.pi, 0, 0]).as_matrix()
    with pytest.raises(ValueError, match="pi"):
        reference_world_twist(poses, 0, np.eye(4), 0.05)
