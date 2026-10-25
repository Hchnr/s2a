import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from s2a.training_state_recovery import recovery_reference


def endpoints():
    poses = np.repeat(np.eye(4)[None], 81, axis=0)
    poses[:, :3, :3] = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    poses[:, 0, 3] = np.linspace(0.1, 0.2, 81)
    return poses, np.linspace(0.8, 0.2, 81)


@pytest.mark.parametrize("offset", [0.0, 0.02, -0.02])
def test_base_offset_return_and_original_native_endpoints(offset):
    poses, opening = endpoints()
    original = poses.copy()
    targets, aperture = recovery_reference(poses, opening, base_x_offset_m=offset)
    assert len(targets) == 136
    assert targets[15, 0, 3] == pytest.approx(0.1 + offset)
    np.testing.assert_array_equal(targets[55], poses[0])
    np.testing.assert_array_equal(targets[56:], poses[1:])
    np.testing.assert_array_equal(aperture[:56], np.full(56, opening[0]))
    np.testing.assert_array_equal(aperture[56:], opening[1:])
    np.testing.assert_array_equal(targets[:56, :3, :3], np.repeat(poses[:1, :3, :3], 56, axis=0))
    np.testing.assert_array_equal(targets[:56, 1:3, 3], np.repeat(poses[:1, 1:3, 3], 56, axis=0))
    targets[:] = 0
    np.testing.assert_array_equal(poses, original)


def test_reject_invalid_reference_and_step_budget():
    poses, opening = endpoints()
    with pytest.raises(ValueError):
        recovery_reference(poses, opening[:-1], base_x_offset_m=0.02)
    with pytest.raises(ValueError):
        recovery_reference(poses, opening, base_x_offset_m=float("nan"))
    with pytest.raises(ValueError):
        recovery_reference(poses, opening, base_x_offset_m=0.02, perturb_steps=True)
    opening[1] = 1.1
    with pytest.raises(ValueError):
        recovery_reference(poses, opening, base_x_offset_m=0.02)
