import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from s2a.reference_gains import ReferenceGains, relative_features


def trajectory():
    anchor = np.eye(4)
    anchor[:3, :3] = Rotation.from_euler("xyz", [0.4, -0.2, 0.7]).as_matrix()
    anchor[:3, 3] = [0.2, -0.1, 0.8]
    poses = np.repeat(anchor[None], 3, axis=0)
    poses[:, :3, 3] += [[0.01, -0.03, 0.02], [0.04, -0.02, 0.05], [0.02, 0.01, 0.07]]
    poses[:, :3, :3] = (
        anchor[:3, :3]
        @ Rotation.from_rotvec([[0.05, 0.02, 0.01], [-0.02, 0.04, 0.03], [0.03, -0.06, 0.02]]).as_matrix()
    )
    return anchor, poses


def test_identity_exact_and_orientation_damping_preserves_world_positions():
    anchor, poses = trajectory()
    opening = np.array([0.9, 0.6, 0.2])
    same, grip = ReferenceGains().apply(anchor, poses, opening)
    np.testing.assert_array_equal(same, poses)
    np.testing.assert_array_equal(grip, opening)
    held, grip = ReferenceGains(1, 0, 0.2).apply(anchor, poses, opening)
    np.testing.assert_array_equal(held[:, :3, 3], poses[:, :3, 3])
    np.testing.assert_allclose(held[:, :3, :3], np.repeat(anchor[None, :3, :3], 3, axis=0), atol=1e-15)
    np.testing.assert_allclose(grip, [1, 0.8, 0.4])


def test_world_frame_equivariance_and_separate_gain_geometry():
    anchor, poses = trajectory()
    world = np.eye(4)
    world[:3, :3] = Rotation.from_euler("xyz", [-0.8, 0.5, -0.3]).as_matrix()
    world[:3, 3] = [0.3, -0.8, 1.2]
    gains = ReferenceGains(1.4, 0.3, -0.1)
    result, opening = gains.apply(anchor, poses, [0.8, 0.6, 0.4])
    changed, changed_opening = gains.apply(world @ anchor, world @ poses, [0.8, 0.6, 0.4])
    np.testing.assert_allclose(changed, world @ result, atol=2e-15)
    np.testing.assert_array_equal(changed_opening, opening)
    before = relative_features(anchor, poses)
    after = relative_features(anchor, result)
    np.testing.assert_allclose(after[0], 1.4 * before[0], atol=1e-15)
    np.testing.assert_allclose(after[1], 0.3 * before[1], atol=1e-15)


def test_fit_box_constraints_zero_signal_and_weight_normalization():
    zero = np.zeros((4, 3))
    one = np.ones((4, 3))
    gains = ReferenceGains.fit(
        (zero, zero), (one, one), np.zeros(4), np.ones(4), np.full(4, 0.25), ridge=0.1, scales=[0.1, 0.5, 1]
    )
    assert gains == ReferenceGains(1, 1, 0.25)
    bounded = ReferenceGains.fit(
        (one, one), (4 * one, -one), np.ones(4), np.zeros(4), np.full(4, 0.25), ridge=0.1, scales=[0.1, 0.5, 1]
    )
    assert bounded == ReferenceGains(2, 0, -0.25)
    with pytest.raises(ValueError, match="sum"):
        ReferenceGains.fit((one, one), (one, one), np.ones(4), np.ones(4), np.ones(4), ridge=0.1, scales=[0.1, 0.5, 1])


def test_rejects_ambiguous_rotation_and_invalid_gains():
    anchor, poses = trajectory()
    poses[0, :3, :3] = anchor[:3, :3] @ Rotation.from_rotvec([np.pi, 0, 0]).as_matrix()
    with pytest.raises(ValueError, match="pi"):
        ReferenceGains().apply(anchor, poses, [0.1, 0.2, 0.3])
    for args in [(True, 1, 0), (1, np.nan, 0), (2.1, 0, 0), (1, 1, 0.3)]:
        with pytest.raises(ValueError):
            ReferenceGains(*args)
