import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from s2a.rotation_drift import RotationDrift, episode_query_weights, world_rotation_residual


def poses():
    result = np.repeat(np.eye(4)[None], 3, axis=0)
    result[:, :3, :3] = Rotation.from_euler("xyz", [[0.4, -0.2, 0.7], [-0.1, 0.5, 0.6], [0.8, 0.3, -0.2]]).as_matrix()
    result[:, :3, 3] = [[0.2, -0.1, 0.8], [0.3, -0.2, 0.9], [0.4, -0.3, 1]]
    return result


def test_world_left_noncommuting_correction_preserves_position_opening_and_zero():
    source = poses()
    opening = np.array([1.2, 0.4, -0.1])
    times = np.array([0, 0.5, 2])
    bias = np.array([0.1, -0.1, 0.1])
    identity, same_opening = RotationDrift().apply(source, opening, times)
    np.testing.assert_array_equal(identity, source)
    np.testing.assert_array_equal(same_opening, opening)
    corrected, copied_opening = RotationDrift(tuple(bias)).apply(source, opening, times)
    expected = Rotation.from_rotvec(-times[:, None] * bias).as_matrix() @ source[:, :3, :3]
    np.testing.assert_allclose(corrected[:, :3, :3], expected, atol=1e-15, rtol=0)
    assert not np.allclose(expected[1], source[1, :3, :3] @ Rotation.from_rotvec(-times[1] * bias).as_matrix())
    np.testing.assert_array_equal(corrected[0], source[0])
    np.testing.assert_array_equal(corrected[:, :3, 3], source[:, :3, 3])
    np.testing.assert_array_equal(copied_opening, opening)
    np.testing.assert_allclose(np.linalg.det(corrected[:, :3, :3]), 1, atol=1e-15, rtol=0)


def test_fixed_right_tcp_rotation_and_reference_prefix_do_not_change_the_correction():
    source, times = poses(), [0.05, 0.1, 0.15]
    model = RotationDrift((0.1, 0.05, -0.02))
    right = np.eye(4)
    right[:3, :3] = Rotation.from_euler("xyz", [0.7, 0.3, -0.4]).as_matrix()
    result, _ = model.apply(source, [0.9, 0.7, 0.3], times)
    right_result, _ = model.apply(source @ right, [0.9, 0.7, 0.3], times)
    np.testing.assert_allclose(right_result, result @ right, atol=1e-15, rtol=0)
    prefix, _ = model.apply(source[:2], [0.9, 0.7], times[:2])
    np.testing.assert_array_equal(prefix, result[:2])
    world = np.eye(4)
    world[:3, :3] = Rotation.from_euler("xyz", [-0.8, 0.3, 0.6]).as_matrix()
    world[:3, 3] = [0.2, -0.3, 0.7]
    rotated_bias = world[:3, :3] @ model.bias_rad_per_second
    changed, _ = RotationDrift(tuple(rotated_bias)).apply(world @ source, [0.9, 0.7, 0.3], times)
    np.testing.assert_allclose(changed, world @ result, atol=1e-15, rtol=0)


def test_weighted_closed_form_world_residual_sign_and_ridge():
    truth = poses()[:, :3, :3]
    times, weights = np.array([0.05, 0.4, 1.2]), np.array([0.2, 0.3, 0.5])
    actual_bias = np.array([0.04, -0.03, 0.02])
    predicted = Rotation.from_rotvec(times[:, None] * actual_bias).as_matrix() @ truth
    np.testing.assert_allclose(world_rotation_residual(predicted, truth), times[:, None] * actual_bias, atol=1e-15)
    fitted = RotationDrift.fit(predicted, truth, times, weights)
    expected = actual_bias * np.sum(weights * times**2) / (np.sum(weights * times**2) + 0.1)
    np.testing.assert_allclose(fitted.bias_rad_per_second, expected, atol=1e-15, rtol=0)
    zero = RotationDrift.fit(truth, truth, times, weights)
    np.testing.assert_allclose(zero.bias_rad_per_second, 0, atol=1e-15)
    repeated = RotationDrift.fit(
        np.concatenate([predicted, predicted]),
        np.concatenate([truth, truth]),
        np.tile(times, 2),
        np.tile(weights / 2, 2),
    )
    np.testing.assert_allclose(repeated.bias_rad_per_second, fitted.bias_rad_per_second, atol=1e-15, rtol=0)


def test_norm_cap_bounds_angle_and_bad_weights_or_times_are_rejected():
    truth = np.repeat(np.eye(3)[None], 2, axis=0)
    predicted = Rotation.from_rotvec([[0.5, 0.5, 0], [1, 1, 0]]).as_matrix()
    model = RotationDrift.fit(predicted, truth, [0.5, 1], [0.5, 0.5])
    np.testing.assert_allclose(np.linalg.norm(model.bias_rad_per_second), 0.25, atol=1e-15)
    source = poses()[:2]
    corrected, _ = model.apply(source, [0.5, 0.5], [0, 2])
    angle = Rotation.from_matrix(corrected[1, :3, :3] @ source[1, :3, :3].T).magnitude()
    assert angle <= 0.5 + 1e-15
    with pytest.raises(ValueError, match="sum"):
        RotationDrift.fit(predicted, truth, [0.5, 1], [1, 1])
    for times in ([-0.1, 0.1], [0.2, 0.1], [0, 2.01]):
        with pytest.raises(ValueError, match="times"):
            model.apply(source, [0.2, 0.4], times)


def test_episode_query_endpoint_weights_are_balanced_without_endpoint_count_prior():
    weights = episode_query_weights([1, 1, 2], [2, 4, 3])
    np.testing.assert_allclose(weights[:2].sum(), 0.25)
    np.testing.assert_allclose(weights[2:6].sum(), 0.25)
    np.testing.assert_allclose(weights[6:].sum(), 0.5)
    np.testing.assert_allclose(weights.sum(), 1)
    with pytest.raises(ValueError, match="positive integers"):
        episode_query_weights([1], [True])


def test_ambiguous_relative_rotation_and_invalid_bias_are_rejected():
    with pytest.raises(ValueError, match="pi"):
        RotationDrift.fit(Rotation.from_rotvec([[np.pi, 0, 0]]).as_matrix(), np.eye(3)[None], [1], [1])
    for bias in ((0.26, 0, 0), (np.nan, 0, 0)):
        with pytest.raises(ValueError):
            RotationDrift(bias)
