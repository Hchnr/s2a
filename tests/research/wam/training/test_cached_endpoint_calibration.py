"""Geometry, causality, bounds, and the registered robust opening objective."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from s2a.geometry import transform
from s2a.research.wam.training.cached_endpoint_calibration import (
    apply_calibration,
    features,
    fit_opening,
    fit_pose,
    training_targets,
)


@pytest.fixture
def config():
    return dict(
        pi_margin_rad=0.0001,
        position_scale_m=0.1,
        rotation_scale_rad=0.5,
        position_cap_m=0.05,
        rotation_cap_rad=0.25,
        opening_cap=0.25,
    )


def reference():
    fraction = np.arange(1, 41) / 40
    poses = transform(
        np.column_stack((0.08 * fraction, 0.03 * fraction, -0.02 * fraction)),
        Rotation.from_rotvec(np.outer(fraction, [0.15, -0.2, 0.1])).as_matrix(),
    )
    return poses, np.linspace(0.95, 0.1, 40)


def test_zero_correction_preserves_reference_and_global_frame_equivariance(config):
    anchor, (poses, opening) = np.eye(4), reference()
    zero_pose, zero_opening = np.zeros((7, 6)), np.zeros(7)
    unchanged, unchanged_opening = apply_calibration(anchor, poses, opening, 0.9, zero_pose, zero_opening, config)
    np.testing.assert_allclose(unchanged, poses, atol=1e-14)
    np.testing.assert_allclose(unchanged_opening, opening, atol=1e-14)

    world = transform(np.array([0.4, -0.1, 0.3]), Rotation.from_euler("xyz", [0.7, -0.4, 0.3]).as_matrix())
    weights = np.arange(42).reshape(7, 6) / 100
    aperture_weights = np.arange(7) / 30
    local = apply_calibration(anchor, poses, opening, 0.9, weights, aperture_weights, config)
    shifted = apply_calibration(world @ anchor, world @ poses, opening, 0.9, weights, aperture_weights, config)
    np.testing.assert_allclose(shifted[0], world @ local[0], atol=1e-13)
    np.testing.assert_allclose(shifted[1], local[1], atol=1e-14)


def test_corrections_are_bounded_at_every_endpoint(config):
    poses, opening = reference()
    corrected, corrected_opening = apply_calibration(
        np.eye(4), poses, opening, 0.9, np.full((7, 6), 500.0), np.full(7, -500.0), config
    )
    fraction = np.arange(1, 41) / 40
    position_delta = np.linalg.norm(corrected[:, :3, 3] - poses[:, :3, 3], axis=1)
    rotation_delta = Rotation.from_matrix(corrected[:, :3, :3] @ poses[:, :3, :3].transpose(0, 2, 1)).magnitude()
    assert np.all(position_delta <= config["position_cap_m"] * fraction + 1e-14)
    assert np.all(rotation_delta <= config["rotation_cap_rad"] * fraction + 1e-14)
    assert np.all(np.abs(corrected_opening - opening) <= config["opening_cap"] + 1e-14)
    assert np.all((corrected_opening >= 0) & (corrected_opening <= 1))


def test_training_residual_is_cumulative_and_ridge_is_normalized_by_rows(config):
    poses, opening = reference()
    truth = poses.copy()
    truth[:, :3, 3] += np.column_stack((np.arange(1, 41) * 0.001, np.zeros((40, 2))))
    x, _ = features(np.eye(4), poses, opening, 0.9, config)
    y = training_targets(np.eye(4), poses, truth, config)
    np.testing.assert_allclose(y[:, 0], np.arange(1, 41) * 0.01)
    assert np.any(fit_pose(x, y, 0.1))
    np.testing.assert_allclose(fit_pose(x, y, 0.1), fit_pose(np.tile(x, (2, 1)), np.tile(y, (2, 1)), 0.1))


def test_opening_fit_uses_median_residual_not_squared_error():
    x = np.zeros((5, 7))
    x[:, 0] = 1
    coefficients, diagnostic = fit_opening(x, np.zeros(5), np.array([1, 1, 1, 0, 0]), 0.01, 10)
    assert coefficients[0] == pytest.approx(1.0)
    assert diagnostic["objective"] == pytest.approx(0.41)
    first, second = np.split(np.asarray(diagnostic["inequality_dual"]), 2)
    assert np.max(np.r_[first, second]) <= 1e-9
    assert np.max(np.abs(x.T @ (first - second))) <= 0.01 + 1e-9
    assert np.max(-first - second) <= 0.2 + 1e-9
    assert np.dot(np.array([1, 1, 1, 0, 0]), first - second) == pytest.approx(diagnostic["objective"])


def test_rejects_ambiguous_rotation_and_invalid_current_opening(config):
    poses, opening = reference()
    poses[0, :3, :3] = Rotation.from_rotvec([np.pi - 1e-5, 0, 0]).as_matrix()
    with pytest.raises(ValueError, match="near pi"):
        features(np.eye(4), poses, opening, 0.9, config)
    poses, opening = reference()
    with pytest.raises(ValueError, match="measured opening"):
        features(np.eye(4), poses, opening, 1.1, config)
