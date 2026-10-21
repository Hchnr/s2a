"""Geometry, causality, ridge supervision and bounded reference interventions."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from s2a.generated_reference_calibration import CalibrationConfig, GeneratedReferenceCalibration, reference_features


def references(count=40):
    anchor = np.eye(4)
    anchor[:3, :3] = Rotation.from_euler("xyz", [0.4, -0.3, 0.6]).as_matrix()
    anchor[:3, 3] = [0.3, -0.2, 0.5]
    times = np.arange(1, count + 1) / 20
    poses = np.repeat(anchor[None], count, axis=0)
    u = times[:, None] * np.array([0.02, -0.01, 0.03])
    v = times[:, None] * np.array([0.3, -0.2, 0.1])
    poses[:, :3, 3] += u @ anchor[:3, :3].T
    poses[:, :3, :3] = anchor[:3, :3] @ Rotation.from_rotvec(v).as_matrix()
    return anchor, poses, np.linspace(1.2, -0.1, count), times


def test_identity_copies_pose_dtype_bytes_and_uniform_opening_clip_without_aliasing():
    anchor, poses, openings, times = references()
    poses = poses.astype(np.float32)
    original = poses.tobytes()
    output, opening, detail = GeneratedReferenceCalibration().apply(anchor, 0.8, poses, openings, times)
    assert output.dtype == poses.dtype and output.tobytes() == original
    np.testing.assert_array_equal(opening, np.clip(openings, 0, 1))
    np.testing.assert_array_equal(detail["raw_openings"], openings)
    assert detail["pose_identity_weights"] and not np.shares_memory(output, poses)
    output[0, :3, 3] = 0
    opening[0] = 0
    assert poses.tobytes() == original and openings[0] == 1.2


def test_anchor_body_features_and_noncommuting_rotation_vector_addition():
    anchor, poses, openings, times = references()
    weights = np.zeros((7, 6))
    weights[-1] = [0.02, -0.03, 0.01, 0.1, 0.08, -0.05]
    model = GeneratedReferenceCalibration(weights, [0, 0])
    output, _, detail = model.apply(anchor, 0.8, poses, openings, times)
    features = reference_features(anchor, 0.8, poses, openings, times)
    u, v = features["local_position_m"], features["anchor_relative_rotation_vector_rad"]
    dp = times[:, None] / 2 * weights[-1, :3] * 0.1
    dv = times[:, None] / 2 * weights[-1, 3:] * 0.5
    expected_r = anchor[:3, :3] @ Rotation.from_rotvec(v + dv).as_matrix()
    np.testing.assert_allclose(output[:, :3, :3], expected_r, atol=2e-15, rtol=0)
    np.testing.assert_allclose(output[:, :3, 3], anchor[:3, 3] + (u + dp) @ anchor[:3, :3].T, atol=1e-15)
    assert not np.allclose(expected_r[-1], poses[-1, :3, :3] @ Rotation.from_rotvec(dv[-1]).as_matrix(), atol=1e-6)
    np.testing.assert_array_equal(detail["pose_features"][:, -1], times / 2)
    np.testing.assert_allclose(detail["pose_features"][:, :3], u / 0.1)
    np.testing.assert_allclose(detail["pose_features"][:, 3:6], v / 0.5)


def test_world_basis_change_and_generated_prefix_do_not_change_local_correction():
    anchor, poses, openings, times = references()
    weights = np.arange(42).reshape(7, 6) / 200
    model = GeneratedReferenceCalibration(weights, (0.2, -0.1))
    original, original_opening, _ = model.apply(anchor, 0.8, poses, openings, times)
    world = np.eye(4)
    world[:3, :3] = Rotation.from_euler("xyz", [-0.7, 0.2, 0.8]).as_matrix()
    world[:3, 3] = [-0.3, 0.1, 0.2]
    changed, changed_opening, _ = model.apply(world @ anchor, 0.8, world @ poses, openings, times)
    np.testing.assert_allclose(changed, world @ original, atol=2e-15, rtol=0)
    np.testing.assert_array_equal(changed_opening, original_opening)
    prefix, prefix_opening, _ = model.apply(anchor, 0.8, poses[:8], openings[:8], times[:8])
    np.testing.assert_array_equal(prefix, original[:8])
    np.testing.assert_array_equal(prefix_opening, original_opening[:8])
    poses[8:, :3, 3] += 0.1
    openings[8:] = 0.3
    changed, changed_opening, _ = model.apply(anchor, 0.8, poses, openings, times)
    np.testing.assert_array_equal(changed[:8], prefix)
    np.testing.assert_array_equal(changed_opening[:8], prefix_opening)


def test_endpoint_norm_caps_and_raw_opening_features_precede_final_clip():
    anchor, poses, openings, times = references()
    model = GeneratedReferenceCalibration(np.ones((7, 6)) * 100, (100, 100))
    _, result, detail = model.apply(anchor, 0.8, poses, openings, times)
    np.testing.assert_allclose(np.linalg.norm(detail["translation_correction_m"], axis=1), 0.05 * times, atol=1e-16)
    np.testing.assert_allclose(
        np.linalg.norm(detail["rotation_vector_correction_rad"], axis=1), 0.25 * times, atol=2e-16
    )
    assert np.all(np.abs(detail["opening_correction"]) <= 0.25 * times)
    np.testing.assert_array_equal(detail["opening_features"][:, 0], openings - 0.8)
    assert detail["opening_features"][0, 0] > 0.2
    assert np.all((result >= 0) & (result <= 1))
    assert detail["translation_cap_active"].all() and detail["rotation_cap_active"].all()


def test_weighted_ridge_cross_channels_matches_augmented_least_squares_and_zero_prior():
    rng = np.random.default_rng(42)
    x, xg = rng.normal(size=(30, 7)), rng.normal(size=(30, 2))
    residual, opening = rng.normal(size=(30, 6)), rng.normal(size=30)
    weights = np.arange(1, 31, dtype=float)
    weights /= weights.sum()
    model = GeneratedReferenceCalibration.fit(x, residual, xg, opening, weights)
    pose_design = np.vstack((np.sqrt(weights)[:, None] * x, np.sqrt(0.1) * np.eye(7)))
    opening_design = np.vstack((np.sqrt(weights)[:, None] * xg, np.sqrt(0.1) * np.eye(2)))
    expected = np.linalg.lstsq(
        pose_design, np.vstack((np.sqrt(weights)[:, None] * residual, np.zeros((7, 6)))), rcond=None
    )[0]
    expected_opening = np.linalg.lstsq(opening_design, np.r_[np.sqrt(weights) * opening, np.zeros(2)], rcond=None)[0]
    # Independent normal-equation and SVD solvers differ by float64 roundoff;
    # this is a geometry unit-test tolerance, never a promotion-score margin.
    np.testing.assert_allclose(model.pose_weights, expected, atol=2e-14, rtol=0)
    np.testing.assert_allclose(model.opening_weights, expected_opening, atol=2e-14, rtol=0)
    zero = GeneratedReferenceCalibration.fit(x, np.zeros_like(residual), xg, np.zeros(30), weights)
    np.testing.assert_array_equal(zero.pose_weights, np.zeros((7, 6)))
    repeated = GeneratedReferenceCalibration.fit(
        np.tile(x, (2, 1)), np.tile(residual, (2, 1)), np.tile(xg, (2, 1)), np.tile(opening, 2), np.tile(weights / 2, 2)
    )
    np.testing.assert_allclose(repeated.pose_weights, model.pose_weights, atol=5e-16, rtol=0)


def test_original_and_corrected_pi_branches_are_rejected_without_hidden_fallback():
    anchor, poses, openings, times = references(1)
    poses[0, :3, :3] = anchor[:3, :3] @ Rotation.from_rotvec([np.pi - 0.5e-6, 0, 0]).as_matrix()
    with pytest.raises(ValueError, match="near pi"):
        GeneratedReferenceCalibration().apply(anchor, 0.8, poses, openings, times)
    poses[0, :3, :3] = anchor[:3, :3] @ Rotation.from_rotvec([np.pi - 2e-6, 0, 0]).as_matrix()
    weights = np.zeros((7, 6))
    weights[-1, 3] = 0.1
    with pytest.raises(ValueError, match="corrected.*pi"):
        GeneratedReferenceCalibration(weights).apply(anchor, 0.8, poses, openings, times)


def test_bad_clock_shape_nonfinite_weights_and_mutable_parameter_inputs_are_rejected():
    anchor, poses, openings, times = references()
    for clock in (times[::-1], times - 0.05, times * 2, times[:-1]):
        with pytest.raises(ValueError):
            GeneratedReferenceCalibration().apply(anchor, 0.8, poses, openings, clock)
    weights = np.zeros((7, 6))
    model = GeneratedReferenceCalibration(weights)
    weights[:] = 1
    np.testing.assert_array_equal(model.pose_weights, 0)
    for bad in (np.ones((6, 7)), np.full((7, 6), np.nan)):
        with pytest.raises(ValueError):
            GeneratedReferenceCalibration(bad)
    for bad in ([1, 1], [-1, 2], [np.nan, 1]):
        with pytest.raises(ValueError):
            GeneratedReferenceCalibration.fit(np.zeros((2, 7)), np.zeros((2, 6)), np.zeros((2, 2)), np.zeros(2), bad)
    with pytest.raises(ValueError):
        CalibrationConfig(control_fps=True)
