"""Closed-form geometry checks for saved-trajectory phase diagnostics."""

import importlib
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def diagnostic(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("diagnose_tracking_v5_saved_trajectory_phase")


def z_pose(position=(0.0, 0.0, 0.0), angle=0.0):
    pose = np.eye(4)
    cosine, sine = np.cos(angle), np.sin(angle)
    pose[:3, :3] = [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]]
    pose[:3, 3] = position
    return pose


def straight_reference():
    """A 20 mm/period straight path with phase equal to x / 0.02."""
    anchor = z_pose((-0.02, 0.0, 0.0))
    targets = np.stack([z_pose((0.02 * phase, 0.0, 0.0)) for phase in range(12)])
    return anchor, targets


def test_fixed_grid_world_translation_and_shortest_rotation_across_pi(diagnostic):
    anchor = z_pose((1.0, 2.0, 3.0), np.deg2rad(170.0))
    target = z_pose((1.01, 2.02, 3.03), np.deg2rad(-170.0))
    targets = np.repeat(target[None], 12, axis=0)

    phases, poses, lipschitz = diagnostic.interpolate_phase_grid(anchor, targets)

    assert phases.shape == (12001,)
    assert poses.shape == (12001, 4, 4)
    assert lipschitz.shape == (12,)
    assert phases[0] == -1.0
    assert phases[-1] == 11.0
    np.testing.assert_allclose(np.diff(phases), 0.001, atol=2e-15, rtol=0)
    np.testing.assert_allclose(poses[0], anchor, atol=1e-14, rtol=0)
    np.testing.assert_allclose(poses[1000 * np.arange(1, 13)], targets, atol=1e-14, rtol=0)
    assert phases[500] == -0.5
    np.testing.assert_allclose(poses[500], z_pose((1.005, 2.01, 3.015), np.pi), atol=1e-14, rtol=0)
    # The shortest arc is 20 degrees, whose normalized length dominates
    # the 37.4 mm translation; a long 340-degree arc would fail this check.
    assert lipschitz[0] == pytest.approx(np.deg2rad(20.0) / 0.05, abs=1e-12)
    np.testing.assert_allclose(lipschitz[1:], 0.0, atol=1e-12, rtol=0)


def test_constant_reference_uses_earliest_exact_tie(diagnostic):
    targets = np.repeat(np.eye(4)[None], 12, axis=0)
    result = diagnostic.diagnose_saved_tcp(targets.copy(), np.eye(4), targets)

    assert len(result["endpoints"]) == 12
    assert result["best_phases_monotone"]
    assert result["phase_decreases"] == []
    for index, endpoint in enumerate(result["endpoints"]):
        assert endpoint["timed_phase"] == index
        assert endpoint["best_phase"] == -1.0
        assert endpoint["lag_periods"] == index + 1
        assert endpoint["lag_seconds"] == pytest.approx(0.05 * (index + 1))
        assert endpoint["timed_score"] == 0.0
        assert endpoint["best_score"] == 0.0
        assert endpoint["timed_passed"]
        assert endpoint["best_grid_passed"]
        assert endpoint["continuous_score_lower_bound"] == 0.0
        assert not endpoint["continuous_path_caps_excluded"]


@pytest.mark.parametrize(
    ("position_error", "rotation_error", "expected_score", "passed"),
    [(0.006, 0.04, 0.64, True), (0.011, 0.04, 1.21, False), (0.006, 0.055, 1.21, False), (0.01, 0.0, 1.0, True)],
)
def test_score_uses_worst_normalized_cap_in_metres_and_radians(
    diagnostic, position_error, rotation_error, expected_score, passed
):
    targets = np.repeat(np.eye(4)[None], 12, axis=0)
    predicted = np.repeat(z_pose((position_error, 0.0, 0.0), rotation_error)[None], 12, axis=0)
    endpoint = diagnostic.diagnose_saved_tcp(predicted, np.eye(4), targets)["endpoints"][0]

    for prefix in ("timed", "best"):
        assert endpoint[f"{prefix}_position_error_m"] == pytest.approx(position_error, abs=1e-14)
        assert endpoint[f"{prefix}_rotation_error_rad"] == pytest.approx(rotation_error, abs=1e-14)
        assert endpoint[f"{prefix}_score"] == pytest.approx(expected_score, abs=1e-12)
    assert bool(endpoint["timed_passed"]) is passed
    assert bool(endpoint["best_grid_passed"]) is passed


def test_translation_lag_has_known_phase_and_seconds(diagnostic):
    anchor, targets = straight_reference()
    predicted = targets.copy()
    predicted[:, 0, 3] -= 0.008
    result = diagnostic.diagnose_saved_tcp(predicted, anchor, targets)

    assert result["best_phases_monotone"]
    assert result["phase_decreases"] == []
    for index, endpoint in enumerate(result["endpoints"]):
        assert endpoint["best_phase"] == pytest.approx(index - 0.4, abs=1e-12)
        assert endpoint["lag_periods"] == pytest.approx(0.4, abs=1e-12)
        assert endpoint["lag_seconds"] == pytest.approx(0.02, abs=1e-12)
        assert endpoint["timed_position_error_m"] == pytest.approx(0.008, abs=1e-14)
        assert endpoint["timed_score"] == pytest.approx(0.64, abs=1e-12)
        assert endpoint["best_position_error_m"] < 1e-14
        assert endpoint["best_rotation_error_rad"] == 0.0
        assert endpoint["best_score"] < 1e-24
        assert endpoint["timed_passed"]
        assert endpoint["best_grid_passed"]
        assert not endpoint["continuous_path_caps_excluded"]


def test_perpendicular_offset_can_exclude_the_entire_continuous_path(diagnostic):
    anchor, targets = straight_reference()
    predicted = targets.copy()
    predicted[:, 1, 3] = 0.012
    result = diagnostic.diagnose_saved_tcp(predicted, anchor, targets)

    for index, endpoint in enumerate(result["endpoints"]):
        assert endpoint["best_phase"] == pytest.approx(index, abs=1e-12)
        assert endpoint["best_score"] == pytest.approx(1.44, abs=1e-12)
        # Every continuous path point has at least 12 mm lateral error.
        # The certificate must stay below that exact optimum yet exceed 1.
        assert 1.0 < endpoint["continuous_score_lower_bound"] <= 1.44
        assert not endpoint["best_grid_passed"]
        assert endpoint["continuous_path_caps_excluded"]


def test_failed_grid_samples_do_not_exclude_a_feasible_between_sample_point(diagnostic):
    anchor, targets = straight_reference()
    # At phase 0.0005 the exact continuous error is 10 mm. Adjacent grid
    # samples additionally differ by 10 micrometres in x, so both fail.
    predicted = np.repeat(z_pose((0.00001, 0.01, 0.0))[None], 12, axis=0)
    endpoint = diagnostic.diagnose_saved_tcp(predicted, anchor, targets)["endpoints"][0]

    assert endpoint["best_score"] == pytest.approx(1.000001, abs=1e-12)
    assert not endpoint["best_grid_passed"]
    assert 0.0 <= endpoint["continuous_score_lower_bound"] <= 1.0
    assert not endpoint["continuous_path_caps_excluded"]


@pytest.mark.parametrize(("score_excess", "excluded"), [(5e-11, False), (2e-10, True)])
def test_continuous_exclusion_keeps_its_numerical_margin(diagnostic, score_excess, excluded):
    targets = np.repeat(np.eye(4)[None], 12, axis=0)
    predicted = np.repeat(z_pose((0.01 * np.sqrt(1.0 + score_excess), 0.0, 0.0))[None], 12, axis=0)
    endpoint = diagnostic.diagnose_saved_tcp(predicted, np.eye(4), targets)["endpoints"][0]

    assert not endpoint["best_grid_passed"]
    assert endpoint["continuous_score_lower_bound"] == pytest.approx(1.0 + score_excess, abs=1e-14)
    assert bool(endpoint["continuous_path_caps_excluded"]) is excluded


def test_nonmonotone_best_phases_are_reported_without_reordering(diagnostic):
    anchor, targets = straight_reference()
    phases = np.array([0, 2, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    predicted = targets[phases].copy()
    result = diagnostic.diagnose_saved_tcp(predicted, anchor, targets)

    np.testing.assert_allclose([row["best_phase"] for row in result["endpoints"]], phases, atol=1e-12, rtol=0)
    assert not result["best_phases_monotone"]
    assert len(result["phase_decreases"]) == 1
    assert result["endpoints"][1]["lag_periods"] == pytest.approx(-1.0)
    assert result["endpoints"][2]["lag_periods"] == pytest.approx(1.0)
    np.testing.assert_array_equal(predicted, targets[phases])


def test_broad_transform_validity_does_not_bypass_narrow_so3_projection_gate(diagnostic):
    targets = np.repeat(np.eye(4)[None], 12, axis=0)
    predicted = targets.copy()
    predicted[0, 0, 0] += 1e-9
    # Existing SE(3) input admission accepts this near rotation, but a
    # continuous SO(3) certificate cannot silently absorb its larger change.
    diagnostic.validate_transform(predicted)
    with pytest.raises(ValueError, match="SO\\(3\\) projection exceeds 1e-12 gate"):
        diagnostic.diagnose_saved_tcp(predicted, np.eye(4), targets)
    assert predicted[0, 0, 0] == 1.0 + 1e-9
