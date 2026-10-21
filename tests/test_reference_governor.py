"""Geometry, physical retiming, feedback causality and chunk continuity."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from s2a.geometry import pose_error, transform
from s2a.reference_governor import ReferenceGovernor, ReferenceGovernorConfig


def test_known_motion_shares_clock_and_respects_three_physical_bounds():
    anchor = np.eye(4)
    target = transform([0.2, 0, 0], Rotation.from_rotvec([0, 0, 1.0]).as_matrix())
    governor = ReferenceGovernor()
    summary = governor.set_reference(anchor, target[None], [0.0], anchor_opening=1.0, model_dt=0.05)
    assert summary["retimed_duration_s"] == 1.0
    measured, previous, opening = anchor, anchor, 1.0
    for step in range(20):
        result = governor.advance(measured, 0.05)
        error = pose_error(result["target_tcp"], previous)
        assert np.linalg.norm(error[:3]) <= 0.2 * 0.05 + 1e-12
        assert np.linalg.norm(error[3:]) <= 1.0 * 0.05 + 1e-12
        assert abs(result["opening"] - opening) <= 2 * 0.05 + 1e-12
        assert result["model_time_s"] == pytest.approx((step + 1) * 0.05 / 20)
        measured = previous = result["target_tcp"]
        opening = result["opening"]
    assert result["complete"]
    np.testing.assert_allclose(result["target_tcp"], target, atol=1e-12)
    assert result["opening"] == 0


def test_tracking_pause_does_not_integrate_unexecuted_model_time():
    governor = ReferenceGovernor()
    target = transform([0.2, 0, 0], np.eye(3))
    governor.set_reference(np.eye(4), target[None], [1], anchor_opening=1, model_dt=0.05)
    first = governor.advance(np.eye(4), 0.05)
    distant = transform([-0.1, 0, 0], np.eye(3))
    paused = governor.advance(distant, 0.05)
    assert paused["tracking_scale"] == 0
    np.testing.assert_array_equal(paused["target_tcp"], first["target_tcp"])
    assert paused["model_time_s"] == first["model_time_s"]
    resumed = governor.advance(first["target_tcp"], 0.05)
    assert resumed["model_time_s"] > paused["model_time_s"]


def test_new_chunk_bridges_continuously_without_spending_model_time():
    governor = ReferenceGovernor()
    start = np.eye(4)
    governor.set_reference(start, start[None], [1], anchor_opening=1, model_dt=0.05)
    preceding = governor.advance(start, 0.05)
    anchor = transform([0.1, 0, 0], np.eye(3))
    target = transform([0.11, 0, 0], np.eye(3))
    governor.set_reference(anchor, target[None], [0], anchor_opening=0.5, model_dt=0.05)
    result = governor.advance(preceding["target_tcp"], 0.05)
    assert result["bridging"] and result["model_time_s"] == 0
    assert result["target_tcp"][0, 3] == pytest.approx(0.01)
    assert result["opening"] == pytest.approx(0.95)


def test_snapshots_inputs_and_reset_between_independent_rollouts():
    targets = np.eye(4)[None].copy()
    openings = np.array([0.5])
    governor = ReferenceGovernor()
    governor.set_reference(np.eye(4), targets, openings, anchor_opening=1, model_dt=0.05)
    targets[0, 0, 3], openings[0] = 50, 0
    result = governor.advance(np.eye(4), 0.05)
    assert result["target_tcp"][0, 3] == 0
    assert result["opening"] == pytest.approx(0.9)
    independent = ReferenceGovernor()
    independent.set_reference(np.eye(4), np.eye(4)[None], [1], anchor_opening=1, model_dt=0.05)
    assert independent.advance(np.eye(4), 0.05)["opening"] == 1


def test_shortest_rotation_crosses_pi_without_euler_or_axis_discontinuity():
    start = transform([0, 0, 0], Rotation.from_euler("z", 179, degrees=True).as_matrix())
    end = transform([0, 0, 0], Rotation.from_euler("z", -179, degrees=True).as_matrix())
    governor = ReferenceGovernor(config=ReferenceGovernorConfig(max_rotation_speed_rad_s=0.1))
    summary = governor.set_reference(start, end[None], [1], anchor_opening=1, model_dt=0.05)
    assert summary["retimed_duration_s"] == pytest.approx(np.deg2rad(2) / 0.1)
    result = governor.advance(start, 0.05)
    assert np.linalg.norm(pose_error(result["target_tcp"], start)[3:]) == pytest.approx(0.005)


def test_invalid_interval_and_reference_leave_valid_clock_unchanged():
    governor = ReferenceGovernor()
    governor.set_reference(np.eye(4), np.eye(4)[None], [1], anchor_opening=1, model_dt=0.1)
    with pytest.raises(ValueError):
        governor.advance(np.eye(4), float("nan"))
    with pytest.raises(ValueError):
        governor.set_reference(np.eye(4), np.eye(4)[None], [2], anchor_opening=1, model_dt=0.1)
    assert governor.advance(np.eye(4), 0.05)["model_time_s"] == 0.05


def test_duplicate_stationary_bridge_and_complete_hold_are_well_defined():
    governor = ReferenceGovernor()
    for _ in range(2):
        governor.set_reference(np.eye(4), np.eye(4)[None], [1], anchor_opening=1, model_dt=0.05)
        result = governor.advance(np.eye(4), 0.05)
        assert result["complete"] and not result["bridging"]
        np.testing.assert_array_equal(governor.advance(np.eye(4), 0.05)["target_tcp"], np.eye(4))
