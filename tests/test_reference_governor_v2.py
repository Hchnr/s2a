"""A reference can retreat from an unreachable target without advancing a plan."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from s2a.geometry import pose_error, transform
from s2a.reference_governor import ReferenceGovernor
from s2a.reference_governor_v2 import RecoveringReferenceGovernor


def primed(kind, *, rotation=0):
    governor = kind()
    start = np.eye(4)
    end = transform([0.04, 0, 0], Rotation.from_rotvec([0, 0, rotation]).as_matrix())
    governor.set_reference(start, end[None], [0.2], anchor_opening=0.2, model_dt=0.05)
    measured = start
    for _ in range(20):
        result = governor.advance(measured, 0.05)
        measured = result["target_tcp"]
    assert result["complete"]
    return governor, end


def test_bridge_can_retreat_when_original_clock_is_stuck_at_hard_stop():
    original, old = primed(ReferenceGovernor)
    recovering, _ = primed(RecoveringReferenceGovernor)
    anchor = np.eye(4)
    target = transform([0.01, 0, 0], np.eye(3))
    for governor in (original, recovering):
        governor.set_reference(anchor, target[None], [0.1], anchor_opening=0.7, model_dt=0.05)
    stuck = original.advance(anchor, 0.05)
    fixed = recovering.advance(anchor, 0.05)
    assert stuck["tracking_scale"] == 0 and stuck["model_time_s"] == 0
    np.testing.assert_allclose(stuck["target_tcp"], old, atol=1e-15)
    assert fixed["nominal_tracking_scale"] == 0 and fixed["bridge_recovery_advanced"]
    assert fixed["model_time_s"] == 0
    assert fixed["target_tcp"][0, 3] == pytest.approx(0.03)
    assert fixed["opening"] == pytest.approx(0.2)
    assert np.linalg.norm(pose_error(fixed["target_tcp"], old)[:3]) <= 0.2 * 0.05 + 1e-12


def test_recovery_does_not_advance_the_generated_path_without_a_new_bridge():
    governor, old = primed(RecoveringReferenceGovernor)
    result = governor.advance(np.eye(4), 0.05)
    assert not result["bridge_recovery_advanced"]
    np.testing.assert_allclose(result["target_tcp"], old, atol=1e-15)


def test_bridge_must_not_move_farther_from_the_current_robot_position():
    governor, old = primed(RecoveringReferenceGovernor)
    governor.set_reference(np.eye(4), np.eye(4)[None], [0.1], anchor_opening=0.7, model_dt=0.05)
    robot = transform([0.1, 0, 0], np.eye(3))
    result = governor.advance(robot, 0.05)
    assert result["nominal_tracking_scale"] == 0 and not result["bridge_recovery_advanced"]
    np.testing.assert_allclose(result["target_tcp"], old, atol=1e-15)


def test_bridge_cannot_trade_better_position_for_worse_orientation():
    governor, old = primed(RecoveringReferenceGovernor, rotation=0.4)
    anchor = transform([0, 0, 0], Rotation.from_rotvec([0, 0, 0.8]).as_matrix())
    governor.set_reference(anchor, anchor[None], [0.1], anchor_opening=0.7, model_dt=0.05)
    result = governor.advance(np.eye(4), 0.05)
    assert not result["bridge_recovery_advanced"]
    np.testing.assert_allclose(result["target_tcp"], old, atol=1e-15)


def test_recovered_bridge_then_generated_opening_share_speed_limited_clock():
    governor, old = primed(RecoveringReferenceGovernor)
    anchor = np.eye(4)
    governor.set_reference(anchor, anchor[None], [0.0], anchor_opening=0.7, model_dt=0.05)
    previous, opening = old, 0.2
    first = governor.advance(anchor, 0.05)
    measured = first["target_tcp"]
    for result in [first] + [None] * 9:
        if result is None:
            result = governor.advance(measured, 0.05)
        error = pose_error(result["target_tcp"], previous)
        assert np.linalg.norm(error[:3]) <= 0.2 * 0.05 + 1e-12
        assert abs(result["opening"] - opening) <= 2 * 0.05 + 1e-12
        if result["bridging"]:
            assert result["opening"] == pytest.approx(0.2)
        previous = measured = result["target_tcp"]
        opening = result["opening"]
    assert result["complete"] and result["opening"] == 0
