"""A reference lead uses only the already-generated opening horizon."""

import numpy as np
import pytest

from s2a.references.opening_lead import opening_lead_h40, score_opening_against_references
from scripts.simulation.runtime.run_bowl_h40_opening_lead import checker_window


def reference():
    poses = np.repeat(np.eye(4)[None], 40, axis=0)
    poses[:, 0, 3] = np.arange(40) / 100
    return poses, np.linspace(1, 0, 40)


@pytest.mark.parametrize("consumed", [0, 8, 16, 24, 32])
def test_lead_preserves_poses_and_inputs_and_marks_all_terminal_holds(consumed):
    poses, opening = reference()
    original_poses, original_opening = poses.copy(), opening.copy()
    pose_window, shifted, info = opening_lead_h40(poses, opening, consumed=consumed, lead_steps=4)
    indices = np.minimum(consumed + np.arange(40) + 4, 39)
    np.testing.assert_array_equal(shifted, opening[indices])
    np.testing.assert_array_equal(pose_window, poses[np.minimum(consumed + np.arange(40), 39)])
    np.testing.assert_array_equal(poses, original_poses)
    np.testing.assert_array_equal(opening, original_opening)
    assert info["generated_endpoints"] == 40 - consumed
    assert info["opening_generated_endpoints"] == 36 - consumed
    assert info["opening_terminal_hold_endpoints"] == consumed + 4
    independent_pose, independent_opening, independent_info = checker_window(poses, opening, consumed=consumed)
    np.testing.assert_array_equal(independent_pose, pose_window)
    np.testing.assert_array_equal(independent_opening, shifted)
    assert independent_info == info


@pytest.mark.parametrize("lead", [-1, 41, 1.5, True])
def test_rejects_invalid_lead(lead):
    poses, opening = reference()
    with pytest.raises(ValueError, match="Opening lead"):
        opening_lead_h40(poses, opening, consumed=0, lead_steps=lead)


def test_score_reports_separate_original_and_shifted_errors():
    result = score_opening_against_references(0.6, 0.8, 0.5)
    assert result["original_error"] == pytest.approx(-0.2)
    assert result["lead_error"] == pytest.approx(0.1)
    with pytest.raises(ValueError, match="finite values"):
        score_opening_against_references(float("nan"), 0.8, 0.5)
