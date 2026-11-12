"""Amplification uses only the generated horizon and marks terminal holds."""

import numpy as np
import pytest

from s2a.references.opening_intent import amplified_opening_h40
from scripts.simulation.runtime.run_bowl_h40_closing_intent import checker_window


@pytest.mark.parametrize("consumed", [0, 8, 16, 24, 32])
def test_modified_opening_is_causal_bounded_and_independently_rebuilt(consumed):
    poses = np.repeat(np.eye(4)[None], 40, axis=0)
    poses[:, 0, 3] = np.arange(40) / 100
    opening = np.linspace(1, 0, 40)
    saved_poses, saved_opening = poses.copy(), opening.copy()
    pose_window, actual, info = amplified_opening_h40(poses, opening, consumed=consumed, lead_steps=8, gain=3)
    indices = np.minimum(consumed + np.arange(40) + 8, 39)
    np.testing.assert_allclose(actual, np.clip(3 * opening[indices] - 2, 0, 1), atol=1e-15, rtol=0)
    np.testing.assert_array_equal(pose_window, poses[np.minimum(consumed + np.arange(40), 39)])
    np.testing.assert_array_equal(poses, saved_poses)
    np.testing.assert_array_equal(opening, saved_opening)
    assert info["opening_generated_endpoints"] == 32 - consumed
    assert info["opening_terminal_hold_endpoints"] == consumed + 8
    check_pose, check_opening, check_info = checker_window(poses, opening, consumed=consumed)
    np.testing.assert_array_equal(check_pose, pose_window)
    np.testing.assert_allclose(check_opening, actual, atol=1e-15, rtol=0)
    assert check_info == info


@pytest.mark.parametrize("lead,gain", [(-1, 3), (41, 3), (True, 3), (8, 0.9), (8, float("nan"))])
def test_invalid_transform_parameters_are_rejected(lead, gain):
    with pytest.raises(ValueError, match="bounded integer lead"):
        amplified_opening_h40(
            np.repeat(np.eye(4)[None], 40, axis=0), np.ones(40), consumed=0, lead_steps=lead, gain=gain
        )
