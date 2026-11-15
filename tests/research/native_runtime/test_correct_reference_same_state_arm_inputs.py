"""A matched-state action comparison must retain the original interface limits."""

import numpy as np
import pytest

from scripts.simulation.runtime.prepare_correct_reference_same_state_arm import fixed_actions, recorded_prefix


def test_action_swap_preserves_gripper_and_rejects_rate_violation():
    candidate = np.zeros((8, 7))
    baseline = candidate.copy()
    baseline[3:7, 0] = 0.1
    actions, previous, rates = fixed_actions(candidate, baseline, start=3, count=4, rate_limit=0.25)
    np.testing.assert_array_equal(actions["candidate_actions"][:, 6], actions["original_arm"][:, 6])
    assert rates["original_arm"] == pytest.approx(0.1)
    np.testing.assert_array_equal(previous, candidate[2])
    baseline[3, 0] = 0.3
    with pytest.raises(ValueError, match="rate"):
        fixed_actions(candidate, baseline, start=3, count=4, rate_limit=0.25)


def test_action_swap_rejects_a_different_gripper_command():
    candidate = np.zeros((8, 7))
    baseline = candidate.copy()
    baseline[4, 6] = 1.0
    with pytest.raises(ValueError, match="gripper"):
        fixed_actions(candidate, baseline, start=3, count=4, rate_limit=0.25)


def test_prefix_selection_refuses_a_stopped_or_partial_segment():
    segments = [
        dict(start_step=i, boundary="window_end", continuation=dict(steps_completed=8, stop=None)) for i in (0, 8)
    ]
    assert len(recorded_prefix(segments, 16)["segments"]) == 2
    segments[1]["continuation"]["steps_completed"] = 7
    with pytest.raises(ValueError, match="complete"):
        recorded_prefix(segments, 16)
