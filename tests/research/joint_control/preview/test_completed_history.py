"""Restored histories must represent a completed action and one current endpoint."""

import numpy as np
import pytest

from s2a.research.joint_control.preview.completed_history import (
    CompletedHistory,
    recompute_completed_innovation,
    restore_completed_history,
)


def completed():
    plan = np.array([[0.1, 0, 0, 0, 0, 0], [0.2, 0, 0, 0, 0, 0]])
    return CompletedHistory(np.r_[plan[0], 0.0], np.arange(12), -np.arange(12), plan, 3)


def test_restoration_uses_distinct_copies_and_shifted_completed_plan():
    past = completed()
    coupled, tracking = restore_completed_history(past, np.eye(4), np.zeros(6))
    np.testing.assert_array_equal(coupled.current(np.eye(4), np.zeros(6)), np.arange(12))
    np.testing.assert_array_equal(tracking.history.current(np.eye(4), np.zeros(6)), -np.arange(12))
    np.testing.assert_array_equal(tracking.history.shifted(2), np.tile(past.last_plan[1], (2, 1)))
    assert coupled.completed_periods == tracking.history.completed_periods == 3
    assert tracking.phase == "ready" and tracking.history.pending_plan is None and coupled.pending is None
    tracking.history.last_plan[0, 0] = 0.9
    assert past.last_plan[0, 0] == 0.1
    assert not past.last_plan.flags.writeable


def test_history_rejects_unexecuted_first_command_and_infeasible_plan():
    past = completed()
    plan = past.last_plan.copy()
    plan[0, 0] = 0.11
    with pytest.raises(ValueError, match="actual previous command"):
        CompletedHistory(past.previous_command, past.coupled_innovation, past.tracking_innovation, plan, 3)
    plan = past.last_plan.copy()
    plan[1, 0] = 0.9
    with pytest.raises(ValueError, match="rate"):
        CompletedHistory(past.previous_command, past.coupled_innovation, past.tracking_innovation, plan, 3)


def test_restored_history_rejects_a_different_current_endpoint():
    coupled, tracking = restore_completed_history(completed(), np.eye(4), np.zeros(6))
    moved = np.eye(4)
    moved[0, 3] = 0.1
    with pytest.raises(ValueError, match="last completed"):
        coupled.current(moved, np.zeros(6))
    with pytest.raises(ValueError, match="last completed"):
        tracking.history.current(moved, np.zeros(6))


def test_completed_innovation_uses_measured_endpoint_and_prior_prediction():
    previous = np.eye(4)
    actual = np.eye(4)
    actual[0, 3] = 0.03
    prediction = np.zeros(12)
    prediction[0] = 0.01
    prediction[6] = 0.1
    twist = np.zeros(6)
    twist[0] = 0.3
    innovation = recompute_completed_innovation(previous, np.zeros(6), prediction, actual, twist)
    np.testing.assert_allclose(innovation[0], 0.02, atol=1e-12, rtol=0)
    np.testing.assert_allclose(innovation[6], 0.01, atol=1e-12, rtol=0)
