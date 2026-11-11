"""An empty reset can carry observed history across reference segments."""

import numpy as np
import pytest

from s2a.research.joint_control.preview.segmented_reset_history import make_segmented_reset_history


def test_reset_view_is_the_refiner_history_and_remains_unknown_until_observed():
    captured = []

    def factory(inner):
        captured.append(inner)
        return lambda *args, **kwargs: None

    a = make_segmented_reset_history(np.eye(4), np.zeros(6), refiner_factory=factory)
    b = make_segmented_reset_history(np.eye(4), np.zeros(6), refiner_factory=factory)
    assert captured == [a.inner_error, b.inner_error]
    assert a.owner is not b.owner and a.inner_error is not b.inner_error
    assert a.inner_error.snapshot()["available"] is False
    assert a.tracking.history.last_plan is None
    assert a.coupled.completed_periods == a.tracking.history.completed_periods == 0
    np.testing.assert_array_equal(a.previous_command, np.zeros(7))
    assert not a.previous_command.flags.writeable
    predicted = np.zeros((25, 7))
    a.inner_error.stage(0, predicted)
    assert a.inner_error.snapshot()["available"] is False
    with pytest.raises(ValueError, match="preceding prediction"):
        a.owner.prepare_segment(1)
    a.inner_error.complete(0, predicted + 0.02)
    assert a.inner_error.snapshot()["available"] is True
    assert b.inner_error.snapshot()["available"] is False


def test_completed_inner_profile_survives_segment_switch_and_old_view_expires():
    history = make_segmented_reset_history(np.eye(4), np.zeros(6), refiner_factory=lambda _inner: lambda: None)
    for step in range(8):
        prediction = np.zeros((25, 7))
        history.inner_error.stage(step, prediction)
        history.inner_error.complete(step, prediction + (step + 1) * 0.001)
    before = history.owner.total_snapshot()
    next_view = history.owner.begin_segment(8)
    after = next_view.snapshot()
    assert before["completed_periods"] == after["total_completed_periods"] == 8
    assert after["completed_periods"] == 0 and after["last_completed_step"] == 7
    np.testing.assert_allclose(after["inner_position_error_rad"], before["inner_position_error_rad"])
    np.testing.assert_allclose(after["inner_position_error_rad"], 0.008)
    with pytest.raises(ValueError, match="expired"):
        history.inner_error.snapshot()
    with pytest.raises(ValueError, match="next absolute"):
        next_view.stage(9, np.zeros((25, 7)))
    next_view.stage(8, np.zeros((25, 7)))
    next_view.complete(8, np.zeros((25, 7)))
    assert next_view.snapshot()["completed_periods"] == 1
    assert history.owner.total_snapshot()["completed_periods"] == 9
