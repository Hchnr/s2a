import numpy as np
import pytest

from s2a.research.joint_control.preview.quality_feedback_refiner import QualityFeedbackRefiner
from s2a.research.response_models.causal.causal_inner_error import CausalInnerError
from s2a.research.response_models.causal.segmented_inner_error import SegmentedInnerHistory


def test_completed_error_envelope_survives_new_segments():
    history = SegmentedInnerHistory()
    first = history.begin_segment(200)
    first.stage(200, np.zeros((25, 7)))
    actual = np.zeros((25, 7))
    actual[3, 0] = 0.3
    first.complete(200, actual)
    first.stage(201, np.zeros((25, 7)))
    actual[3, 0] = 0.1
    actual[10, 1] = 0.2
    first.complete(201, actual)
    previous = first.snapshot()
    second = history.begin_segment(202)
    rebased = second.snapshot()
    assert isinstance(second, CausalInnerError)
    assert QualityFeedbackRefiner(second, {}, start_step=202, start_cursor=0).history is second
    assert rebased["completed_periods"] == 0 and rebased["total_completed_periods"] == 2
    assert rebased["last_completed_step"] == 201 and rebased["available"]
    np.testing.assert_array_equal(rebased["inner_position_error_rad"], previous["inner_position_error_rad"])
    np.testing.assert_array_equal(rebased["position_error_rad"], previous["position_error_rad"])
    assert rebased["inner_position_error_rad"][24, 0] == pytest.approx(0.3)
    assert rebased["inner_position_error_rad"][24, 1] == pytest.approx(0.2)
    second.stage(202, np.zeros((25, 7)))
    second.complete(202, np.full((25, 7), 0.01))
    assert second.snapshot()["completed_periods"] == 1
    assert history.total_snapshot()["completed_periods"] == 3
    assert history.total_snapshot()["inner_position_error_rad"][24, 0] == pytest.approx(0.3)
    with pytest.raises(ValueError, match="expired"):
        first.snapshot()


def test_boundary_refuses_pending_prediction_and_absolute_step_gaps():
    history = SegmentedInnerHistory()
    first = history.begin_segment(5)
    with pytest.raises(ValueError, match="next absolute"):
        first.stage(6, np.zeros((25, 7)))
    first.stage(5, np.zeros((25, 7)))
    with pytest.raises(ValueError, match="physically complete"):
        history.begin_segment(6)
    with pytest.raises(ValueError, match="staged absolute"):
        first.complete(6, np.zeros((25, 7)))
    first.complete(5, np.zeros((25, 7)))
    with pytest.raises(ValueError, match="next absolute"):
        history.begin_segment(7)
    second = history.begin_segment(6)
    with pytest.raises(ValueError, match="expired"):
        first.stage(6, np.zeros((25, 7)))
    assert second.snapshot()["last_completed_step"] == 5


def test_prepared_segment_cannot_replace_history_changed_before_activation():
    history = SegmentedInnerHistory()
    first = history.begin_segment(1)
    first.stage(1, np.zeros((25, 7)))
    first.complete(1, np.zeros((25, 7)))
    pending = history.prepare_segment(2)
    assert first.snapshot()["completed_periods"] == 1
    first.stage(2, np.zeros((25, 7)))
    with pytest.raises(ValueError, match="History changed"):
        history.activate_segment(pending)
    assert first.snapshot()["completed_periods"] == 1
    first.complete(2, np.zeros((25, 7)))
    with pytest.raises(ValueError, match="History changed"):
        history.activate_segment(pending)
    assert first.snapshot()["completed_periods"] == 2
