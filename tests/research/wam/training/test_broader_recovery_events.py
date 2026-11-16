"""Offline event candidates retain timing and actual-window boundaries."""

import numpy as np
import pytest

from s2a.research.wam.training.broader_recovery_events import event_candidates
from s2a.research.wam.training.recovery_event_coverage import event_coverage


def test_first_low_and_success_events_produce_separate_actual_candidates():
    opening = np.full(181, 0.8)
    success = np.zeros(181, dtype=bool)
    opening[101:104] = [0.34, 0.2, 0.35]
    success[151:] = True
    result = event_candidates(opening, success)
    assert result["low_opening"] == dict(total=2, covered=0, first_action_index=100, last_action_index=101)
    assert result["native_success"] == dict(total=30, covered=0, first_action_index=150, last_action_index=179)
    assert [(row["anchor_step"], row["events"][0]["actual_lead_periods"]) for row in result["candidates"]] == [
        (93, 8),
        (140, 11),
    ]
    assert all(
        row["selection_uses_future_training_labels"] and not row["current_image_available"]
        for row in result["candidates"]
    )
    prior = event_coverage(opening, success)
    assert result["low_opening"]["total"] == prior["low_opening_total"]
    assert result["candidates"][0]["anchor_step"] == prior["selected_anchor"]["anchor_step"]


def test_same_anchor_deduplicates_reasons_and_existing_image_is_reused():
    opening = np.full(121, 0.8)
    success = np.zeros(121, dtype=bool)
    opening[24:] = 0.3
    success[24:] = True
    result = event_candidates(opening, success)
    assert len(result["candidates"]) == 1
    candidate = result["candidates"][0]
    assert candidate["anchor_step"] == 16 and candidate["current_image_available"] is True
    assert [event["kind"] for event in candidate["events"]] == ["low_opening", "native_success"]
    assert all(event["actual_lead_periods"] == 8 for event in candidate["events"])


def test_absent_and_reset_events_are_not_invented():
    opening = np.full(121, 0.8)
    success = np.zeros(121, dtype=bool)
    success[0] = True
    result = event_candidates(opening, success)
    assert result["reset_success"] is True and result["candidates"] == []
    assert result["low_opening"]["first_action_index"] is None
    with pytest.raises(ValueError, match="boolean"):
        event_candidates(opening, success.astype(int))
    with pytest.raises(ValueError, match="finite event"):
        event_candidates(opening, success, threshold=np.nan)
