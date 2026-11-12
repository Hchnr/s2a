"""Separate completed prefixes from generated, unexecuted closing intent."""

import numpy as np
import pytest

from scripts.wam.tracking.references.audit_consumed_gripper_intent import window_summary


def test_partial_segment_does_not_claim_later_generated_closure_was_consumed():
    openings = np.full(40, 0.95)
    openings[2] = 0.70
    openings[9] = 0.2
    result = window_summary(openings, consumed=2, anchor=0.95)
    assert result["all"]["absolute_close_count"] == 1 and result["all"]["first_absolute_close_index"] == 9
    assert result["consumed"]["absolute_close_count"] == result["consumed"]["anchor_drop_count"] == 0
    assert result["unconsumed"]["first_anchor_drop_index"] == 2
    assert result["unconsumed"]["first_absolute_close_index"] == 9
    assert result["consumed"]["endpoints"] == 2 and result["unconsumed"]["endpoints"] == 38


@pytest.mark.parametrize("consumed", [-1, 9, True])
def test_rejects_unregistered_consumption_count(consumed):
    with pytest.raises(ValueError, match="at most eight consumed"):
        window_summary(np.full(40, 0.9), consumed=consumed, anchor=0.9)
