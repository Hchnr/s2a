"""Check absolute reference timing and causal quality-stop inputs."""

import numpy as np
import pytest

from s2a.research.native_runtime.reference_choice_comparison import paired_errors, quality_stop


def test_paired_errors_compare_old_and_new_at_same_absolute_endpoint():
    old = np.tile(np.eye(4), (40, 1, 1))
    new = np.tile(np.eye(4), (40, 1, 1))
    old[:, 0, 3] = np.arange(40)
    new[:, 0, 3] = np.arange(40) + 8.5
    actual = np.eye(4)
    actual[0, 3] = 13.2
    errors = paired_errors(actual, old, new, 5)
    assert errors["old"][0] == pytest.approx(-0.2)
    assert errors["new"][0] == pytest.approx(0.3)
    with pytest.raises(ValueError, match="32-period"):
        paired_errors(actual, old, new, 32)


def test_quality_stop_exposes_only_completed_evidence():
    previous = {
        "step": 16,
        "endpoint_receipt": {"endpoint_step": 17, "private": "ignored"},
        "actual_inner": [{"q": [0.0]}] * 25,
        "decision": {"quality": {"verification": {"passed": True}}},
        "completed_feedback": {
            "policy_history": {"past_inner_error": {"last_completed_step": 16, "future_guarantee": False}}
        },
        "task_success": True,
        "target_grasp": True,
    }
    received = []

    def trigger(view, *, current_step):
        received.append(view)
        return {"replan": True, "current_step": current_step, "reason": "quality_margin_consumed"}

    result = quality_stop(previous, 17, trigger)
    assert result["stop"]["reason"] == "quality_refresh_required"
    assert "task_success" not in received[0] and "target_grasp" not in received[0]
    assert received[0]["endpoint_receipt"] == {"endpoint_step": 17}

    def available(_view, *, current_step):
        return {"replan": False, "current_step": current_step, "reason": "quality_margin_available"}

    assert quality_stop(previous, 17, available) is None

    def invalid(_view, *, current_step):
        return {"replan": 1, "current_step": current_step, "reason": "bad"}

    with pytest.raises(ValueError, match="quality signal"):
        quality_stop(previous, 17, invalid)
