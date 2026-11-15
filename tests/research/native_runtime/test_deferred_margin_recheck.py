"""A diagnostic can defer exactly one identified quality-margin signal."""

import pytest

from s2a.research.native_runtime.deferred_margin_recheck import DeferredMarginRecheck


def period(step=16):
    return {
        "step": step,
        "endpoint_receipt": {"endpoint_step": step + 1, "private": "excluded"},
        "actual_inner": [{"q": [0.0]}] * 25,
        "decision": {"quality": {"verification": {"passed": True}}},
        "completed_feedback": {
            "policy_history": {"past_inner_error": {"last_completed_step": step, "future_guarantee": False}}
        },
        "task_success": True,
        "target_grasp": True,
    }


def signal(reason, *, replan=True, received=None):
    def trigger(view, *, current_step):
        if received is not None:
            received.append(view)
        return {"replan": replan, "reason": reason, "current_step": current_step}

    return trigger


def test_only_one_targeted_margin_trigger_is_deferred():
    gate = DeferredMarginRecheck(target_step=17)
    received = []
    trigger = signal("quality_margin_consumed", received=received)
    assert gate(period(15), 16, trigger)["stop"]["reason"] == "quality_refresh_required"
    assert gate(period(), 17, trigger) is None
    assert gate(period(), 17, trigger)["stop"]["reason"] == "quality_refresh_required"
    assert gate(period(17), 18, trigger)["stop"]["reason"] == "quality_refresh_required"
    assert len(gate.events) == 1 and gate.events[0]["current_step"] == 17
    assert all("task_success" not in view and "target_grasp" not in view for view in received)
    assert all(view["endpoint_receipt"] == {"endpoint_step": view["step"] + 1} for view in received)
    altered = gate.events
    altered.clear()
    assert len(gate.events) == 1


def test_unchecked_second_point_and_available_margin_remain_unchanged():
    gate = DeferredMarginRecheck(target_step=17)
    assert gate(period(), 17, signal("no_audited_second_pose"))["stop"]["trigger"]["reason"] == "no_audited_second_pose"
    assert gate(period(), 17, signal("quality_margin_available", replan=False)) is None
    assert gate.events == []


@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_fixed_trigger_step_must_be_a_positive_integer(value):
    with pytest.raises(ValueError):
        DeferredMarginRecheck(target_step=value)
