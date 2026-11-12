"""Preserve uncertainty when physical workers stop without complete receipts."""

import pytest

from s2a.research.native_runtime.physical_cost_bounds import physical_cost_bounds

LIMITS = dict(control_steps=115, native_steps=2875, models=3, contexts=3)


def runtime():
    return dict(
        control_steps_attempted=8,
        control_steps_completed=7,
        counters={
            "mj_step": dict(attempted=198, completed=197),
            "model_construct": dict(attempted=3, completed=3),
            "context": dict(attempted=3, completed=2),
        },
    )


def test_missing_receipt_keeps_the_full_registered_budget_unknown():
    bounds = physical_cost_bounds(None, limits=LIMITS, receipt_final=True)
    assert bounds == {key: dict(lower=0, upper=value, evidence="not_recorded") for key, value in LIMITS.items()}


def test_final_partial_receipt_bounds_inflight_work_without_claiming_completion():
    bounds = physical_cost_bounds(runtime(), limits=LIMITS, receipt_final=True)
    assert bounds == dict(
        control_steps=dict(lower=7, upper=8, evidence="final_receipt"),
        native_steps=dict(lower=197, upper=198, evidence="final_receipt"),
        models=dict(lower=3, upper=3, evidence="final_receipt"),
        contexts=dict(lower=2, upper=3, evidence="final_receipt"),
    )


def test_nonfinal_snapshot_does_not_cap_future_work_at_observed_attempts():
    bounds = physical_cost_bounds(runtime(), limits=LIMITS, receipt_final=False)
    assert bounds["control_steps"] == dict(lower=7, upper=115, evidence="partial_snapshot")
    assert bounds["native_steps"] == dict(lower=197, upper=2875, evidence="partial_snapshot")


def test_missing_individual_counters_are_not_treated_as_zero():
    value = runtime()
    del value["counters"]["context"]
    del value["control_steps_attempted"], value["control_steps_completed"]
    bounds = physical_cost_bounds(value, limits=LIMITS, receipt_final=True)
    assert bounds["contexts"] == dict(lower=0, upper=3, evidence="not_recorded")
    assert bounds["control_steps"] == dict(lower=0, upper=115, evidence="not_recorded")


@pytest.mark.parametrize("change", ["missing_completed", "reversed", "over_budget", "boolean", "bad_limits"])
def test_invalid_or_incomplete_counters_are_rejected(change):
    value, limits = runtime(), dict(LIMITS)
    if change == "missing_completed":
        del value["counters"]["mj_step"]["completed"]
    elif change == "reversed":
        value["control_steps_completed"] = 9
    elif change == "over_budget":
        value["counters"]["mj_step"]["attempted"] = 2876
    elif change == "boolean":
        value["counters"]["context"]["completed"] = True
    else:
        del limits["contexts"]
    with pytest.raises(ValueError):
        physical_cost_bounds(value, limits=limits, receipt_final=True)
