"""The old verified pose bound must use the newly replayed fallback stops."""

import copy

import numpy as np
import pytest

from scripts.tracking.trajectory_search import audit_stopped_pose_safety_bound as original
from scripts.tracking.trajectory_search import audit_task2_fallback_stop_pose_bound as runner


def test_isolated_worker_preserves_original_function_and_uses_new_binding():
    worker = runner.isolated_worker()
    assert worker.__code__ is original.worker.__code__ and worker.__globals__ is not original.worker.__globals__
    assert worker.__globals__["OUTPUT"] == runner.OUTPUT and original.OUTPUT != runner.OUTPUT
    assert worker.__globals__["binding"].load_inputs is runner.load_inputs


def test_bound_inputs_use_the_actual_stops_and_last_completed_shift():
    inputs = runner.load_inputs()
    assert [item["bound_spec"] for item in inputs["cases"]] == runner.CASES
    assert [len(item["feedback_periods"]) for item in inputs["cases"]] == [1, 5]
    for case in inputs["cases"]:
        np.testing.assert_array_equal(case["bound_seed"], case["stop_inputs"]["shifted"])
        np.testing.assert_array_equal(case["closest_error"], case["stop_inputs"]["recorded_shifted_errors"][0])


def test_aggregate_support_budget_cannot_be_exceeded():
    cases = []
    for spec in runner.CASES:
        stats = dict(
            model_initializations_completed=1,
            forward_queries_completed=1,
            baseline_rollouts=1,
            support_lp_attempted=10,
            support_lp_completed=10,
            independent_first_responses_charged=1,
            independent_first_responses_completed=1,
            independent_transition_steps_charged=25,
            independent_transition_steps_completed=25,
        )
        cases.append(dict(case=spec, stats=stats))
    config = runner.read_config()[1]
    assert runner.summarize(cases, config)["support_lp_completed"] == 20
    changed = copy.deepcopy(cases)
    changed[0]["stats"]["support_lp_completed"] += 1
    with pytest.raises(ValueError, match="costs"):
        runner.summarize(changed, config)
