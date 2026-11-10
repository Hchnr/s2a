"""Bind both solvers to the same causal states, histories and original seeds."""

import numpy as np
import pytest

from scripts.simulation.runtime import prepare_multitask_feasibility_first as binding


@pytest.fixture(scope="module")
def cases():
    return binding.load_inputs()["cases"]


def test_four_development_states_cover_three_tasks_without_fresh_cases(cases):
    assert [case["comparison_spec"]["case_index"] for case in cases] == [2, 6, 10, 11]
    assert [case["comparison_spec"]["step"] for case in cases] == [48, 57, 34, 82]
    assert [case["comparison_spec"]["completed_feedback_steps"] for case in cases] == [5, 5, 0, 0]
    assert all(case["config"]["seed"] == 43 for case in cases)


def test_recorded_plan_scopes_are_explicit_and_bounded(cases):
    for case in cases[:2]:
        assert case["seed_scope"] == "shifted_last_completed_plan_not_original_stop_seed"
        assert np.asarray(case["stop_inputs"]["seed"]).shape == (8, 6)
    for case in cases[2:]:
        assert case["seed_scope"] == "matched_original_refiner_plan_at_warning"
        assert len(case["feedback_periods"]) == 1
        assert np.asarray(case["feedback_periods"][0]["decision"]["refined_plan"]).shape == (8, 6)
        assert case["feedback_periods"][0]["step"] == case["comparison_spec"]["step"]
