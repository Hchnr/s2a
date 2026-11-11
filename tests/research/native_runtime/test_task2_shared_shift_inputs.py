"""Shifted plans can only use independently replayed completed history."""

import copy

import numpy as np
import pytest

from scripts.simulation.runtime import prepare_task2_shared_shift as binding
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix


def saved_branch(ref, index):
    result = prefix.read(ref)
    for item in result["branch_refs"]:
        branch = prefix.read(item)
        if branch["case"]["case_index"] == index:
            return branch
    raise AssertionError("Registered branch not found")


def test_two_replayed_stops_bind_only_their_completed_plans():
    inputs = binding.load_inputs()
    assert [item["stop_spec"] for item in inputs["cases"]] == binding.CASES
    assert [len(item["feedback_periods"]) for item in inputs["cases"]] == [1, 4]
    for case in inputs["cases"]:
        last = case["feedback_periods"][-1]
        plan = np.asarray(last["decision"]["refined_plan"])
        np.testing.assert_array_equal(case["stop_inputs"]["shifted"], np.vstack((plan[1:], plan[-1])))
        assert last["step"] == case["stop_spec"]["stop_step"] - 1


def test_replay_mismatch_and_future_error_profile_are_rejected():
    produced = saved_branch(binding.PRODUCER, 11)
    checked = saved_branch(binding.CHECKER, 11)
    spec = binding.CASES[1]
    altered = copy.deepcopy(checked)
    altered["runtime"]["continuation"]["periods"][-1]["actual_inner"][-1]["q"][0] += 1e-4
    with pytest.raises(ValueError, match="inner samples"):
        binding.completed_stop(produced, altered, spec)
    altered = copy.deepcopy(produced)
    past = altered["runtime"]["continuation"]["periods"][-1]["completed_feedback"]["past_error_after_completion"]
    past["completed_periods"] += 1
    with pytest.raises(ValueError, match="branch-local"):
        binding.completed_stop(altered, checked, spec)


def test_uncompleted_or_mismatched_stop_cannot_supply_a_shifted_plan():
    produced = saved_branch(binding.PRODUCER, 10)
    checked = saved_branch(binding.CHECKER, 10)
    altered = copy.deepcopy(produced)
    altered["runtime"]["continuation"]["stop"]["next_action_dispatched"] = True
    with pytest.raises(ValueError, match="recorded stop"):
        binding.completed_stop(altered, checked, binding.CASES[0])
