"""Three-task stopped states must come from independently replayed histories."""

import copy

import numpy as np
import pytest

from scripts.simulation.runtime import prepare_stopped_pose_safety_bound as binding
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix


def test_four_stops_bind_only_completed_histories_and_shifted_plans():
    inputs = binding.load_inputs()
    cases = inputs["cases"]
    assert [case["bound_spec"]["case_index"] for case in cases] == [2, 6, 10, 11]
    assert [case["bound_spec"]["step"] for case in cases] == [48, 57, 33, 81]
    assert [len(case["feedback_periods"]) for case in cases] == [5, 5, 1, 1]
    for case in cases:
        completed = case["feedback_periods"][-1]
        plan = np.asarray(completed["decision"]["refined_plan"])
        np.testing.assert_array_equal(case["bound_seed"], np.vstack((plan[1:], plan[-1])))
        assert case["bound_profile"].shape == (25, 7)
        assert case["closest_commands"].shape == (8, 6)
        assert case["closest_error"].shape == (6,)
        assert "not_original_stop_seed" in case["seed_scope"]


def test_staged_binding_rejects_changed_replayed_state():
    producer = prefix.read(binding.STAGED_PRODUCER)
    checker = prefix.read(binding.STAGED_CHECKER)
    left = prefix.read(producer["branch_refs"][0])
    right = copy.deepcopy(prefix.read(checker["branch_refs"][0]))
    right["runtime"]["continuation"]["periods"][0]["actual_state"][1] += 0.001
    with pytest.raises(AssertionError):
        binding.completed_staged_stop(left, right, binding.CASES[2])
