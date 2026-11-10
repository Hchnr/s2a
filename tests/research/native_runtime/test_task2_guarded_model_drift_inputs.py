"""Model-drift diagnostics may inspect only completed, replayed periods."""

import copy

import numpy as np
import pytest

from scripts.simulation.runtime import prepare_task2_guarded_model_drift as binding
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix


def test_binds_the_five_independently_replayed_completed_periods():
    inputs = binding.load_inputs()
    assert [case["drift_spec"] for case in inputs["cases"]] == binding.CASES
    assert [len(case["feedback_periods"]) for case in inputs["cases"]] == [1, 4]
    for case in inputs["cases"]:
        for offset, period in enumerate(case["feedback_periods"]):
            assert period["step"] == case["drift_spec"]["start_step"] + offset
            np.testing.assert_array_equal(period["decision"]["refined_plan"][0], period["command"][:6])
    assert binding.input_receipt(inputs)["producer_ref"] == binding.PRODUCER


def test_rejects_a_changed_replayed_endpoint_or_inner_sample():
    producer, checker = prefix.read(binding.PRODUCER), prefix.read(binding.CHECKER)
    left = prefix.read(producer["branch_refs"][0])
    right = prefix.read(checker["branch_refs"][0])
    changed = copy.deepcopy(right)
    changed["runtime"]["continuation"]["periods"][0]["actual_state"][1] += 0.001
    with pytest.raises(AssertionError):
        binding.completed_periods(left, changed, binding.CASES[0])
    changed = copy.deepcopy(right)
    changed["runtime"]["continuation"]["periods"][0]["actual_inner"][0]["qvel"][0] += 0.001
    with pytest.raises(ValueError, match="inner samples"):
        binding.completed_periods(left, changed, binding.CASES[0])


def test_rejects_a_future_or_noncausal_error_profile():
    producer, checker = prefix.read(binding.PRODUCER), prefix.read(binding.CHECKER)
    left = copy.deepcopy(prefix.read(producer["branch_refs"][0]))
    right = prefix.read(checker["branch_refs"][0])
    past = left["runtime"]["continuation"]["periods"][0]["completed_feedback"]["past_error_after_completion"]
    past["future_guarantee"] = True
    with pytest.raises(ValueError, match="Past error"):
        binding.completed_periods(left, right, binding.CASES[0])
