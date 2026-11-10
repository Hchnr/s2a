"""Cache inspection binds the actual terminal85 and raw33 replayed stops."""

import copy

import pytest

from scripts.simulation.runtime import prepare_task2_cached_first_pose as binding
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix


def test_stops_bind_replayed_histories_and_saved_causal_seeds():
    inputs = binding.load_inputs()
    assert [item["stop_spec"] for item in inputs["cases"]] == binding.CASES
    assert [len(item["feedback_periods"]) for item in inputs["cases"]] == [1, 5]
    for item in inputs["cases"]:
        assert item["stop_inputs"]["lp_start"].shape == item["stop_inputs"]["shifted"].shape == (8, 6)
        assert item["stop_inputs"]["prior_primary_responses"] <= 320


def test_changed_replay_or_future_profile_is_rejected():
    producer, checker = prefix.read(binding.PRODUCER), prefix.read(binding.CHECKER)
    left, right = prefix.read(producer["branch_refs"][1]), prefix.read(checker["branch_refs"][1])
    altered = copy.deepcopy(right)
    altered["runtime"]["continuation"]["periods"][-1]["actual_inner"][-1]["qvel"][0] += 0.001
    with pytest.raises(ValueError, match="inner sample"):
        binding.completed_stop(left, altered, binding.CASES[1])
    altered = copy.deepcopy(left)
    altered["runtime"]["continuation"]["periods"][-1]["completed_feedback"]["past_error_after_completion"][
        "future_guarantee"
    ] = True
    with pytest.raises(ValueError, match="causal error profile"):
        binding.completed_stop(altered, right, binding.CASES[1])
