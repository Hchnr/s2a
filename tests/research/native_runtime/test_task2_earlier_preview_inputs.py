"""Earlier task2 interventions expose only completed state and the current chunk."""

import numpy as np
import pytest

from scripts.simulation.runtime import prepare_task2_earlier_preview as binding


@pytest.fixture(scope="module")
def inputs():
    return binding.load_inputs()


def test_each_prefix_ends_before_the_warning_and_keeps_current_chunk(inputs):
    assert [case["spec"]["prefix_steps"] for case in inputs["cases"]] == [32, 80]
    for case in inputs["cases"]:
        n = case["spec"]["prefix_steps"]
        assert len(case["commands"]) == len(case["rows"]) == n
        assert len(case["receipts"]) == n + 1
        assert all(len(value) == n + 1 for value in case["saved"].values())
        assert case["intervention_spec"]["chunk_step"] == n
        assert case["intervention_spec"]["cursor"] == 0
        np.testing.assert_array_equal(case["fixed_targets"], case["targets"][:8])


def test_online_policy_case_excludes_future_rows_and_scoring_object(inputs):
    for case in inputs["cases"]:
        actual = binding.policy_case(case)
        assert "rows" not in actual and "record" not in actual and "saved" not in actual
        assert actual["intervention_spec"] == {"cursor": 0}
        assert actual["spec"] == {"prefix_steps": case["spec"]["prefix_steps"]}
        assert actual["completed_history"].completed_periods == case["spec"]["prefix_steps"]
        np.testing.assert_array_equal(actual["completed_history"].previous_command, case["commands"][-1])


def test_rejects_misaligned_chunk_identity(inputs):
    case = inputs["cases"][0]
    old_style = dict(case, intervention_spec={**case["intervention_spec"], "prefix_steps": 34, "cursor": 2})
    bad_spec = {**case["intervention_spec"], "chunk_step": 24}
    with pytest.raises(ValueError, match="Earlier boundary"):
        binding.completed_prefix(old_style, bad_spec)
