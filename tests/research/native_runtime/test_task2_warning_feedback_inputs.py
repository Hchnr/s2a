"""The third-task feedback uses only completed history and generated chunks."""

import numpy as np
import pytest

from scripts.simulation.runtime import prepare_task2_warning_feedback as binding


@pytest.fixture(scope="module")
def cases():
    return binding.load_inputs()["cases"]


def test_fixed_reference_and_completed_history_are_causally_aligned(cases):
    assert [item["spec"]["prefix_steps"] for item in cases] == [34, 82]
    for case in cases:
        spec = case["intervention_spec"]
        assert spec["chunk_step"] + spec["cursor"] == spec["prefix_steps"]
        np.testing.assert_array_equal(case["fixed_targets"], case["targets"][2:10])
        assert len(case["rows"]) == spec["prefix_steps"]
        assert case["rows"][-1]["step"] == spec["prefix_steps"] - 1
        restored = binding.producer_case(case)
        assert restored["completed_history"].completed_periods == spec["prefix_steps"]
        np.testing.assert_array_equal(restored["completed_history"].previous_command, case["commands"][-1])


def test_incomplete_last_period_cannot_be_restored(cases):
    case = dict(cases[0], rows=[*cases[0]["rows"]])
    case["rows"][-1] = dict(case["rows"][-1], step=0)
    with pytest.raises(ValueError, match="Completed causal"):
        binding.producer_case(case)
