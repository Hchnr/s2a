"""Match screened cases and generated reference endpoints to fixed replay inputs."""

import numpy as np
import pytest

from scripts.simulation.runtime import prepare_causal_opening_feedback as binding


@pytest.fixture(scope="module")
def inputs():
    return binding.load_inputs()


def test_screened_and_rejected_cases_keep_the_full_original_denominator(inputs):
    assert inputs["admitted_cases"] == [3, 7]
    assert inputs["rejected_cases"] == [10, 11]
    assert inputs["branches"] == [(3, "latest"), (3, "remembered"), (7, "latest"), (7, "remembered")]
    assert inputs["phase_budget"]["control_steps"] == 752
    assert inputs["phase_budget"]["native_steps"] == 18800
    assert inputs["phase_budget"]["model_initializations"] == inputs["phase_budget"]["contexts"] == 12


def test_selected_references_preserve_both_components_and_absolute_time(inputs):
    for case in inputs["cases"]:
        assert len(case["commands"]) == case["spec"]["prefix_steps"]
        for mode in ("latest", "remembered"):
            selected = binding.selected_case(case, mode)
            ref = case["references"][mode]
            cursor = ref["cursor"]
            assert case["spec"]["prefix_steps"] + 1 == (
                (case["spec"]["source_step"] if mode == "remembered" else case["spec"]["prefix_steps"]) + cursor + 1
            )
            np.testing.assert_array_equal(selected["fixed_targets"], ref["poses"][cursor : cursor + 8])
            np.testing.assert_array_equal(selected["fixed_openings"], ref["openings"][cursor : cursor + 8])
        with pytest.raises(ValueError):
            binding.selected_case(case, "mixed")
