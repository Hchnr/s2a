"""Check the selected reference and history on both reviewed development prefixes."""

import numpy as np

from scripts.simulation.runtime import run_combined_warning_feedback as run


def test_real_inputs_end_at_first_warning_and_keep_only_completed_history():
    inputs = run.load_inputs()
    assert [item["spec"]["prefix_steps"] for item in inputs["cases"]] == [9, 9]
    for case in inputs["cases"]:
        assert case["first_warning"]["endpoint"] == 9
        assert case["first_warning"]["warning"]["reasons"] == ["position"]
        assert case["completed_history"].completed_periods == 9
        assert len(case["commands"]) == 9 and len(case["saved"]["sim_states"]) == 10
        assert case["intervention_spec"]["chunk_index"] == 1
        assert case["intervention_spec"]["cursor"] == 1
        np.testing.assert_array_equal(case["fixed_targets"], case["targets"][1:9])
