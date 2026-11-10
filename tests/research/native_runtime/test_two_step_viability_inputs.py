"""Pre-stop selection must not expose a future physical endpoint to the solver."""

import numpy as np

from scripts.simulation.runtime import prepare_two_step_viability as binding


def test_pre_stop_inputs_use_recorded_plan_and_only_earlier_completed_feedback():
    cases = binding.load_inputs()["cases"]
    assert [case["decision_spec"]["case_index"] for case in cases] == [2, 6, 10, 11]
    assert [case["decision_spec"]["step"] for case in cases] == [47, 56, 32, 80]
    assert [case["decision_inputs"]["offset"] for case in cases] == [4, 4, 0, 0]
    for case in cases:
        inputs = case["decision_inputs"]
        current = inputs["periods"][-1]
        assert set(current) == {"step", "decision"}
        assert set(current["decision"]) == {"refined_plan"}
        np.testing.assert_array_equal(current["decision"]["refined_plan"], inputs["recorded_plan"])
        assert inputs["profile"].shape == (25, 7)
    for case in cases[-2:]:
        assert case["decision_inputs"]["profile_scope"] == "no_paired_inner_error_available"
        assert not np.any(case["decision_inputs"]["profile"])
