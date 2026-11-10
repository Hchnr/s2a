"""The longer experiment reuses the same causal prefix and one generated chunk."""

import numpy as np
import pytest

from scripts.simulation.runtime import run_combined_warning_feedback as run


def test_sixteen_targets_extend_only_the_same_previously_generated_reference():
    inputs = run.load_long_inputs()
    assert [case["spec"]["prefix_steps"] for case in inputs["cases"]] == [9, 9]
    for case in inputs["cases"]:
        assert case["first_warning"]["endpoint"] == 9
        assert case["completed_history"].completed_periods == 9
        assert case["fixed_targets"].shape == (16, 4, 4)
        np.testing.assert_array_equal(case["fixed_targets"], case["targets"][1:17])
        np.testing.assert_array_equal(case["fixed_targets"][:8], case["targets"][1:9])


def test_long_checker_accepts_late_stop_only_with_explicit_horizon():
    stop = {"reason": "quality_no_witness", "at_step": 21, "next_action_dispatched": False}
    continuation = {"periods": [{} for _ in range(12)], "stop": None}
    result = run.reconcile_checker_stop(continuation, stop, mode="quality", start_step=9, horizon=16)
    assert result["decision_stop_recorded_not_recomputed"] is True
    with pytest.raises(ValueError, match="stop must follow"):
        run.reconcile_checker_stop(
            {"periods": [{} for _ in range(12)], "stop": None}, stop, mode="quality", start_step=9
        )
