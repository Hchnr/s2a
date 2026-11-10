"""Bind both branches to the same complete predecision history."""

import numpy as np
import pytest

from scripts.simulation.runtime import run_replan_reference_switch as run


def test_reviewed_old_and_new_chunks_share_the_same_physical_endpoints():
    inputs = run.load_inputs()
    assert [case["spec"]["prefix_steps"] for case in inputs["cases"]] == [16, 16]
    for case in inputs["cases"]:
        assert case["completed_history"].completed_periods == 16
        assert len(case["commands"]) == 16 and len(case["saved"]["sim_states"]) == 17
        old, new = (run.selected_case(case, name) for name in ("old", "new"))
        assert old["intervention_spec"]["cursor"] == 8 and new["intervention_spec"]["cursor"] == 0
        assert "reference_chunks" not in old and "scoring_targets" not in old
        np.testing.assert_array_equal(old["fixed_targets"], case["reference_chunks"]["old"][8:16])
        np.testing.assert_array_equal(new["fixed_targets"], case["reference_chunks"]["new"][:8])
        np.testing.assert_array_equal(old["commands"], new["commands"])


def test_unknown_reference_choice_is_rejected():
    with pytest.raises(ValueError, match="reference choice"):
        run.selected_case({}, "later")
