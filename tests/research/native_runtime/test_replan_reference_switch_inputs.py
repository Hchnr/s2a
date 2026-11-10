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


def test_completed_scoring_checks_own_reference_and_detects_tampering():
    targets = {name: np.tile(np.eye(4), (8, 1, 1)) for name in ("old", "new")}
    targets["old"][0, 0, 3] = 0.01
    expected = run.completed_reference_scores(np.eye(4), targets, "new", 0, np.zeros(6))
    np.testing.assert_allclose(expected["reference_errors"]["old"][:3], [0.01, 0, 0])
    run.completed_reference_scores(np.eye(4), targets, "new", 0, np.zeros(6), expected=expected)
    expected["reference_errors"]["old"][0] = 0.02
    with pytest.raises(AssertionError):
        run.completed_reference_scores(np.eye(4), targets, "new", 0, np.zeros(6), expected=expected)


def test_scoring_rejects_inconsistent_own_error_and_missing_reference():
    targets = {name: np.tile(np.eye(4), (8, 1, 1)) for name in ("old", "new")}
    with pytest.raises(AssertionError):
        run.completed_reference_scores(np.eye(4), targets, "old", 0, np.ones(6))
    with pytest.raises(ValueError, match="Both aligned"):
        run.completed_reference_scores(np.eye(4), {"old": targets["old"]}, "old", 0, np.zeros(6))
