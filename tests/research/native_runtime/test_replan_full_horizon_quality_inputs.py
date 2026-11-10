"""Bound the saved current decision and do not promote relaxed plans."""

import numpy as np
import pytest

from scripts.tracking.trajectory_search import audit_replan_full_horizon_quality as run


def test_model_inputs_keep_only_current_decision_and_generated_reference():
    inputs = run.load_inputs()
    for case in inputs["cases"]:
        assert case["spec"]["prefix_steps"] == 16
        assert case["intervention_spec"] == {"cursor": 0, "chunk_index": 2}
        assert len(case["feedback_periods"]) == 1
        assert set(case["feedback_periods"][0]) == {"step", "decision"}
        np.testing.assert_array_equal(case["feedback_periods"][0]["decision"]["refined_plan"], case["decision"]["plan"])
        assert case["first_point_warm"].shape == (8, 6)


def test_only_auditable_joint_candidates_are_returned_for_each_mode():
    full = {"witness_found": True, "planned_commands": [1]}
    relaxed = {"witness_found": False, "best_joint_safe": {"planned_commands": [2]}}
    assert run.candidate_for_audit("hard_seed", full) is full
    assert run.candidate_for_audit("hard_seed", relaxed) is None
    assert run.candidate_for_audit("restore_warm", relaxed) == {"planned_commands": [2]}
    assert run.candidate_for_audit("restore_seed", {"witness_found": False}) is None
    with pytest.raises(ValueError, match="Registered solver"):
        run.candidate_for_audit("unregistered", full)


def test_independent_quality_slack_uses_the_worst_of_all_eight_tcp_points():
    errors = np.zeros((8, 6))
    errors[2, 0] = 0.012
    errors[7, 3] = 0.075
    assert run.worst_quality_slack(errors) == pytest.approx(0.5)
    errors[7, 3] = 0
    assert run.worst_quality_slack(errors) == pytest.approx(0.2)
    with pytest.raises(ValueError):
        run.worst_quality_slack(errors[:1])
