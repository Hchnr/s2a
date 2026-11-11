"""Qualification distinguishes two hard points from a soft future tail."""

import numpy as np
import pytest

from scripts.tracking.trajectory_search import audit_task2_shifted_seed as runner


def test_two_point_candidate_requires_both_pose_points_and_independent_safety():
    errors = np.zeros((8, 6))
    errors[2, 0] = 0.02
    result = runner.qualify(errors, True)
    assert result["model_prefix_candidate"] and result["prefix_pose_caps_satisfied"]
    assert not result["all_pose_caps_satisfied"] and result["future_quality_slack"] == pytest.approx(1.0)
    assert not runner.qualify(errors, False)["model_prefix_candidate"]
    errors[1, 3] = 0.050001
    assert not runner.qualify(errors, True)["model_prefix_candidate"]
    errors[1, 3] = 0.05 + 0.5e-10
    assert runner.qualify(errors, True)["model_prefix_candidate"]
    with pytest.raises(ValueError):
        runner.qualify(np.full((8, 6), np.nan), True)


def test_inputs_bind_the_last_replayed_period_and_prior_drift():
    inputs = runner.load_inputs()
    assert [item["stop_spec"] for item in inputs["cases"]] == runner.CASES
    assert [len(item["feedback_periods"]) for item in inputs["cases"]] == [1, 4]
    assert all(item["prior_transition"]["step"] == item["stop_spec"]["stop_step"] - 1 for item in inputs["cases"])
    assert runner.input_receipt(inputs)["prior_ref"] == runner.PRIOR


def test_audit_budget_must_not_exceed_registered_scope():
    stats = runner.new_stats()
    stats.update(
        model_initializations_attempted=1,
        model_initializations_completed=1,
        forward_queries_attempted=1,
        forward_queries_completed=1,
        baseline_rollouts=1,
        independent_audits_attempted=1,
        independent_audits_completed=1,
        independent_endpoint_queries_charged=200,
        independent_transition_steps_charged=2600,
    )
    runner.require_case_counts(stats)
    stats["independent_transition_steps_charged"] += 1
    with pytest.raises(ValueError, match="costs"):
        runner.require_case_counts(stats)
