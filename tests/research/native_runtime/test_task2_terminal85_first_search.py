"""The first-only search retains the original response budget and stop identity."""

import numpy as np
import pytest

from s2a.infrastructure.publication import result_json
from s2a.research.trajectory_search import staged_first_feasibility as staged
from scripts.tracking.trajectory_search import audit_task2_terminal85_first_search as runner
from tests.research.joint_control.preview.test_full_horizon_feedback import solve_and_audit_inputs


def test_counted_solver_keeps_frozen_output_and_counts_actual_rollouts():
    args, _audit_args, targets = solve_and_audit_inputs()
    targets[-1, 0, 3] += 0.1
    original_rollout = staged.solve_staged_first_feasibility.__globals__["rollout_joint_preview"]
    primary = staged.solve_staged_first_feasibility(*args, past_inner_error_rad=np.zeros((25, 7)))
    stats = runner.new_stats()
    counted = runner.counted_solver(stats)
    result = counted(*args, past_inner_error_rad=np.zeros((25, 7)))
    assert result_json(result) == result_json(primary)
    assert (
        stats["solver_responses_attempted"]
        == stats["solver_responses_completed"]
        == result["response_evaluations"]
        <= 320
    )
    assert staged.solve_staged_first_feasibility.__globals__["rollout_joint_preview"] is original_rollout


def test_inputs_bind_the_inconclusive_terminal_bound_and_five_completed_periods():
    inputs = runner.load_inputs()
    assert len(inputs["cases"]) == 1 and inputs["cases"][0]["bound_spec"] == dict(case_index=11, step=85)
    assert len(inputs["cases"][0]["feedback_periods"]) == 5
    assert runner.input_receipt(inputs)["prior_ref"] == runner.PRIOR


def test_aggregate_budget_rejects_an_unaccounted_audit():
    config, stats = runner.read_config()[1], runner.new_stats()
    stats.update(
        model_initializations_attempted=1,
        model_initializations_completed=1,
        forward_queries_attempted=1,
        forward_queries_completed=1,
        baseline_rollouts=1,
        additional_preparation_rollouts=1,
        solver_calls_attempted=2,
        solver_calls_completed=2,
        solver_responses_attempted=100,
        solver_responses_completed=100,
        independent_audits_attempted=1,
        independent_audits_passed=1,
        independent_endpoint_queries_charged=200,
        independent_transition_steps_charged=2600,
    )
    runner.require_counts(stats, config)
    stats["independent_endpoint_queries_charged"] -= 1
    with pytest.raises(ValueError, match="costs"):
        runner.require_counts(stats, config)
