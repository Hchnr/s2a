"""The static comparison must enforce its shared four-case budgets."""

import pytest

from scripts.simulation.runtime import prepare_two_step_viability as binding
from scripts.tracking.trajectory_search import audit_two_step_viability as runner


def test_aggregate_response_budget_cannot_be_exceeded():
    cases = []
    for _ in range(4):
        cases.append(
            dict(
                stats=dict(
                    model_initializations_completed=1,
                    forward_queries_completed=1,
                    baseline_rollouts=1,
                    solver_calls_attempted=1,
                    solver_calls_completed=1,
                    solver_responses_attempted=320,
                    solver_responses_completed=320,
                    start_lp_budget_charged=2,
                    start_lp_actual=1,
                    audit_attempts=1,
                    audit_passed=1,
                    audit_endpoint_queries_charged=200,
                    audit_transition_steps_charged=2600,
                )
            )
        )
    config = binding.read_config()[1]
    assert runner.summarize(cases, config)["solver_responses_completed"] == 1280
    cases[0]["stats"]["solver_responses_attempted"] += 1
    cases[0]["stats"]["solver_responses_completed"] += 1
    with pytest.raises(ValueError, match="budgets"):
        runner.summarize(cases, config)
