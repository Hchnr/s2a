"""Shared feedback retains matched evidence, cost limits and replay isolation."""

import sys
from types import ModuleType

import pytest

from scripts.simulation.runtime import run_task2_earlier_preview_feedback as original
from scripts.simulation.runtime import run_task2_margin_guarded_feedback as runner
from scripts.simulation.runtime import run_task2_margin_two_step_feedback as previous
from tests.research.native_runtime.test_task2_earlier_restoration_feedback import branch


def test_worker_is_isolated_and_uses_shared_budget_factory():
    worker = runner.isolated_worker()
    assert worker.__code__ is original.worker.__code__ and worker.__globals__ is not original.worker.__globals__
    assert worker.__globals__["ROOT"] == runner.ROOT and previous.ROOT != runner.ROOT
    assert worker.__globals__["full"].make_full_refiner is runner.make_shared_refiner


def test_static_and_prior_margin_feedback_use_the_same_recorded_starts():
    inputs = runner.load_inputs()
    assert [case["spec"]["prefix_steps"] for case in inputs["cases"]] == [32, 80]
    receipt = runner.input_receipt(inputs)
    assert receipt["static_ref"] == runner.STATIC
    assert receipt["prior_producer_ref"] == runner.PRIOR_PRODUCER
    assert receipt["prior_checker_ref"] == runner.PRIOR_CHECKER


def test_shared_accounting_and_checker_import_isolation(monkeypatch):
    config = runner.read_config()[1]
    branches = [branch(10, 1), branch(11, 1)]
    for item in branches:
        item["branch"] = "margin_guarded_fallback"
        runtime = item["runtime"]
        runtime["quality_refiner"].update(quality_calls_completed=1, witness_audits_passed=1)
        runtime["first_hard_restoration_traces"] = [{"audit_status": "passed"}]
        runtime["feasible_start_traces"] = [
            dict(start_lp_attempts=1, solver_lp_attempted=False, seed_rollout_reused=True)
        ]
        runtime["margin_guarded_traces"] = [
            dict(
                selected_scope="guarded_two_step",
                audit_status="passed",
                primary_responses=2,
                fallback_responses=2,
                response_evaluations=4,
                fallback_lp_attempts=0,
                fallback_initializer_reused=True,
                response_budget=320,
                feasibility_iteration_budget=60,
                total_iteration_budget=120,
            )
        ]
    counts = runner.summarize_phase(branches, "producer", config)
    assert counts["start_lp_calls"] == 2 and counts["guarded_selected"] == 2 and counts["fallback_responses"] == 4
    branches[0]["runtime"]["margin_guarded_traces"][0]["fallback_lp_attempts"] = 1
    with pytest.raises(ValueError, match="Shared budgets"):
        runner.summarize_phase(branches, "producer", config)
    monkeypatch.setitem(sys.modules, runner.CHECKER_FORBIDDEN[-1], ModuleType(runner.CHECKER_FORBIDDEN[-1]))
    with pytest.raises(ValueError, match="candidates"):
        runner.replay_action([])
