"""Two-step feedback retains matched inputs, budgets and replay isolation."""

import sys
from types import ModuleType

import pytest

from scripts.simulation.runtime import run_task2_earlier_preview_feedback as original
from scripts.simulation.runtime import run_task2_staged_first_feasibility as previous
from scripts.simulation.runtime import run_task2_two_step_feedback as runner
from tests.research.native_runtime.test_task2_earlier_restoration_feedback import branch


def test_isolated_worker_preserves_original_and_uses_two_step_factory():
    worker = runner.isolated_worker()
    assert worker.__code__ is original.worker.__code__ and worker.__globals__ is not original.worker.__globals__
    assert worker.__globals__["ROOT"] == runner.ROOT and previous.ROOT != runner.ROOT
    assert worker.__globals__["full"].make_full_refiner is runner.make_two_step_refiner


def test_prior_static_results_and_matched_earlier_prefixes_are_bound():
    inputs = runner.load_inputs()
    assert [case["spec"]["prefix_steps"] for case in inputs["cases"]] == [32, 80]
    assert runner.input_receipt(inputs)["static_ref"] == runner.STATIC


def test_two_step_accounting_and_checker_import_isolation(monkeypatch):
    config = runner.read_config()[1]
    branches = [branch(10, 1), branch(11, 1)]
    for item in branches:
        item["branch"] = "two_step_viability"
        item["runtime"]["quality_refiner"]["quality_calls_completed"] = 1
        item["runtime"]["first_hard_restoration_traces"] = [{"audit_status": "passed"}]
        item["runtime"]["feasible_start_traces"] = [
            dict(start_lp_attempts=1, solver_lp_attempted=False, seed_rollout_reused=True)
        ]
        item["runtime"]["two_step_feedback_traces"] = [dict(response_evaluations=4, phases=[], audit_status="passed")]
    counts = runner.summarize_phase(branches, "producer", config)
    assert counts["start_lp_calls"] == 2 and counts["solver_lp_calls"] == 0
    branches[0]["runtime"]["two_step_feedback_traces"][0]["response_evaluations"] = 321
    with pytest.raises(ValueError, match="audit and response"):
        runner.summarize_phase(branches, "producer", config)
    monkeypatch.setitem(sys.modules, runner.CHECKER_FORBIDDEN[-1], ModuleType(runner.CHECKER_FORBIDDEN[-1]))
    with pytest.raises(ValueError, match="candidates"):
        runner.replay_action([])
