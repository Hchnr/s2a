"""Staged feedback preserves prior data, process isolation and budgets."""

import sys
from types import ModuleType

import pytest

from scripts.simulation.runtime import run_task2_earlier_preview_feedback as original
from scripts.simulation.runtime import run_task2_feasible_start_restoration as previous
from scripts.simulation.runtime import run_task2_staged_first_feasibility as runner
from tests.research.native_runtime.test_task2_earlier_restoration_feedback import branch


def test_isolated_worker_keeps_original_globals_and_uses_staged_factory():
    worker = runner.isolated_worker()
    assert worker.__code__ is original.worker.__code__ and worker.__globals__ is not original.worker.__globals__
    assert worker.__globals__["ROOT"] == runner.ROOT and previous.ROOT != runner.ROOT
    assert worker.__globals__["full"].make_full_refiner is runner.make_staged_refiner


def test_prior_feasible_start_results_and_earlier_prefixes_are_bound():
    inputs = runner.load_inputs()
    assert [case["spec"]["prefix_steps"] for case in inputs["cases"]] == [32, 80]
    assert runner.input_receipt(inputs)["prior_checker_ref"] == runner.PRIOR_CHECKER


def test_staged_accounting_and_replay_isolation(monkeypatch):
    config = runner.read_config()[1]
    branches = [branch(10, 1), branch(11, 1)]
    for item in branches:
        item["branch"] = "staged_first_feasibility"
        item["runtime"]["quality_refiner"]["quality_calls_completed"] = 1
        item["runtime"]["first_hard_restoration_traces"] = [{"audit_status": "passed"}]
        item["runtime"]["feasible_start_traces"] = [
            dict(start_lp_attempts=1, solver_lp_attempted=False, seed_rollout_reused=True)
        ]
        item["runtime"]["staged_first_feasibility_traces"] = [dict(response_evaluations=4, phases=[])]
    counts = runner.summarize_phase(branches, "producer", config)
    assert counts["start_lp_calls"] == 2 and counts["solver_lp_calls"] == 0
    branches[0]["runtime"]["staged_first_feasibility_traces"][0]["response_evaluations"] = 321
    with pytest.raises(ValueError, match="phase and response"):
        runner.summarize_phase(branches, "producer", config)
    monkeypatch.setitem(sys.modules, runner.CHECKER_FORBIDDEN[-1], ModuleType(runner.CHECKER_FORBIDDEN[-1]))
    with pytest.raises(ValueError, match="candidates"):
        runner.replay_action([])
