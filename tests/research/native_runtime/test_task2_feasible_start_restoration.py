"""Feasible-start feedback retains prior data and bounds initializer costs."""

import sys
from types import ModuleType

import pytest

from scripts.simulation.runtime import run_task2_earlier_preview_feedback as original
from scripts.simulation.runtime import run_task2_feasible_start_restoration as runner
from scripts.simulation.runtime import run_task2_first_hard_restoration as previous
from tests.research.native_runtime.test_task2_earlier_restoration_feedback import branch


def test_isolated_worker_binds_new_factory_without_changing_previous_globals():
    worker = runner.isolated_worker()
    assert worker.__code__ is original.worker.__code__ and worker.__globals__ is not original.worker.__globals__
    assert worker.__globals__["ROOT"] == runner.ROOT and previous.ROOT != runner.ROOT
    assert worker.__globals__["full"].make_full_refiner is runner.make_feasible_start_refiner


def test_inputs_bind_both_prior_first_hard_results():
    inputs = runner.load_inputs()
    assert [case["spec"]["prefix_steps"] for case in inputs["cases"]] == [32, 80]
    assert runner.input_receipt(inputs)["prior_checker_ref"] == runner.PRIOR_CHECKER


def test_initializer_lp_and_audit_accounting_cannot_be_hidden(monkeypatch):
    config = runner.read_config()[1]
    branches = [branch(10, 1), branch(11, 1)]
    for item in branches:
        item["branch"] = "feasible_start_restoration"
        item["runtime"]["quality_refiner"]["quality_calls_completed"] = 1
        item["runtime"]["first_hard_restoration_traces"] = [{"audit_status": "passed"}]
        item["runtime"]["feasible_start_traces"] = [
            dict(start_lp_attempts=2, solver_lp_attempted=True, seed_rollout_reused=True)
        ]
    counts = runner.summarize_phase(branches, "producer", config)
    assert counts["start_lp_calls"] == 4 and counts["solver_lp_calls"] == 2 and counts["cached_seed_rollouts"] == 2
    branches[0]["runtime"]["feasible_start_traces"][0]["start_lp_attempts"] = 3
    with pytest.raises(ValueError, match="initializer costs"):
        runner.summarize_phase(branches, "producer", config)
    monkeypatch.setitem(sys.modules, runner.CHECKER_FORBIDDEN[-1], ModuleType(runner.CHECKER_FORBIDDEN[-1]))
    with pytest.raises(ValueError, match="candidates"):
        runner.replay_action([])
