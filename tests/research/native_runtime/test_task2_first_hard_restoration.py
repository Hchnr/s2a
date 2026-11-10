"""First-hard feedback preserves prior inputs and independently replayed actions."""

import sys
from types import ModuleType

import pytest

from scripts.simulation.runtime import run_task2_earlier_preview_feedback as original
from scripts.simulation.runtime import run_task2_earlier_restoration_feedback as previous
from scripts.simulation.runtime import run_task2_first_hard_restoration as runner
from tests.research.native_runtime.test_task2_earlier_restoration_feedback import branch


def test_isolated_worker_preserves_original_and_binds_new_factory():
    worker = runner.isolated_worker()
    assert worker.__code__ is original.worker.__code__
    assert worker.__globals__ is not original.worker.__globals__
    assert worker.__globals__["ROOT"] == runner.ROOT and previous.ROOT != runner.ROOT
    assert worker.__globals__["full"].make_full_refiner is runner.make_first_hard_refiner


def test_inputs_bind_both_prior_restoration_results():
    inputs = runner.load_inputs()
    assert [case["spec"]["prefix_steps"] for case in inputs["cases"]] == [32, 80]
    receipt = runner.input_receipt(inputs)
    assert receipt["prior_producer_ref"] == runner.PRIOR_PRODUCER
    assert receipt["prior_checker_ref"] == runner.PRIOR_CHECKER


def test_accounting_requires_a_completed_trace_and_checker_isolation(monkeypatch):
    config = runner.read_config()[1]
    branches = [branch(10, 1), branch(11, 1)]
    for item in branches:
        item["branch"] = "first_hard_restoration"
        item["runtime"]["quality_refiner"]["quality_calls_completed"] = 1
        item["runtime"]["first_hard_restoration_traces"] = [{"audit_status": "passed"}]
    counts = runner.summarize_phase(branches, "producer", config)
    assert counts["quality_response_evaluations"] == 8 and counts["audit_endpoint_queries"] == 400
    branches[0]["runtime"]["first_hard_restoration_traces"][0]["audit_status"] = "pending"
    with pytest.raises(ValueError, match="audit status"):
        runner.summarize_phase(branches, "producer", config)
    monkeypatch.setitem(sys.modules, runner.CHECKER_FORBIDDEN[-1], ModuleType(runner.CHECKER_FORBIDDEN[-1]))
    with pytest.raises(ValueError, match="candidates"):
        runner.replay_action([])
