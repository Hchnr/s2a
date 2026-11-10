"""Restoration uses isolated worker globals and the exact earlier comparison."""

import sys
from types import ModuleType

import pytest

from scripts.simulation.runtime import run_task2_earlier_preview_feedback as original
from scripts.simulation.runtime import run_task2_earlier_restoration_feedback as runner


def branch(index, selected):
    return dict(
        complete=True,
        passed=True,
        case={"case_index": index},
        branch="restoration",
        runtime=dict(
            cleanup_complete=True,
            control_steps_completed=2,
            newly_selected_commands=selected,
            counters={"mj_step": {"completed": 50}, "model_construct": {"completed": 3}, "context": {"completed": 3}},
            refiner_calls={"attempted": 1},
            quality_refiner=dict(
                baseline_response_evaluations=3,
                quality_calls_attempted=1,
                quality_response_evaluations=4,
                endpoint_queries_charged=200,
                transition_steps_charged=2600,
            ),
        ),
    )


def test_isolated_worker_keeps_original_module_and_selects_restoration():
    isolated = runner.isolated_worker()
    assert isolated.__code__ is original.worker.__code__
    assert isolated.__globals__ is not original.worker.__globals__
    assert isolated.__globals__["ROOT"] == runner.ROOT and original.ROOT != runner.ROOT
    calls = []
    assert (
        runner.make_restoration_refiner(1, 2, 3, factory=lambda *args, **kwargs: calls.append((args, kwargs))) is None
    )
    assert calls == [((1, 2, 3), {"restore": True})]


def test_bound_inputs_retain_all_original_branches_and_current_references():
    inputs = runner.load_inputs()
    assert [item["spec"]["prefix_steps"] for item in inputs["cases"]] == [32, 80]
    receipt = runner.input_receipt(inputs)
    assert receipt["prior_producer_ref"] == runner.PRIOR_PRODUCER
    assert receipt["prior_checker_ref"] == runner.PRIOR_CHECKER


def test_restoration_costs_and_replay_isolation(monkeypatch):
    config = runner.read_config()[1]
    branches = [branch(10, 1), branch(11, 1)]
    counts = runner.summarize_phase(branches, "producer", config)
    assert counts["refiner_response_evaluations"] == 6 and counts["quality_response_evaluations"] == 8
    assert counts["audit_endpoint_queries"] == 400 and counts["audit_transition_steps"] == 5200
    branches[0]["runtime"]["newly_selected_commands"] = 1
    with pytest.raises(ValueError, match="checker"):
        runner.summarize_phase(branches, "checker", config)
    monkeypatch.setitem(sys.modules, runner.CHECKER_FORBIDDEN[-1], ModuleType(runner.CHECKER_FORBIDDEN[-1]))
    with pytest.raises(ValueError, match="candidate"):
        runner.replay_action([])
