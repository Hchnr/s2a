"""Earlier-preview accounting keeps all six branches and replay isolation."""

import sys
from types import ModuleType

import pytest

from scripts.simulation.runtime import prepare_task2_earlier_preview as binding
from scripts.simulation.runtime import run_task2_earlier_preview_feedback as runner


def branch(index, mode, selected):
    result = dict(
        complete=True,
        passed=True,
        case={"case_index": index},
        branch=mode,
        runtime=dict(
            cleanup_complete=True,
            control_steps_completed=2,
            newly_selected_commands=selected,
            counters={"mj_step": {"completed": 50}, "model_construct": {"completed": 3}, "context": {"completed": 3}},
            refiner_calls={"attempted": 1},
            refiner_response_evaluations=3,
        ),
    )
    if mode != "original":
        result["runtime"]["quality_refiner"] = dict(
            baseline_response_evaluations=3,
            quality_calls_attempted=1,
            quality_response_evaluations=4,
            endpoint_queries_charged=200,
            transition_steps_charged=2600,
        )
    return result


def test_all_six_branches_and_quality_baseline_costs_are_counted():
    config = binding.read_config()[1]
    branches = [branch(index, mode, 1) for index, mode in runner.BRANCHES]
    counts = runner.summarize_phase(branches, "producer", config)
    assert counts["control_steps"] == 12 and counts["native_steps"] == 300
    assert counts["refiner_response_evaluations"] == 18 and counts["quality_response_evaluations"] == 16
    assert counts["audit_endpoint_queries"] == 800 and counts["audit_transition_steps"] == 10400
    with pytest.raises(ValueError, match="identities"):
        runner.summarize_phase(list(reversed(branches)), "producer", config)


def test_checker_rejects_new_actions_and_candidate_imports(monkeypatch):
    config = binding.read_config()[1]
    branches = [branch(index, mode, 0) for index, mode in runner.BRANCHES]
    assert runner.summarize_phase(branches, "checker", config)["newly_selected_commands"] == 0
    branches[1]["runtime"]["newly_selected_commands"] = 1
    with pytest.raises(ValueError, match="checker"):
        runner.summarize_phase(branches, "checker", config)
    module = runner.CHECKER_FORBIDDEN[-1]
    monkeypatch.setitem(sys.modules, module, ModuleType(module))
    with pytest.raises(ValueError, match="candidate"):
        runner.replay_action([])
