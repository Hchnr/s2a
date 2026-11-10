"""The third-task feedback keeps its full budget and replay-only checker."""

import pytest

from scripts.simulation.runtime import prepare_task2_warning_feedback as inputs
from scripts.simulation.runtime import run_task2_warning_feedback as runner


def branch(mode, selected):
    result = dict(
        complete=True,
        passed=True,
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
    if mode == "quality":
        result["runtime"]["quality_refiner"] = {
            "baseline_response_evaluations": 3,
            "quality_calls_attempted": 1,
            "quality_response_evaluations": 4,
            "endpoint_queries_charged": 200,
            "transition_steps_charged": 2600,
        }
    return result


def test_cost_accounting_includes_failed_quality_baseline_responses():
    config = inputs.read_config()[1]
    branches = [branch("original", 1), branch("quality", 1), branch("original", 1), branch("quality", 1)]
    counts = runner.summarize_phase(branches, "producer", config)
    assert counts["control_steps"] == 8 and counts["native_steps"] == 200
    assert counts["refiner_response_evaluations"] == 12 and counts["quality_response_evaluations"] == 8
    assert counts["audit_endpoint_queries"] == 400 and counts["audit_transition_steps"] == 5200


def test_checker_cannot_select_actions_or_hide_incomplete_cleanup():
    config = inputs.read_config()[1]
    branches = [branch("original", 0), branch("quality", 0), branch("original", 0), branch("quality", 0)]
    assert runner.summarize_phase(branches, "checker", config)["newly_selected_commands"] == 0
    branches[0]["runtime"]["newly_selected_commands"] = 1
    with pytest.raises(ValueError, match="checker"):
        runner.summarize_phase(branches, "checker", config)
    branches[0]["runtime"]["newly_selected_commands"] = 0
    branches[1]["runtime"]["cleanup_complete"] = False
    with pytest.raises(ValueError, match="cleanup"):
        runner.summarize_phase(branches, "checker", config)
