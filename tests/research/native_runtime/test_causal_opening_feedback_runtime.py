"""Guard physical budget and independent replay accounting for release feedback."""

from copy import deepcopy

import pytest

from scripts.simulation.runtime import run_causal_opening_feedback as runner


def branch(*, selected=1):
    return {
        "complete": True,
        "passed": True,
        "gripper_history": {"planner_calls": 2, "candidate_sequences": 2 * 3**8},
        "runtime": {
            "cleanup_complete": True,
            "control_steps_completed": 3,
            "newly_selected_commands": selected,
            "counters": {
                "mj_step": {"completed": 75},
                "model_construct": {"completed": 3},
                "context": {"completed": 3},
            },
            "gripper_proposals": {"planner_calls": 1, "candidate_sequences": 3**8},
            "refiner_calls": {"attempted": 1},
            "quality_refiner": {
                "baseline_response_evaluations": 10,
                "quality_calls_attempted": 1,
                "quality_response_evaluations": 20,
            },
        },
    }


def budget():
    return dict(
        control_steps=3,
        native_steps=75,
        model_initializations=3,
        contexts=3,
        gripper_planner_calls=3,
        gripper_candidate_sequences=3 * 3**8,
        refiner_calls=1,
        refiner_response_evaluations=10,
        quality_calls=1,
        quality_response_evaluations=20,
    )


def test_producer_counts_actual_history_proposals_and_model_calls():
    result = runner.summarize_phase([branch()], "producer", budget())
    assert result["gripper_planner_calls"] == 3
    assert result["gripper_candidate_sequences"] == 3 * 3**8
    assert result["native_steps"] == 75 and result["newly_selected_commands"] == 1


@pytest.mark.parametrize("fault", ["physics", "planner", "quality", "cleanup"])
def test_over_budget_or_unclosed_producer_is_rejected(fault):
    item = deepcopy(branch())
    if fault == "physics":
        item["runtime"]["counters"]["mj_step"]["completed"] += 1
    elif fault == "planner":
        item["gripper_history"]["planner_calls"] += 1
    elif fault == "quality":
        item["runtime"]["quality_refiner"]["quality_response_evaluations"] += 1
    else:
        item["runtime"]["cleanup_complete"] = False
    with pytest.raises(ValueError):
        runner.summarize_phase([item], "producer", budget())


def test_checker_cannot_select_or_charge_new_actions():
    item = branch(selected=0)
    result = runner.summarize_phase([item], "checker", budget())
    assert result["newly_selected_commands"] == 0 and "quality_calls" not in result
    item["runtime"]["newly_selected_commands"] = 1
    with pytest.raises(ValueError):
        runner.summarize_phase([item], "checker", budget())
