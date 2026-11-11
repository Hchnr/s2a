import copy

import pytest

from s2a.research.gripper.ordered_correct_history import reconstruct_ordered_gripper
from scripts.simulation.runtime.rebuild_ordered_correct_gripper import load_inputs


def rebuild(inputs, rows):
    return reconstruct_ordered_gripper(
        rows,
        inputs["chunks"][:1],
        inputs["tool"],
        inputs["parameters"],
        velocity_envelope=inputs["velocity_envelope"],
        gripper_config=inputs["gripper_config"],
    )


def test_recorded_completed_prefix_rebuilds_without_task_labels():
    inputs = load_inputs()
    rows = copy.deepcopy(inputs["rows"][:2])
    for row in rows:
        row.pop("objects")
        row.pop("success")
    _controller, result = rebuild(inputs, rows)
    assert result["steps"] == 2
    assert result["last_completed_command"] == rows[-1]["command"][6]
    assert result["candidate_sequences"] == result["planner_calls"] * 3**8


@pytest.mark.parametrize("corruption", ["step", "reference", "command", "latch"])
def test_reconstruction_rejects_mismatched_history(corruption):
    inputs = load_inputs()
    rows = copy.deepcopy(inputs["rows"][:1])
    if corruption == "step":
        rows[0]["step"] = 2
    elif corruption == "reference":
        rows[0]["reference_index"] = 1
    elif corruption == "command":
        rows[0]["command"][6] = -rows[0]["gripper"]["command"] if rows[0]["gripper"]["command"] else 0.5
    else:
        rows[0]["gripper"]["latch"]["closing_episode"] = not rows[0]["gripper"]["latch"]["closing_episode"]
    with pytest.raises(ValueError):
        rebuild(inputs, rows)
