"""Historical gripper state must be reconstructed from causal observations."""

import copy
import json
from pathlib import Path

import pytest

from s2a.research.gripper.causal_loaded_history import compare_recorded, loaded_state, reconstruct_loaded_history
from scripts.simulation.runtime import compare_multicase_emergency_retreat as prior
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.wam.tracking.preview.audit_replan_task_events import load_inputs


def historical_prefix(count=3):
    item = load_inputs()["cases"][0]
    manifest = prefix.read(item["refs"]["manifest"])
    folder = Path(item["refs"]["manifest"]["path"]).parent
    controller = json.loads(prefix.checked(prefix.reference(folder / "controller.json")).read_text())
    rows = prefix.read_recorded_prefix(prefix.checked(item["refs"]["steps"]), count, 600)
    chunks = [prior.read_array_item(prefix.checked(item["refs"]["chunks"]), 0)]
    tool = prefix.read(item["refs"]["tool"])["source_from_tcp"]
    parameters = prefix.read(manifest["experiment"]["response_parameters_ref"])
    return rows, chunks, tool, parameters, controller


def reconstruct(rows, chunks, tool, parameters, controller):
    return reconstruct_loaded_history(
        rows,
        chunks,
        tool,
        parameters,
        velocity_envelope=controller["gripper_velocity_envelope"],
        gripper_config=controller["gripper_config"],
    )


def test_reconstructs_real_completed_decisions_and_latch_state():
    args = historical_prefix()
    controller, result = reconstruct(*args)
    assert result["steps"] == 3 and sum(result["routes"].values()) == 3
    assert result["last_completed_command"] == args[0][-1]["command"][6]
    assert result["max_recorded_difference"] <= 1e-10
    assert result["latch_state"] == loaded_state(controller)


@pytest.mark.parametrize("corruption", ["clock", "command", "motor", "preview", "velocity", "future_chunk"])
def test_rejects_broken_history_or_extra_generated_chunk(corruption):
    rows, chunks, tool, parameters, controller = copy.deepcopy(historical_prefix())
    if corruption == "clock":
        rows[1]["chunk_physical_step"] = 3
    elif corruption == "command":
        rows[1]["command"][6] = 0.123
    elif corruption == "motor":
        rows[1]["motor_opening_before"] += 0.1
    elif corruption == "preview":
        rows[1]["gripper"]["preview"]["reference"] += 0.1
    elif corruption == "velocity":
        rows[1]["measured_period_velocity"] += 0.1
    else:
        chunks.append(copy.deepcopy(chunks[0]))
    with pytest.raises(ValueError):
        reconstruct(rows, chunks, tool, parameters, controller)


def test_receipt_comparison_rejects_nonfinite_and_discrete_mismatches():
    assert compare_recorded({"values": (1.0, 2), "flag": True}, {"values": [1.0, 2], "flag": True}) == 0
    with pytest.raises(ValueError):
        compare_recorded({"flag": 1}, {"flag": True})
    with pytest.raises(ValueError):
        compare_recorded({"value": float("nan")}, {"value": 1.0})


def test_partial_reconstruction_preserves_completed_work_accounting():
    rows, chunks, tool, parameters, controller = historical_prefix()
    rows[1]["chunk_physical_step"] = 3
    progress = []
    with pytest.raises(ValueError, match="chunk clock"):
        reconstruct_loaded_history(
            rows,
            chunks,
            tool,
            parameters,
            velocity_envelope=controller["gripper_velocity_envelope"],
            gripper_config=controller["gripper_config"],
            progress=progress.append,
        )
    assert progress[-1]["steps_attempted"] == 2 and progress[-1]["steps_completed"] == 1
    assert progress[-1]["planner_calls_completed"] <= 1
