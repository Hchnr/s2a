"""Reconstruct completed ordered-correct gripper history from recorded inputs."""

from dataclasses import asdict

import numpy as np

from s2a.geometry import validate_transform
from s2a.research.gripper.causal_loaded_history import compare_recorded, loaded_state
from s2a.research.joint_control.tracking.loaded_velocity_control import LoadedVelocityControl


def reconstruct_ordered_gripper(rows, chunks, tool, parameters, *, velocity_envelope, gripper_config, progress=None):
    """Recompute only completed decisions; future rows and task labels are not inputs."""
    count = len(rows)
    if not count or len(chunks) != (count + 7) // 8:
        raise ValueError("Exactly the completed rows and their reference chunks are required")
    tool = validate_transform(np.asarray(tool, dtype=float))
    if tool.shape != (4, 4):
        raise ValueError("A single fixed TCP tool transform is required")
    controller = LoadedVelocityControl(
        parameters, float(rows[0]["opening_before"]), velocity_envelope=velocity_envelope
    )
    if asdict(controller.latch.config) != gripper_config:
        raise ValueError("Original gripper latch configuration differs")
    previous, maximum, planner_calls, candidates = 0.0, 0.0, 0, 0
    routes = {}
    ledger = {"steps_attempted": 0, "steps_completed": 0, "planner_calls_completed": 0, "candidate_sequences": 0}
    for step, row in enumerate(rows):
        ledger["steps_attempted"] += 1
        if progress is not None:
            progress(dict(ledger))
        chunk_index, offset = divmod(step, 8)
        chunk = chunks[chunk_index]
        if (
            row["step"] != step
            or row["chunk_index"] != chunk_index
            or row["reference_index"] != offset
            or chunk["step"] != chunk_index * 8
        ):
            raise ValueError("Completed row and generated reference clocks differ")
        poses = validate_transform(np.asarray(chunk["effective_targets_tcp"], dtype=float))
        openings = np.asarray(chunk["openings"], dtype=float)
        if poses.shape != (40, 4, 4) or openings.shape != (40,) or not np.isfinite(openings).all():
            raise ValueError("The original full generated reference is required")
        tcp = validate_transform(np.asarray(row["measured_source"], dtype=float)) @ tool
        result = controller.update(
            float(row["opening_before"]),
            np.asarray(row["motor_target_before"], dtype=float),
            float(row["measured_period_velocity"]),
            poses,
            openings,
            offset,
            tcp,
            previous_command=previous,
            dt=0.05,
        )
        actual = asdict(result)
        if result.plan is not None:
            planner_calls += 1
            plan = result.plan
            candidates += (
                plan["valid_sequences"] + plan["rejected_nonfinite_sequences"] + plan["rejected_velocity_sequences"]
            )
        ledger.update(planner_calls_completed=planner_calls, candidate_sequences=candidates)
        if progress is not None:
            progress(dict(ledger))
        maximum = max(maximum, compare_recorded(actual, row["gripper"]))
        executed = float(row["command"][6])
        if result.command != executed:
            raise ValueError("Reconstructed gripper command differs from the completed action")
        routes[result.route] = routes.get(result.route, 0) + 1
        previous = executed
        ledger["steps_completed"] += 1
        if progress is not None:
            progress(dict(ledger))
    return controller, {
        "steps": count,
        "routes": routes,
        "planner_calls": planner_calls,
        "candidate_sequences": candidates,
        "max_recorded_difference": maximum,
        "last_completed_command": previous,
        "latch_state": loaded_state(controller),
    }
