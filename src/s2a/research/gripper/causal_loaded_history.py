"""Reconstruct the original loaded-gripper controller from completed history."""

import math
from dataclasses import asdict
from numbers import Real

import numpy as np

from s2a.research.gripper.gripper_tracking import panda_opening_from_motor_target
from s2a.research.joint_control.tracking.loaded_velocity_control import LoadedVelocityControl


def compare_recorded(actual, recorded, *, atol=1e-10):
    """Compare complete nested decision receipts while preserving discrete fields."""
    maximum = 0.0

    def compare(a, b):
        nonlocal maximum
        if isinstance(b, dict):
            if not isinstance(a, dict) or set(a) != set(b):
                raise ValueError("Recorded decision fields differ")
            for key in b:
                compare(a[key], b[key])
        elif isinstance(b, list):
            if not isinstance(a, (list, tuple)) or len(a) != len(b):
                raise ValueError("Recorded decision sequence differs")
            for left, right in zip(a, b, strict=True):
                compare(left, right)
        elif isinstance(b, bool) or b is None or isinstance(b, str):
            if type(a) is not type(b) or a != b:
                raise ValueError("Recorded discrete decision differs")
        elif isinstance(b, int):
            if isinstance(a, bool) or not isinstance(a, (int, np.integer)) or int(a) != b:
                raise ValueError("Recorded integer decision differs")
        elif isinstance(b, float):
            if isinstance(a, bool) or not isinstance(a, Real) or not math.isfinite(float(a)) or not math.isfinite(b):
                raise ValueError("Recorded numerical decision must be finite")
            difference = abs(float(a) - b)
            if difference > atol:
                raise ValueError("Recorded numerical decision differs")
            maximum = max(maximum, difference)
        else:
            raise ValueError("Unsupported recorded decision field")

    compare(actual, recorded)
    return maximum


def loaded_state(controller):
    """Expose the historical latch state needed to continue the same episode."""
    latch = controller.latch
    return {
        "intent_extreme": latch.intent.extreme,
        "intent_direction": latch.intent.direction,
        "closing": latch._closing,
        "reference_minimum": latch._reference_min,
        "holding_width": latch._holding_width,
        "release_s": latch._release_s,
        "release_previous": latch._release_previous,
    }


def reconstruct_loaded_history(
    rows, chunks, tool, parameters, *, velocity_envelope, gripper_config, atol=1e-10, progress=None
):
    """Replay completed observations and generated references without future state."""
    n, stride, dt = len(rows), 8, 0.05
    if not n or len(chunks) != (n + stride - 1) // stride:
        raise ValueError("Exactly the completed history and its generated chunks are required")
    if not math.isfinite(atol) or atol != 1e-10:
        raise ValueError("Original absolute comparison tolerance required")
    tool = np.asarray(tool, dtype=float)
    if tool.shape != (4, 4) or not np.isfinite(tool).all():
        raise ValueError("Finite original TCP transform required")
    controller = LoadedVelocityControl(
        parameters, float(rows[0]["opening_before"]), velocity_envelope=velocity_envelope
    )
    if asdict(controller.latch.config) != gripper_config:
        raise ValueError("Original gripper latch configuration differs")
    previous, max_difference, planner_calls, candidate_sequences = 0.0, 0.0, 0, 0
    routes = {"native_close_load": 0, "model_tracking": 0}
    ledger = {
        "steps_attempted": 0,
        "steps_completed": 0,
        "planner_calls_completed": 0,
        "candidate_sequences_completed": 0,
    }
    for step, row in enumerate(rows):
        ledger["steps_attempted"] += 1
        if progress is not None:
            progress(dict(ledger))
        query, offset = divmod(step, stride)
        chunk = chunks[query]
        if (
            row["step"] != step
            or row["chunk_index"] != query
            or row["chunk_physical_step"] != offset
            or chunk["step"] != query * stride
        ):
            raise ValueError("Original completed step and generated chunk clock required")
        poses = np.asarray(chunk["effective_targets_tcp"], dtype=float)
        openings = np.asarray(chunk["openings"], dtype=float)
        motor = np.asarray(row["motor_target_before"], dtype=float)
        velocity = np.asarray(row["measured_gripper_qvel"], dtype=float)
        if poses.shape != (40, 4, 4) or openings.shape != (40,) or velocity.shape != (2,):
            raise ValueError("Original generated horizon and gripper velocity required")
        measured_tcp = np.asarray(row["measured_source"], dtype=float) @ tool
        period_velocity = dt * float(velocity[0] - velocity[1]) / 0.079
        max_difference = max(
            max_difference,
            compare_recorded(period_velocity, float(row["measured_period_velocity"]), atol=atol),
            compare_recorded(panda_opening_from_motor_target(motor), float(row["motor_opening_before"]), atol=atol),
        )
        if float(row["opening_target"]) != float(openings[offset]):
            raise ValueError("Original opening target differs from its generated endpoint")
        result = controller.update(
            float(row["opening_before"]),
            motor,
            period_velocity,
            poses,
            openings,
            offset,
            measured_tcp,
            previous_command=previous,
            dt=dt,
        )
        max_difference = max(max_difference, compare_recorded(asdict(result), row["gripper"], atol=atol))
        executed = float(row["command"][6])
        if result.command != executed or result.route not in routes:
            raise ValueError("Reconstructed command or route differs from the completed actual action")
        routes[result.route] += 1
        if result.plan is not None:
            planner_calls += 1
            candidate_sequences += (
                result.plan["valid_sequences"]
                + result.plan["rejected_nonfinite_sequences"]
                + result.plan["rejected_velocity_sequences"]
            )
        previous = executed
        ledger.update(
            steps_completed=step + 1,
            planner_calls_completed=planner_calls,
            candidate_sequences_completed=candidate_sequences,
        )
        if progress is not None:
            progress(dict(ledger))
    return controller, {
        "steps": n,
        "routes": routes,
        "planner_calls": planner_calls,
        "candidate_sequences": candidate_sequences,
        "max_recorded_difference": max_difference,
        "last_completed_command": previous,
        "latch_state": loaded_state(controller),
    }
