"""Causal tracking-quality warning and retrospective endpoint comparison."""

import numpy as np

from s2a.geometry import finite

POSITION_CAP_M = 0.01
ROTATION_CAP_RAD = 0.05
JOINT_BUFFER_RAD = 0.05


def combined_warning(velocity_alarm, previous_tcp_error):
    """Use only the current velocity flag and the last completed pose error."""
    if not isinstance(velocity_alarm, (bool, np.bool_)):
        raise ValueError("The velocity warning must be a boolean")
    position = rotation = None
    position_alarm = rotation_alarm = False
    if previous_tcp_error is not None:
        error = finite(previous_tcp_error, (6,))
        position, rotation = float(np.linalg.norm(error[:3])), float(np.linalg.norm(error[3:]))
        position_alarm = position >= POSITION_CAP_M / 2
        rotation_alarm = rotation >= ROTATION_CAP_RAD / 2
    reasons = [
        name
        for name, active in (
            ("velocity", velocity_alarm),
            ("position", position_alarm),
            ("rotation", rotation_alarm),
        )
        if active
    ]
    return {
        "alarm": bool(reasons),
        "reasons": reasons,
        "previous_position_error_m": position,
        "previous_rotation_error_rad": rotation,
    }


def _first(flags):
    return next((i for i, value in enumerate(flags) if value), None)


def _relation(warning, event):
    if event is None:
        return "event_not_observed"
    if warning is None:
        return "warning_not_observed"
    if warning < event:
        return "before"
    return "same_endpoint" if warning == event else "after"


def summarize_observations(observations):
    """Compare fixed alarms against the original completed endpoint history.

    Input must be a contiguous prefix beginning at endpoint zero. The final
    actual endpoint after this prefix is intentionally not inferred.
    """
    if not observations:
        raise ValueError("A nonempty contiguous endpoint sequence is required")
    rows = []
    for endpoint, observation in enumerate(observations):
        if observation["endpoint"] != endpoint:
            raise ValueError("Endpoints must be contiguous and begin at zero")
        previous = observation["previous_tcp_error"]
        if (previous is None) != (endpoint == 0):
            raise ValueError("Only endpoint zero can lack a previous completed error")
        velocity = observation["warnings"]["velocity_0p2s"]["alarm"]
        combined = combined_warning(velocity, previous)
        position, rotation = combined["previous_position_error_m"], combined["previous_rotation_error_rad"]
        quality_failure = position is not None and (position > POSITION_CAP_M or rotation > ROTATION_CAP_RAD)
        stored = observation["previous_pose_caps_satisfied"]
        if stored is not None and (not isinstance(stored, bool) or stored == quality_failure):
            raise ValueError("Stored and independently calculated pose caps disagree")
        if (stored is None) != (endpoint == 0):
            raise ValueError("Stored pose caps must follow completed-error availability")
        margin = float(observation["minimum_margin_rad"])
        if not np.isfinite(margin):
            raise ValueError("Finite joint margin required")
        rows.append(
            {
                "endpoint": endpoint,
                "velocity_alarm": bool(velocity),
                "combined": combined,
                "quality_failure": bool(quality_failure),
                "joint_buffer_lost": margin < JOINT_BUFFER_RAD,
                "minimum_margin_rad": margin,
            }
        )
    quality_event = _first([row["quality_failure"] for row in rows])
    joint_event = _first([row["joint_buffer_lost"] for row in rows])
    rules = {}
    for name in ("velocity", "combined"):
        flags = [row["velocity_alarm"] if name == "velocity" else row["combined"]["alarm"] for row in rows]
        first = _first(flags)
        rules[name] = {
            "first_warning_endpoint": first,
            "warning_count": sum(flags),
            "first_warning": None if first is None else rows[first],
            "relative_to_first_quality_failure": _relation(first, quality_event),
            "relative_to_first_joint_buffer_loss": _relation(first, joint_event),
        }
    return {
        "first_quality_failure_endpoint": quality_event,
        "first_joint_buffer_loss_endpoint": joint_event,
        "unobserved_endpoint_after_prefix": len(rows),
        "rules": rules,
        "observations": rows,
    }
