"""Bounded current-state inputs for an earlier causal velocity warning."""

import xml.etree.ElementTree as ET

import numpy as np

from s2a.geometry import finite
from s2a.research.joint_control.limits.causal_warning import joint_warnings


def first_velocity_warning(states, record_xml):
    """Recompute only current-or-past alarms up through the provided endpoint."""
    states = finite(states)
    if states.ndim != 2 or not len(states):
        raise ValueError("A bounded nonempty endpoint prefix is required")
    xml = ET.fromstring(record_xml)
    joints = list(xml.find("worldbody").iter("joint"))
    if [j.get("name") for j in joints[:7]] != [f"robot0_joint{i}" for i in range(1, 8)]:
        raise ValueError("Original seven joint addresses required")
    types = [j.get("type", "hinge") for j in joints]
    if any(joint_type not in ("hinge", "slide", "free") for joint_type in types):
        raise ValueError("Known original joint widths required")
    nq = sum(7 if joint_type == "free" else 1 for joint_type in types)
    limits = np.asarray([[float(x) for x in j.get("range").split()] for j in joints[:7]])
    if states.shape[1] < 1 + nq + 7:
        raise ValueError("Endpoint state must contain seven joint positions and velocities")
    results = [joint_warnings(state[1:8], state[1 + nq : 1 + nq + 7], limits)["velocity_0p2s"] for state in states]
    if any(item["alarm"] for item in results[:-1]) or not results[-1]["alarm"]:
        raise ValueError("Only the first causal velocity-warning endpoint is eligible")
    margin = float(np.min(np.minimum(states[-1, 1:8] - limits[:, 0], limits[:, 1] - states[-1, 1:8])))
    if margin < 0.05:
        raise ValueError("The original joint buffer must hold at the warning")
    return {"endpoint": len(states) - 1, "minimum_margin_rad": margin, "warning": results[-1], "nq": nq}


def current_recorded_decision(row, *, step, previous_command):
    """Extract only the logged decision and control-before quantities."""
    if type(step) is not int or step < 1 or row["step"] != step:
        raise ValueError("Exact current decision endpoint required")
    previous = finite(previous_command, (7,))
    refinement, dynamics, current = row["tracking_refinement"], row["coupled_dynamics"], row["qualified_forward_input"]
    plan = finite(refinement["planned_commands"], (8, 6)).copy()
    command = finite(row["command"], (7,))
    if not np.array_equal(plan[0], command[:6]):
        raise ValueError("The logged original plan must begin with its original command")
    if np.max(abs(plan)) > 1 + 1e-12 or np.max(abs(np.diff(np.vstack((previous[:6], plan)), axis=0))) > 0.25 + 1e-12:
        raise ValueError("Original seed must satisfy the controller box and rate")
    return {
        "plan": plan,
        "nominal_response": finite(refinement["nominal_response"], (12,)).copy(),
        "mass": finite(dynamics["mass"], (7, 7)).copy(),
        "jacobian": finite(dynamics["jacobian"], (6, 7)).copy(),
        "q": finite(dynamics["robot_joint_position"], (7,)).copy(),
        "velocity": finite(dynamics["robot_joint_velocity"], (7,)).copy(),
        "site": finite(current["x_site_pose"], (1, 4, 4))[0].copy(),
        "twist": finite(current["x_site_twist"], (1, 6))[0].copy(),
        "chunk_index": row["chunk_index"],
        "cursor": row["chunk_physical_step"],
    }
