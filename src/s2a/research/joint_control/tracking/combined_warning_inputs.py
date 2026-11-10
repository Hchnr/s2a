"""Identify a first combined warning from a strictly bounded saved prefix."""

import xml.etree.ElementTree as ET

import numpy as np

from s2a.geometry import finite
from s2a.research.joint_control.limits.causal_warning import joint_warnings
from s2a.research.joint_control.limits.quality_warning import combined_warning


def first_combined_warning(states, completed_tcp_errors, record_xml):
    """Use current q/qvel and at most the preceding completed TCP error."""
    states, errors = finite(states), finite(completed_tcp_errors)
    if states.ndim != 2 or len(states) < 2 or errors.shape != (len(states) - 1, 6):
        raise ValueError("One completed error per transition in a nonempty bounded prefix is required")
    root = ET.fromstring(record_xml)
    joints = list(root.find("worldbody").iter("joint"))
    if [j.get("name") for j in joints[:7]] != [f"robot0_joint{i}" for i in range(1, 8)]:
        raise ValueError("Original seven joint addresses required")
    types = [joint.get("type", "hinge") for joint in joints]
    if any(joint_type not in ("hinge", "slide", "free") for joint_type in types):
        raise ValueError("Known original joint widths required")
    nq = sum(7 if joint_type == "free" else 1 for joint_type in types)
    limits = np.asarray([[float(x) for x in joint.get("range").split()] for joint in joints[:7]])
    if states.shape[1] < 1 + nq + 7:
        raise ValueError("Each endpoint must contain seven joint positions and velocities")
    warnings = []
    for endpoint, state in enumerate(states):
        velocity = joint_warnings(state[1:8], state[1 + nq : 1 + nq + 7], limits)["velocity_0p2s"]
        warning = combined_warning(velocity["alarm"], None if endpoint == 0 else errors[endpoint - 1])
        warnings.append(warning)
    if any(item["alarm"] for item in warnings[:-1]) or not warnings[-1]["alarm"]:
        raise ValueError("The bounded endpoint must be the first combined warning")
    margin = float(np.min(np.minimum(states[-1, 1:8] - limits[:, 0], limits[:, 1] - states[-1, 1:8])))
    if margin < 0.05:
        raise ValueError("The original joint buffer must hold at the warning")
    return {"endpoint": len(states) - 1, "minimum_margin_rad": margin, "warning": warnings[-1], "nq": nq}
