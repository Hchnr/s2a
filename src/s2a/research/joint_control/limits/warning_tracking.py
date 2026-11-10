"""Generated-reference preview with an optional causal joint retreat term."""

import numpy as np

from s2a.controllers.osc import OSCInterface
from s2a.geometry import finite, inverse, validate_transform
from s2a.research.joint_control.limits.causal_warning import joint_warnings
from s2a.research.joint_control.preview.dependency_safe_preview_runtime import solve_preview


def site_references(targets_tcp, anchor_tcp, source_from_tcp, source_from_site):
    """Use the same fixed source/TCP/site extrinsics for references and anchor."""
    targets_tcp, anchor_tcp = validate_transform(targets_tcp), validate_transform(anchor_tcp)
    source_from_tcp = validate_transform(source_from_tcp)
    source_from_site = validate_transform(source_from_site)
    if targets_tcp.ndim != 3 or anchor_tcp.shape != (4, 4):
        raise ValueError("A generated chunk and one anchor are required")
    tcp_from_site = inverse(source_from_tcp) @ source_from_site
    return targets_tcp @ tcp_from_site, anchor_tcp @ tcp_from_site


def select_tracking_action(
    mass,
    jacobian,
    q,
    qvel,
    limits,
    site,
    poses,
    anchor,
    cursor,
    previous,
    contract,
    *,
    guard,
    solver=solve_preview,
):
    """Preview with zero innovation; add a small inward task delta if warned.

    The Jacobian term is a heuristic and does not certify actual joint motion.
    Future object states, observations and ungenerated references are not inputs.
    """
    mass, jacobian = finite(mass, (7, 7)), finite(jacobian, (6, 7))
    q, qvel = finite(q, (7,)), finite(qvel, (7,))
    limits, previous = finite(limits, (7, 2)), finite(previous, (6,))
    if type(guard) is not bool:
        raise ValueError("Guard mode must be explicit")
    interface = OSCInterface(*(contract[key] for key in ("input_min", "input_max", "output_min", "output_max")))
    if np.any(previous < interface.input_min - 1e-12) or np.any(previous > interface.input_max + 1e-12):
        raise ValueError("Previous command outside the controller box")
    preview = solver(
        mass,
        jacobian,
        site,
        jacobian @ qvel,
        poses,
        anchor,
        cursor,
        previous,
        contract,
        innovation=np.zeros(12),
    )
    nominal = finite(preview["command"], (6,)).copy()
    if (
        np.any(nominal < interface.input_min - 1e-12)
        or np.any(nominal > interface.input_max + 1e-12)
        or np.max(np.abs(nominal - previous)) > 0.25 + 1e-12
    ):
        raise ValueError("Preview must respect the original box and rate")
    warning = joint_warnings(q, qvel, limits)["static_0p10"]
    correction = np.zeros(6)
    requested, saturated = nominal.copy(), False
    if guard and warning["alarm"]:
        sign = 1.0 if warning["side"] == "lower" else -1.0
        correction = jacobian[:, warning["joint_index"]] * (sign * 0.3 * 0.05)
        requested, saturated = interface.encode(interface.decode(nominal) + correction)
    lower = np.maximum(interface.input_min, previous - 0.25)
    upper = np.minimum(interface.input_max, previous + 0.25)
    command = np.clip(requested, lower, upper)
    return {
        "command": command,
        "preview_command": nominal,
        "warning": warning,
        "guard_active": bool(guard and warning["alarm"]),
        "retreat_site_delta": correction,
        "interface_saturated": saturated,
        "rate_limited": bool(np.any(np.abs(command - requested) > 1e-12)),
        "rotation_goal_may_latch": bool(np.all(interface.decode(command)[3:] == 0)),
        "preview_current_index": cursor,
        "innovation": np.zeros(12),
    }
