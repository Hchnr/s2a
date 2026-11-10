"""Small causal safety-action candidates for fixed OSC development trials."""

import numpy as np

from s2a.controllers.osc import OSCInterface
from s2a.geometry import finite


def _rate_limited(target, previous, interface):
    target = finite(target, (6,))
    previous = finite(previous, (6,))
    if np.any(previous < interface.input_min - 1e-12) or np.any(previous > interface.input_max + 1e-12):
        raise ValueError("Previous action is outside the fixed controller input box")
    lower = np.maximum(interface.input_min, previous - 0.25)
    upper = np.minimum(interface.input_max, previous + 0.25)
    if np.any(lower > upper):
        raise ValueError("No rate-admitted action inside the controller input box")
    return np.clip(target, lower, upper)


def rate_to_zero(previous, interface):
    """Move toward zero; exact zero rotation may retain the previous OSC goal."""
    if not isinstance(interface, OSCInterface):
        raise TypeError("Fixed OSC interface required")
    command = _rate_limited(np.zeros(6), previous, interface)
    return {
        "command": command,
        "requested_command": np.zeros(6),
        "rotation_goal_may_latch": bool(np.all(command[3:] == 0)),
    }


def jacobian_retreat(q, limits, jacobian, previous, interface):
    """Map a fixed inward joint displacement through the current site Jacobian.

    This is a kinematic candidate only. Redundancy, contact and the fixed
    controller response can prevent the requested joint from moving inward.
    """
    if not isinstance(interface, OSCInterface):
        raise TypeError("Fixed OSC interface required")
    q = finite(q, (7,))
    limits = finite(limits, (7, 2))
    jacobian = finite(jacobian, (6, 7))
    if np.any(limits[:, 0] >= limits[:, 1]):
        raise ValueError("Joint limits must be increasing")
    margins = np.stack((q - limits[:, 0], limits[:, 1] - q), axis=1)
    joint, side = np.unravel_index(int(np.argmin(margins)), margins.shape)
    signed_delta = 0.3 * 0.05 * (1.0 if side == 0 else -1.0)
    physical_delta = jacobian[:, joint] * signed_delta
    requested, saturated = interface.encode(physical_delta)
    command = _rate_limited(requested, previous, interface)
    return {
        "command": command,
        "joint_index": int(joint),
        "side": "lower" if side == 0 else "upper",
        "current_margin_rad": float(margins[joint, side]),
        "signed_joint_delta_rad": signed_delta,
        "physical_site_delta": physical_delta,
        "requested_command": requested,
        "interface_saturated": saturated,
        "rotation_goal_may_latch": bool(np.all(command[3:] == 0)),
    }
