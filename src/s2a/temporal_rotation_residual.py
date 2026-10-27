"""Reversible world-frame SO(3) path residual around endpoint trapezoidal rotation.

Physical packets use [motion, world rotation vector, dt*v_next, dt*omega_next].
Encoding may consume training labels. Runtime decoding must consume predicted
packets and the currently measured twist only.
"""

import numpy as np
from scipy.spatial.transform import Rotation


def _inputs(packet, current_twist, dt):
    value = np.asarray(packet, dtype=float)
    twist = np.asarray(current_twist, dtype=float)
    if value.ndim != 2 or value.shape[1] != 12 or not len(value):
        raise ValueError("Nonempty Nx12 response packet required")
    if twist.shape != (len(value), 6):
        raise ValueError("Matching Nx6 current world twist required")
    if not np.isfinite(value).all() or not np.isfinite(twist).all():
        raise ValueError("Finite response and current twist required")
    if np.ndim(dt) != 0 or not np.isfinite(dt) or dt <= 0:
        raise ValueError("Finite positive scalar control period required")
    trapezoid = Rotation.from_rotvec(0.5 * (dt * twist[:, 3:] + value[:, 9:12]))
    return value, trapezoid


def encode(physical_packet, current_twist, dt):
    """Encode observed training response; preserve all non-rotation channels."""
    value, trapezoid = _inputs(physical_packet, current_twist, dt)
    result = value.copy()
    result[:, 3:6] = (Rotation.from_rotvec(value[:, 3:6]) * trapezoid.inv()).as_rotvec()
    return result


def decode(predicted_packet, current_twist, dt):
    """Decode predicted latent response using current measured world twist."""
    value, trapezoid = _inputs(predicted_packet, current_twist, dt)
    result = value.copy()
    result[:, 3:6] = (Rotation.from_rotvec(value[:, 3:6]) * trapezoid).as_rotvec()
    return result
