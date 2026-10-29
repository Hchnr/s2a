"""Absolute-unit TCP cap check shared by candidate selection."""

import numpy as np

from s2a.geometry import finite


def within_pose_caps(error, caps):
    """Internal 1e-11 guard leaves numerical room for the outer 1e-10 check."""
    error, caps = finite(error, (6,)), finite(caps, (2,))
    if np.any(caps <= 0):
        raise ValueError("Positive position/rotation caps required")
    return bool(np.linalg.norm(error[:3]) <= caps[0] + 1e-11 and np.linalg.norm(error[3:]) <= caps[1] + 1e-11)
