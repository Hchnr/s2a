"""Preserve the currently known Panda motor target without aperture inference."""

import numpy as np


def preserve_motor_target(current_action):
    """Expand reset scalar cache using the actuator's existing broadcast rule."""
    value = np.asarray(current_action, dtype=float)
    if value.shape not in ((1,), (2,)) or not np.isfinite(value).all() or np.any(np.abs(value) > 1):
        raise ValueError("Requires actual finite one/two-value normalized motor cache")
    return np.broadcast_to(value, (2,)).copy()
