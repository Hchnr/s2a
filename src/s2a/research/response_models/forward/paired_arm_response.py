"""086 one-period translation-only intervention on a frozen native prefix."""

import numpy as np


def paired_arm_commands(native_commands, anchor, axis, offset):
    commands = np.asarray(native_commands, dtype=float)
    if commands.ndim != 2 or commands.shape[1] != 7 or not np.isfinite(commands).all():
        raise ValueError("Finite native7 commands required")
    if np.any(np.abs(commands) > 1):
        raise ValueError("Native commands must remain in the fixed normalized interface")
    if type(anchor) is not int or not 0 <= anchor < len(commands):
        raise ValueError("Existing native command anchor required")
    if axis is None:
        if offset != 0 or isinstance(offset, bool):
            raise ValueError("Baseline requires zero offset")
    elif type(axis) is not int or axis not in (0, 1, 2) or isinstance(offset, bool) or offset not in (-0.1, 0.1):
        raise ValueError("Registered translation axis and fixed +/-0.1 required")
    selected = commands[: anchor + 1].copy()
    if axis is not None:
        selected[anchor, axis] = np.clip(selected[anchor, axis] + offset, -1, 1)
    return selected
