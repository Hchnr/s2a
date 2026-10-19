"""Explicit actuator-input delay for independent simulator stress tests."""

from collections import deque

import numpy as np


class ActuatorDelay:
    """Delay commands, never an autoregressive plant output or a state label."""

    def __init__(self, steps, dimensions=7, initial_command=None):
        if not isinstance(steps, int) or steps < 0 or dimensions < 1:
            raise ValueError("invalid actuator delay")
        self.dimensions = dimensions
        initial = np.zeros(dimensions) if initial_command is None else np.asarray(initial_command, dtype=float)
        if initial.shape != (dimensions,) or not np.isfinite(initial).all():
            raise ValueError("invalid initial actuator command")
        self.queue = deque(initial.copy() for _ in range(steps))

    def update(self, command):
        command = np.asarray(command, dtype=float)
        if command.shape != (self.dimensions,) or not np.isfinite(command).all():
            raise ValueError("invalid actuator command")
        self.queue.append(command.copy())
        return self.queue.popleft()
