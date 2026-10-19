"""Causal opening-trajectory intent for a sign-controlled parallel gripper.

Observed aperture is not an actuator target: contact can stop fingers while a
closing command continues to apply force. This experimental adapter retains
direction through a flat reference and uses a hysteresis band to reject noise.
"""

import numpy as np


class OpeningIntent:
    """Map an already generated aperture reference to native -1 open/+1 close.

    No contact label or expert action is accepted. New episodes need a new
    instance. Direction reverses only after a reference excursion of `band`.
    """

    def __init__(self, measured_opening, *, band=0.02):
        if not np.isfinite(band) or not 0 < band < 1:
            raise ValueError("opening hysteresis band must be in (0, 1)")
        self.band = float(band)
        self.extreme = self._opening(measured_opening)
        self.direction = -1.0  # Conservative initial open; no implicit grasp.

    @staticmethod
    def _opening(value):
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("opening must be finite and within [0, 1]")
        return float(value)

    def update(self, desired_opening):
        desired = self._opening(desired_opening)
        if self.direction < 0:
            self.extreme = max(self.extreme, desired)
            if desired < self.extreme - self.band:
                self.direction, self.extreme = 1.0, desired
        else:
            self.extreme = min(self.extreme, desired)
            if desired > self.extreme + self.band:
                self.direction, self.extreme = -1.0, desired
        return self.direction


def opening_position_target(opening):
    """Historical LIBERO aperture -> two normalized position actuators."""
    opening = OpeningIntent._opening(opening)
    internal = (0.079 * opening + 0.001) / 0.04 - 1
    return np.array([internal, -internal])
