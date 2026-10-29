"""Prequential joint position error history, with explicit period ownership."""

import numpy as np

from s2a.geometry import finite


class CausalJointError:
    """Stage a prediction before observing its period; update after completion.

    The running maximum is empirical. It does not bound future model errors.
    No history means zero additional tightening, explicitly marked unavailable.
    """

    def __init__(self, joints=7):
        if not isinstance(joints, int) or joints < 1:
            raise ValueError("Positive joint count required")
        self.joints = joints
        self._maximum = np.zeros(joints)
        self._pending = None
        self._last = None
        self.completed = 0

    def snapshot(self):
        return dict(
            available=self.completed > 0,
            completed_periods=self.completed,
            last_completed_step=self._last,
            position_error_rad=self._maximum.copy(),
            future_guarantee=False,
        )

    def stage(self, step, predicted_inner):
        if not isinstance(step, int) or step < 0 or self._pending is not None:
            raise ValueError("One unobserved nonnegative period at a time")
        if self._last is not None and step != self._last + 1:
            raise ValueError("Consecutive periods required")
        predicted = finite(predicted_inner, (25, self.joints)).copy()
        self._pending = (step, predicted)

    def complete(self, step, actual_inner):
        if self._pending is None or self._pending[0] != step:
            raise ValueError("Matching prior staged prediction required")
        actual = finite(actual_inner, (25, self.joints))
        residual = np.max(abs(actual - self._pending[1]), axis=0)
        self._maximum = np.maximum(self._maximum, residual)
        self._last = step
        self.completed += 1
        self._pending = None
        return residual.copy()
