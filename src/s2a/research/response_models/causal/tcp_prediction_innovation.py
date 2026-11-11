"""Prequential TCP prediction errors with frame-consistent SO(3) updates."""

import numpy as np
from scipy.spatial.transform import Rotation

from s2a.geometry import finite


def _error(value):
    value = finite(value, (6,)).copy()
    if np.linalg.norm(value[3:]) >= np.pi - 1e-6:
        raise ValueError("TCP orientation errors must remain away from the SO(3) pi ambiguity")
    return value


class CausalTCPPredictionInnovation:
    """Stage one prediction, then update only from its completed observation."""

    def __init__(self):
        self.completed = 0
        self._last_step = None
        self._pending = None
        self._last_translation = None
        self._last_rotation = None
        self._max_position = 0.0
        self._max_rotation = 0.0

    def stage(self, step, predicted_error):
        if type(step) is not int or step < 0 or self._pending is not None:
            raise ValueError("One unobserved nonnegative step is required")
        if self._last_step is not None and step != self._last_step + 1:
            raise ValueError("Completed TCP predictions must be consecutive")
        predicted = _error(predicted_error)
        available = self.completed > 0
        corrected, upper_position, upper_rotation = None, None, None
        if available:
            error_rotation = Rotation.from_rotvec(predicted[3:])
            corrected_rotation = error_rotation * Rotation.from_matrix(self._last_rotation).inv()
            corrected = np.r_[predicted[:3] + self._last_translation, corrected_rotation.as_rotvec()]
            upper_position = float(np.linalg.norm(predicted[:3]) + self._max_position)
            upper_rotation = float(np.linalg.norm(predicted[3:]) + self._max_rotation)
        self._pending = (step, predicted)
        return {
            "step": step,
            "completed_history": self.completed,
            "history_available": available,
            "raw": predicted.copy(),
            "previous_innovation": corrected,
            "past_maximum_bound_m": upper_position,
            "past_maximum_bound_rad": upper_rotation,
            "future_guarantee": False,
        }

    def complete(self, step, actual_error):
        if self._pending is None or self._pending[0] != step:
            raise ValueError("A matching prediction must be staged before its observation")
        actual = _error(actual_error)
        predicted = self._pending[1]
        translation = actual[:3] - predicted[:3]
        actual_rotation = Rotation.from_rotvec(actual[3:])
        predicted_rotation = Rotation.from_rotvec(predicted[3:])
        discrepancy = actual_rotation.inv() * predicted_rotation
        position_norm, rotation_norm = float(np.linalg.norm(translation)), float(discrepancy.magnitude())
        self._last_translation = translation.copy()
        self._last_rotation = discrepancy.as_matrix()
        self._max_position = max(self._max_position, position_norm)
        self._max_rotation = max(self._max_rotation, rotation_norm)
        self.completed += 1
        self._last_step = step
        self._pending = None
        return {
            "step": step,
            "position_innovation_m": translation.copy(),
            "position_innovation_norm_m": position_norm,
            "rotation_innovation_rad": discrepancy.as_rotvec(),
            "rotation_innovation_norm_rad": rotation_norm,
            "completed_history": self.completed,
        }
