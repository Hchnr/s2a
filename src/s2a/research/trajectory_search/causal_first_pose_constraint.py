"""Additional past-only TCP prediction constraint for first-pose search."""

import numpy as np

from s2a.geometry import finite
from s2a.research.trajectory_search.innovation_capped_quality_solver import (
    corrected_error,
    corrected_error_jacobian,
    validated_corrected_caps,
)


class CausalFirstPoseConstraint:
    """Constrain the current first pose using a completed world-frame innovation.

    The caller owns the causality and independent audit of the supplied history.
    A previous discrepancy is an empirical correction, not a future bound.
    """

    def __init__(self, translation, rotation, *, position_cap_m=0.01, rotation_cap_rad=0.05):
        self.translation = finite(translation, (3,)).copy()
        self.rotation = finite(rotation, (3, 3)).copy()
        corrected_error(np.zeros(6), self.translation, self.rotation)
        self.caps = validated_corrected_caps(position_cap_m, rotation_cap_rad)

    def correction(self, raw):
        return corrected_error(raw, self.translation, self.rotation)

    def satisfied(self, raw):
        error = self.correction(raw)
        return bool(
            np.linalg.norm(error[:3]) <= self.caps[0] + 1e-10 and np.linalg.norm(error[3:]) <= self.caps[1] + 1e-10
        )

    def evaluate(self, score, rollout, raw, scale):
        """Return two normalized corrected-cap constraints and their Jacobian."""
        raw = finite(raw, (6,))
        scale = finite(scale, (48,))
        weights = np.array([100.0] * 3 + [20.0] * 3)
        np.testing.assert_allclose(finite(score["residual"], (96,))[:6] / weights, raw, atol=1e-12, rtol=0)
        raw_jac = finite(score["jacobian"], (96, 48))[:6] * scale[None, :] / weights[:, None]
        rotation_jac = finite(rollout["rotation_jacobian"], (8, 3, 48))[0] * scale[None, :]
        error = self.correction(raw)
        jac = corrected_error_jacobian(raw, raw_jac, rotation_jac, self.translation, self.rotation)
        values, rows = [], []
        for start, end, cap in ((0, 3, self.caps[0]), (3, 6, self.caps[1])):
            norm = np.linalg.norm(error[start:end])
            direction = error[start:end] / norm if norm > 1e-15 else np.zeros(3)
            values.append(1 - norm / cap)
            rows.append(-direction @ jac[start:end] / cap)
        return np.asarray(values), np.asarray(rows)

    def describe(self):
        return {
            "kind": "completed_tcp_innovation",
            "translation_innovation_m": self.translation.copy(),
            "rotation_discrepancy": self.rotation.copy(),
            "corrected_position_cap_m": self.caps[0],
            "corrected_rotation_cap_rad": self.caps[1],
            "future_guarantee": False,
        }
