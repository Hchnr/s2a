"""Finite causal history correction of a frozen forward predictor."""

from collections import deque

import numpy as np
from scipy.spatial.transform import Rotation


class CausalLocalResponse:
    """Predict reads past completed periods; observe must follow endpoint arrival."""

    def __init__(self):
        self.history = deque(maxlen=20)
        self.updates = 0

    @staticmethod
    def feature(action):
        action = np.asarray(action, dtype=float)
        if action.shape != (3,) or not np.isfinite(action).all():
            raise ValueError("Finite standardized actual translation goal required")
        return np.r_[1.0, action]

    @staticmethod
    def vector(value):
        value = np.asarray(value, dtype=float)
        if value.shape != (12,) or not np.isfinite(value).all():
            raise ValueError("Finite motion/world rotation increment/scaled twist12 required")
        return value

    def observe(self, action, prior, realized):
        x = self.feature(action)
        prior, realized = self.vector(prior), self.vector(realized)
        residual = realized - prior
        residual[3:6] = (Rotation.from_rotvec(realized[3:6]) * Rotation.from_rotvec(prior[3:6]).inv()).as_rotvec()
        self.history.append((x.copy(), residual.copy()))
        self.updates += 1

    def coefficients(self):
        if not self.history:
            return np.zeros((4, 12))
        x = np.stack([row[0] for row in self.history])
        y = np.stack([row[1] for row in self.history])
        return np.linalg.solve(x.T @ x / len(x) + np.eye(4), x.T @ y / len(x))

    def predict(self, action, prior):
        prior = self.vector(prior)
        correction = self.feature(action) @ self.coefficients()
        result = prior + correction
        result[3:6] = (Rotation.from_rotvec(correction[3:6]) * Rotation.from_rotvec(prior[3:6])).as_rotvec()
        return result
