"""Condition-local correction from completed periods, with fixed ridge shrinkage."""

from collections import deque

import numpy as np
from scipy.spatial.transform import Rotation

from s2a.anchor_kernel_forward import BLOCKS
from s2a.causal_local_response import CausalLocalResponse


class ConditionWeightedCausalResponse:
    def __init__(self, state_scale):
        scale = np.asarray(state_scale, dtype=float)
        if scale.shape != (89,) or not np.isfinite(scale).all() or np.any(scale <= 0):
            raise ValueError("Finite positive fixed89 training scales required")
        self.scale = scale.copy()
        self.history = deque(maxlen=20)
        self.updates = 0

    @staticmethod
    def condition(raw):
        raw = np.asarray(raw, dtype=float)
        if raw.shape != (89,) or not np.isfinite(raw).all():
            raise ValueError("Finite current89 condition required")
        return raw

    def observe(self, raw, action, prior, realized):
        condition = self.condition(raw)
        x = CausalLocalResponse.feature(action)
        prior = CausalLocalResponse.vector(prior)
        realized = CausalLocalResponse.vector(realized)
        residual = realized - prior
        residual[3:6] = (Rotation.from_rotvec(realized[3:6]) * Rotation.from_rotvec(prior[3:6]).inv()).as_rotvec()
        self.history.append((condition.copy(), x.copy(), residual.copy()))
        self.updates += 1

    def coefficients(self, raw):
        current = self.condition(raw)
        if not self.history:
            return np.zeros((4, 12))
        conditions, features, residuals = (np.stack([r[i] for r in self.history]) for i in range(3))
        difference = (conditions - current) / self.scale
        distance = sum(np.mean(difference[:, block] ** 2, axis=1) for block in BLOCKS) / len(BLOCKS)
        w = (1 + distance) ** -2 / len(features)
        return np.linalg.solve(features.T @ (w[:, None] * features) + np.eye(4), features.T @ (w[:, None] * residuals))

    def predict(self, raw, action, prior):
        prior = CausalLocalResponse.vector(prior)
        correction = CausalLocalResponse.feature(action) @ self.coefficients(raw)
        result = prior + correction
        result[3:6] = (Rotation.from_rotvec(correction[3:6]) * Rotation.from_rotvec(prior[3:6])).as_rotvec()
        return result
