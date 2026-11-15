"""Predict TCP translation innovation from completed commands and outcomes."""

import copy
from collections import deque

import numpy as np

from s2a.geometry import finite
from s2a.research.response_models.causal.tcp_prediction_innovation import CausalTCPPredictionInnovation

HISTORY_WINDOW = 20
RIDGE = 1.0


class CausalActionTCPResidual:
    """Keep a causal per-trajectory residual model; candidate queries are read-only.

    Position residuals use world-frame meters. Rotation prediction reuses the
    existing previous-innovation SO(3) composition. The fitted correction is
    empirical and does not certify a future physical error or action.
    """

    def __init__(self):
        self._innovation = CausalTCPPredictionInnovation()
        self._history = deque(maxlen=HISTORY_WINDOW)
        self._pending = None

    @staticmethod
    def _action(action):
        action = finite(action, (6,)).copy()
        if np.any(np.abs(action) > 1.0 + 1e-12):
            raise ValueError("The six encoded arm commands must remain in the original controller box")
        return action

    def predict(self, step, predicted_error, action):
        """Score any current candidate without staging or updating history."""
        if self._pending is not None:
            raise ValueError("The previously staged action must complete before another prediction")
        action = self._action(action)
        staged = copy.deepcopy(self._innovation).stage(step, predicted_error)
        corrected = None
        if staged["history_available"]:
            if not self._history:
                raise ValueError("Completed innovation and action histories must remain aligned")
            x = np.stack([item[0] for item in self._history])
            y = np.stack([item[1] for item in self._history])
            mean_x, mean_y = x.mean(axis=0), y.mean(axis=0)
            centered_x, centered_y = x - mean_x, y - mean_y
            count = len(x)
            coefficients = np.linalg.solve(
                centered_x.T @ centered_x / count + RIDGE * np.eye(6), centered_x.T @ centered_y / count
            )
            translation = mean_y + (action - mean_x) @ coefficients
            corrected = staged["previous_innovation"].copy()
            corrected[:3] = staged["raw"][:3] + translation
        return dict(
            step=step,
            completed_history=staged["completed_history"],
            fit_history=len(self._history),
            history_available=staged["history_available"],
            raw=staged["raw"].copy(),
            previous_innovation=None if staged["previous_innovation"] is None else staged["previous_innovation"].copy(),
            action_conditioned=corrected,
            ridge_fit_performed=corrected is not None,
            future_guarantee=False,
        )

    def stage(self, step, predicted_error, action):
        """Bind one selected prediction and command before its observation."""
        prediction = self.predict(step, predicted_error, action)
        action = self._action(action)
        staged = self._innovation.stage(step, predicted_error)
        if staged["completed_history"] != prediction["completed_history"]:
            raise ValueError("The prediction and staged history must remain identical")
        self._pending = (step, staged["raw"].copy(), action)
        return prediction

    def complete(self, step, actual_error):
        """Update only after the matching selected action completes."""
        if self._pending is None or self._pending[0] != step:
            raise ValueError("A matching prediction and action must be staged before observation")
        actual = finite(actual_error, (6,)).copy()
        innovation = self._innovation.complete(step, actual)
        _step, predicted, action = self._pending
        self._history.append((action.copy(), (actual[:3] - predicted[:3]).copy()))
        self._pending = None
        return innovation
