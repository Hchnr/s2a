"""Fit-only standardized ridge forward models with causal command/motion history.

The target is BASE-frame physical flange velocity at t+lag. Past observed
velocities are available only through t-1. Never feed current/future labels
into the inverse or model rollout. Models are local log-response approximations.
"""

from dataclasses import dataclass

import numpy as np

from s2a.data import split_indices
from s2a.geometry import finite


def history_vector(commands, velocities, index, history):
    return (
        np.concatenate([np.r_[commands[index - j], velocities[index - j]] for j in range(1, history + 1)])
        if history
        else np.empty(0)
    )


@dataclass
class LinearResponse:
    B: np.ndarray
    C: np.ndarray
    bias: np.ndarray
    lag: int
    history: int
    structure: str
    ridge: float
    response_scale: np.ndarray
    diagnostics: dict

    @classmethod
    def fit(cls, commands, velocities, indices, *, lag=0, history=0, structure="full", ridge=0.01):
        u, y = finite(commands), finite(velocities)
        ids = np.asarray(indices)
        if len(ids) < 12 or structure not in {"diagonal", "full"} or ridge <= 0:
            raise ValueError("insufficient fit support or invalid model configuration")
        x = np.array([np.r_[u[i], history_vector(u, y, i, history)] for i in ids])
        target = y[ids + lag]
        xm, ym = np.mean(x, axis=0), np.mean(target, axis=0)
        xs, ys = np.maximum(np.std(x, axis=0), 1e-6), np.maximum(np.std(target, axis=0), 1e-5)
        xn, yn = (x - xm) / xs, (target - ym) / ys
        coef = np.zeros((6, x.shape[1]))
        for axis in range(6):
            columns = np.arange(x.shape[1]) if structure == "full" else np.arange(axis, x.shape[1], 6)
            xx = xn[:, columns]
            coef[axis, columns] = np.linalg.solve(
                xx.T @ xx + ridge * len(ids) * np.eye(len(columns)), xx.T @ yn[:, axis]
            )
        coef = coef * ys[:, None] / xs
        bias = ym - coef @ xm
        b = coef[:, :6]
        scaled_b = b / ys[:, None] * np.maximum(np.std(u[ids], axis=0), 1e-6)
        singular = np.linalg.svd(scaled_b, compute_uv=False)
        condition = float(singular[0] / max(singular[-1], 1e-15))
        diagnostics = {
            "fit_samples": len(ids),
            "B_rank": int(np.linalg.matrix_rank(scaled_b)),
            "B_scaled_condition": condition,
            "B_scaled_singular_values": singular,
            "command_std_fit": np.std(u[ids], axis=0),
            "velocity_std_fit": ys,
            "feature_rank": int(np.linalg.matrix_rank(xn)),
            "feature_count": x.shape[1],
        }
        return cls(b, coef[:, 6:], bias, lag, history, structure, ridge, ys, diagnostics)

    def predict(self, command, causal_history):
        return self.B @ finite(command, (6,)) + self.C @ finite(causal_history, (12 * self.history,)) + self.bias

    def inverse(self, target_velocity, causal_history, analytic_prior, regularization=0.1):
        """Regularized unconstrained solve. Projection is separate and reported."""
        target = finite(target_velocity, (6,)) - self.C @ finite(causal_history, (12 * self.history,)) - self.bias
        prior = finite(analytic_prior, (6,))
        if regularization <= 0:
            raise ValueError("positive inverse regularization required")
        a = self.B / self.response_scale[:, None]
        rhs = target / self.response_scale
        return np.linalg.solve(a.T @ a + regularization * np.eye(6), a.T @ rhs + regularization * prior)

    def as_dict(self):
        return {key: getattr(self, key) for key in self.__dataclass_fields__}


def temporal_support(frames, interval, history, lag):
    return split_indices(frames, interval, history=history, lag=lag)


def prediction_metrics(prediction, target, scale):
    residual = np.asarray(prediction) - target
    rmse = np.sqrt(np.mean(residual**2, axis=0))
    return {
        "rmse_per_axis": rmse,
        "nrmse_per_axis": rmse / scale,
        "balanced_nrmse": float(np.sqrt(np.mean((residual / scale) ** 2))),
        "samples": len(target),
    }


def evaluate_one_step(model, commands, velocities, ids):
    predictions = np.array(
        [model.predict(commands[i], history_vector(commands, velocities, i, model.history)) for i in ids]
    )
    target = velocities[ids + model.lag]
    return prediction_metrics(predictions, target, model.response_scale)


def replay_model(model, commands):
    """New predicted motions feed history, with an explicit lag queue.

    Each independent interval starts with zero command/motion history and delay
    queue. Warm-up uses only that interval's commands; no cross-split state.
    """
    u = np.asarray(commands)
    y = np.zeros_like(u)
    previous_u, previous_y = [], []
    queue = [np.zeros(6) for _ in range(model.lag)]
    for i, command in enumerate(u):
        h = (
            np.concatenate(
                [
                    np.r_[previous_u[-j], previous_y[-j]] if j <= len(previous_u) else np.zeros(12)
                    for j in range(1, model.history + 1)
                ]
            )
            if model.history
            else np.empty(0)
        )
        predicted = model.predict(command, h)
        queue.append(predicted)
        y[i] = queue.pop(0)
        previous_u.append(command.copy())
        previous_y.append(y[i].copy())
    return y
