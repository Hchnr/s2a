"""079 causal Panda opening response around a constant-current-velocity prior."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class VelocityResponse:
    next_opening: float
    next_period_velocity: float
    next_motor_target: float
    residual_opening: float
    residual_velocity: float


def motor_after_native_period(motor, command):
    if not np.isfinite([motor, command]).all() or not -1 <= motor <= 1 or not -1 <= command <= 1:
        raise ValueError("Finite normalized motor and native command required")
    value = float(motor)
    for _ in range(25):
        value = float(np.clip(value - 0.01 * np.sign(command), -1, 1))
    return value


def design_features(raw, mean, scale):
    raw = np.asarray(raw, dtype=float)
    mean = np.asarray(mean, dtype=float)
    scale = np.asarray(scale, dtype=float)
    if raw.shape[-1:] != (4,) or mean.shape != (4,) or scale.shape != (4,):
        raise ValueError("Four current-state/known-command features required")
    if not all(np.isfinite(v).all() for v in (raw, mean, scale)) or np.any(scale <= 0):
        raise ValueError("Finite fit statistics with positive scales required")
    z = (raw - mean) / scale
    return np.concatenate(
        [
            z,
            np.stack([z[..., i] * z[..., j] for i in range(4) for j in range(i, 4)], axis=-1),
            np.ones(z.shape[:-1] + (1,)),
        ],
        axis=-1,
    )


class VelocityResidualResponse:
    def __init__(self, parameters):
        if (
            parameters.get("velocity_units") != "normalized_opening_per_20Hz_period"
            or parameters.get("prior") != "constant_current_velocity"
        ):
            raise ValueError("Frozen079 units and prior required")
        self.mean = np.array(parameters["mean"], dtype=float, copy=True)
        self.scale = np.array(parameters["scale"], dtype=float, copy=True)
        self.coefficients = np.array(parameters["coefficients"], dtype=float, copy=True)
        if self.coefficients.shape != (15, 2) or not np.isfinite(self.coefficients).all():
            raise ValueError("Fifteen finite coefficients for each of two residual outputs required")
        design_features(np.zeros(4), self.mean, self.scale)
        for a in (self.mean, self.scale, self.coefficients):
            a.setflags(write=False)

    def predict(self, opening, motor, period_velocity, command):
        if not np.isfinite([opening, period_velocity]).all() or not 0 <= opening <= 1:
            raise ValueError("Current measured opening and finite period velocity required")
        next_motor = motor_after_native_period(motor, command)
        with np.errstate(over="ignore", invalid="ignore"):
            residual = (
                design_features([opening, motor, next_motor, period_velocity], self.mean, self.scale)
                @ self.coefficients
            )
            unbounded_q = opening + period_velocity + residual[0]
            next_v = period_velocity + residual[1]
        if not np.isfinite([unbounded_q, next_v]).all():
            raise ValueError("Nonfinite local rollout; no velocity clipping is permitted")
        return VelocityResponse(
            float(np.clip(unbounded_q, 0, 1)), float(next_v), next_motor, float(residual[0]), float(residual[1])
        )
