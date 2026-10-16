"""Causal, per-instance task-space PID with final-command anti-windup."""

from dataclasses import dataclass, field

import numpy as np

from s2a.geometry import finite, pose_error, validate_transform


def limit_vectors(value, translation, rotation):
    result = np.asarray(value, dtype=float).copy()
    for part, bound in [(slice(0, 3), translation), (slice(3, 6), rotation)]:
        norm = np.linalg.norm(result[part])
        if norm > bound:
            result[part] *= bound / norm
    return result


@dataclass(frozen=True)
class PIDConfig:
    kp: tuple = (0.0,) * 6
    ki: tuple = (0.0,) * 6
    kd: tuple = (0.0,) * 6
    tau_d: float = 0.1
    integral_limit: float = 0.2
    max_translation_speed: float = 0.25
    max_rotation_speed: float = 1.0
    max_translation_acceleration: float = 2.0
    max_rotation_acceleration: float = 8.0
    max_feedback_gap: float = 0.25
    max_observation_age: float = 0.15
    derivative_on_measurement: bool = True
    pi_margin: float = 0.05

    def __post_init__(self):
        for key in ["kp", "ki", "kd"]:
            value = finite(getattr(self, key), (6,))
            if np.any(value < 0) or np.any(value > 100):
                raise ValueError("PID gains must be in [0,100]")
        for key in [
            "tau_d",
            "integral_limit",
            "max_translation_speed",
            "max_rotation_speed",
            "max_translation_acceleration",
            "max_rotation_acceleration",
            "max_feedback_gap",
            "max_observation_age",
            "pi_margin",
        ]:
            value = getattr(self, key)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be positive")


@dataclass
class PIDState:
    integral: np.ndarray = field(default_factory=lambda: np.zeros(6))
    derivative: np.ndarray = field(default_factory=lambda: np.zeros(6))
    previous_error: np.ndarray | None = None
    previous_measured: np.ndarray | None = None
    previous_time: float | None = None
    previous_velocity: np.ndarray = field(default_factory=lambda: np.zeros(6))
    last_result: dict | None = None


class PosePID:
    """dt_feedback uses past observations; dt_command belongs to the encoder.

    `apply(raw_velocity)` must include ALL downstream limits and return the
    delivered equivalent physical velocity plus an arbitrary command payload.
    It must be pure: a conditional-integration retry must not send twice.
    No hardware IO occurs here. A near-pi error is rejected rather than guessed.
    """

    def __init__(self, config=PIDConfig()):
        self.config = config
        self.reset()

    def reset(self):
        self.state = PIDState()

    def update(self, measured, desired, feedforward, timestamp, *, now=None, apply=None):
        c, s = self.config, self.state
        measured, desired = validate_transform(measured), validate_transform(desired)
        ff = finite(feedforward, (6,))
        now = timestamp if now is None else now
        if not np.isfinite([timestamp, now]).all() or now < timestamp:
            raise ValueError("invalid observation/decision timestamp")
        if now - timestamp > c.max_observation_age:
            # Freeze state; caller must obtain fresh feedback, no command generated.
            raise ValueError("stale observation: PID state frozen")
        error = pose_error(desired, measured)
        if np.linalg.norm(error[3:]) > np.pi - c.pi_margin:
            raise ValueError("SO(3) error too close to pi; replan through smaller rotations")
        if s.previous_time is not None and timestamp == s.previous_time:
            return s.last_result
        reset_reason = None
        if s.previous_time is not None and (
            timestamp < s.previous_time or timestamp - s.previous_time > c.max_feedback_gap
        ):
            reset_reason = "time_reversal" if timestamp < s.previous_time else "feedback_gap"
            self.reset()
            s = self.state
        dt = None if s.previous_time is None else timestamp - s.previous_time
        trial = s.integral.copy()
        derivative = np.zeros(6)
        if dt is not None:
            trial = np.clip(s.integral + error * dt, -c.integral_limit, c.integral_limit)
            if c.derivative_on_measurement:
                # Base-frame measured translational/angular velocity, avoids setpoint kick.
                raw_d = -pose_error(measured, s.previous_measured) / dt
            else:
                raw_d = (error - s.previous_error) / dt
            alpha = np.exp(-dt / c.tau_d)
            derivative = alpha * s.derivative + (1 - alpha) * raw_d
        kp, ki, kd = map(np.asarray, [c.kp, c.ki, c.kd])

        def compose(integral):
            raw = ff + kp * error + ki * integral + kd * derivative
            velocity = limit_vectors(raw, c.max_translation_speed, c.max_rotation_speed)
            if dt is not None:
                velocity = s.previous_velocity + limit_vectors(
                    velocity - s.previous_velocity,
                    c.max_translation_acceleration * dt,
                    c.max_rotation_acceleration * dt,
                )
            delivered, payload = (velocity, None) if apply is None else apply(velocity.copy())
            return raw, finite(delivered, (6,)), payload

        raw, delivered, payload = compose(trial)
        blocked = (raw - delivered) * (ki * error) > 1e-12
        trial[blocked] = s.integral[blocked]
        # Recompute once: downstream callback is pure; only final command is exposed.
        raw, delivered, payload = compose(trial)
        result = {
            "error": error.copy(),
            "integral": trial.copy(),
            "derivative": derivative.copy(),
            "raw_velocity": raw.copy(),
            "velocity": delivered.copy(),
            "command": payload,
            "saturated": bool(np.linalg.norm(raw - delivered) > 1e-9),
            "dt_feedback": dt,
            "reset_reason": reset_reason,
        }
        s.integral, s.derivative = trial.copy(), derivative.copy()
        s.previous_error, s.previous_measured = error.copy(), measured.copy()
        s.previous_time, s.previous_velocity, s.last_result = timestamp, delivered.copy(), result
        return result
