"""Causal reobservation timing; generated poses and openings are untouched."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TrackingReplanConfig:
    min_interval_steps: int = 8
    max_interval_steps: int = 40
    position_limit_m: float = 0.04
    rotation_limit_rad: float = 0.4
    consecutive_steps: int = 2

    def __post_init__(self):
        for value in (self.min_interval_steps, self.max_interval_steps, self.consecutive_steps):
            if type(value) is not int or value <= 0:
                raise ValueError("intervals and consecutive_steps must be positive integers")
        if self.min_interval_steps > self.max_interval_steps:
            raise ValueError("minimum interval exceeds maximum")
        for value in (self.position_limit_m, self.rotation_limit_rad):
            if isinstance(value, (bool, np.bool_)) or not np.isfinite(value) or value <= 0:
                raise ValueError("tracking limits must be finite and positive")


class TrackingReplan:
    """Query before a physical step using only already completed endpoints.

    Either position OR rotation exceeding its limit constitutes one violation;
    mixed-axis violations on consecutive steps count together. Evidence can
    accumulate before the minimum interval. Query resets the counter; a normal
    endpoint also resets it. No controller or gripper history is modified.
    """

    def __init__(self, config=None):
        self.config = config if config is not None else TrackingReplanConfig()
        if not isinstance(self.config, TrackingReplanConfig):
            raise TypeError("requires TrackingReplanConfig")
        self.last_query_step = None
        self.last_observed_step = -1
        self.violations = 0

    def _next_step(self, step):
        if type(step) is not int or step != self.last_observed_step + 1:
            raise ValueError("decision/query must precede the next unobserved physical step")

    def decision(self, step, *, exhausted=False):
        self._next_step(step)
        if type(exhausted) is not bool:
            raise ValueError("exhausted must be boolean")
        age = 0 if self.last_query_step is None else step - self.last_query_step
        reason = None
        if self.last_query_step is None:
            reason = "initial"
        elif exhausted:
            reason = "reference_exhausted"
        elif age >= self.config.max_interval_steps:
            reason = "maximum_interval"
        elif age >= self.config.min_interval_steps and self.violations >= self.config.consecutive_steps:
            reason = "tracking_limit"
        return dict(
            physical_step=step,
            age_steps=age,
            previous_violation_count=self.violations,
            query_due=reason is not None,
            query_reason=reason,
        )

    def start_query(self, step):
        self._next_step(step)
        if self.last_query_step == step:
            raise ValueError("only one query per physical step")
        self.last_query_step = step
        self.violations = 0

    def observe(self, step, tcp_error):
        self._next_step(step)
        if self.last_query_step is None:
            raise ValueError("a reference must be queried before observing its endpoint")
        error = np.asarray(tcp_error, dtype=float)
        if error.shape != (6,) or not np.all(np.isfinite(error)):
            raise ValueError("TCP error must contain finite world position and SO(3) error vectors")
        position, rotation = float(np.linalg.norm(error[:3])), float(np.linalg.norm(error[3:]))
        exceeded = position > self.config.position_limit_m or rotation > self.config.rotation_limit_rad
        self.violations = self.violations + 1 if exceeded else 0
        self.last_observed_step = step
        return dict(
            position_error_m=position,
            rotation_error_rad=rotation,
            exceeded=exceeded,
            consecutive_violations_after_step=self.violations,
        )
