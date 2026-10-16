"""Pose-to-OSC encoding and causal inverse of the fixed inner PD response.

The small-motion model is x'' = kp * (goal - x) - kd * x'.  It uses
the actual controller gains and planned simulation interval, not DROID gains.
Contact, torque saturation and nonlinear orientation error remain unmodelled.
"""

from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm

from s2a.geometry import finite, pose_error, validate_transform


@dataclass(frozen=True)
class OSCInterface:
    input_min: np.ndarray
    input_max: np.ndarray
    output_min: np.ndarray
    output_max: np.ndarray

    def __post_init__(self):
        for key in ("input_min", "input_max", "output_min", "output_max"):
            object.__setattr__(self, key, finite(getattr(self, key), (6,)).copy())
        if np.any(self.input_max <= self.input_min) or np.any(self.output_max <= self.output_min):
            raise ValueError("OSC affine ranges must be increasing")

    def encode(self, physical_delta):
        physical_delta = finite(physical_delta, (6,))
        raw = self.input_min + (physical_delta - self.output_min) * (
            (self.input_max - self.input_min) / (self.output_max - self.output_min)
        )
        command = np.clip(raw, self.input_min, self.input_max)
        return command, bool(np.any(np.abs(raw - command) > 1e-12))

    def decode(self, command):
        command = np.clip(finite(command, (6,)), self.input_min, self.input_max)
        return self.output_min + (command - self.input_min) * (
            (self.output_max - self.output_min) / (self.input_max - self.input_min)
        )


class OSCStateToAction:
    """Single-step endpoint tracker with an optional analytical inner-loop inverse.

    Velocity is a current measured base-frame twist at the OSC control point.
    `desired` is the already generated next reference endpoint at t + dt.
    No future observation, physics state restore, controller gain change or
    original expert action is used by this converter.
    """

    def __init__(self, interface, kp, kd, dt, *, compensate=False, inverse_weight=1.0):
        self.interface = interface
        self.kp, self.kd = finite(kp, (6,)), finite(kd, (6,))
        if np.any(self.kp <= 0) or np.any(self.kd <= 0):
            raise ValueError("fixed PD gains must be positive")
        if not np.isfinite(dt) or dt <= 0 or not 0 <= inverse_weight <= 1:
            raise ValueError("invalid planned dt or inverse weight")
        self.dt, self.compensate, self.inverse_weight = float(dt), bool(compensate), inverse_weight
        self.response, self.momentum = np.empty(6), np.empty(6)
        for i, (p, d) in enumerate(zip(self.kp, self.kd, strict=True)):
            evolution = expm(np.array([[0.0, 1.0], [-p, -d]]) * self.dt)
            self.response[i], self.momentum[i] = 1 - evolution[0, 0], evolution[0, 1]
        if np.any(self.response < 1e-5):
            raise ValueError("planned interval has insufficient inverse response")

    def convert(self, measured, desired, measured_velocity):
        measured, desired = validate_transform(measured), validate_transform(desired)
        if measured.shape != (4, 4) or desired.shape != (4, 4):
            raise ValueError("OSC conversion requires individual poses")
        velocity = finite(measured_velocity, (6,))
        endpoint_error = pose_error(desired, measured)
        if np.linalg.norm(endpoint_error[3:]) >= np.pi - 0.05:
            raise ValueError("orientation endpoint too close to pi")
        inverse_delta = (endpoint_error - self.momentum * velocity) / self.response
        delta = endpoint_error.copy()
        if self.compensate:
            delta += self.inverse_weight * (inverse_delta - endpoint_error)
        command, saturated = self.interface.encode(delta)
        sent_delta = self.interface.decode(command)
        return {
            "command": command,
            "endpoint_error": endpoint_error,
            "measured_velocity": velocity.copy(),
            "requested_delta": delta,
            "sent_delta": sent_delta,
            "predicted_motion": self.response * sent_delta + self.momentum * velocity,
            "saturated": saturated,
        }
