"""082 full translational inverse with the original half rotational compensation."""

import numpy as np

from s2a.geometry import finite, pose_error, validate_transform
from s2a.osc import OSCStateToAction


class TranslationInverseOSC(OSCStateToAction):
    def __init__(self, interface, kp, kd, dt, *, compensate=True, inverse_weight=0.5, max_command_rate=None):
        if compensate is not True or inverse_weight != 0.5:
            raise ValueError("082 requires fixed full translation and half rotation compensation")
        super().__init__(
            interface,
            kp,
            kd,
            dt,
            compensate=compensate,
            inverse_weight=inverse_weight,
            max_command_rate=max_command_rate,
        )

    def convert(self, measured, desired, measured_velocity):
        measured, desired = validate_transform(measured), validate_transform(desired)
        if measured.shape != (4, 4) or desired.shape != (4, 4):
            raise ValueError("OSC conversion requires individual poses")
        velocity = finite(measured_velocity, (6,))
        endpoint_error = pose_error(desired, measured)
        rotation_inverse_enabled = bool(np.linalg.norm(endpoint_error[3:]) <= 0.25)
        inverse_delta = (endpoint_error - self.momentum * velocity) / self.response
        # The inner orientation spring uses sin(angle), whereas the inverse
        # linearization uses angle. Preserve the baseline for large rotations.
        if not rotation_inverse_enabled:
            inverse_delta[3:] = endpoint_error[3:]
        delta = endpoint_error.copy()
        if self.compensate:
            weights = np.array([1.0, 1.0, 1.0, self.inverse_weight, self.inverse_weight, self.inverse_weight])
            delta += weights * (inverse_delta - endpoint_error)
        command, saturated = self.interface.encode(delta)
        rate_limited = False
        if self.max_command_rate is not None:
            change = self.max_command_rate * self.dt
            sent = np.clip(command, self.previous_command - change, self.previous_command + change)
            rate_limited = bool(np.any(np.abs(sent - command) > 1e-12))
            command = sent
        self.previous_command = command.copy()
        sent_delta = self.interface.decode(command)
        return {
            "command": command,
            "endpoint_error": endpoint_error,
            "measured_velocity": velocity.copy(),
            "requested_delta": delta,
            "sent_delta": sent_delta,
            "predicted_motion": self.response * sent_delta + self.momentum * velocity,
            "saturated": saturated,
            "rate_limited": rate_limited,
            "rotation_inverse_enabled": rotation_inverse_enabled,
        }
