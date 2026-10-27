"""085 lag-one measured arm-response residual; no fit, filter, or task input."""

import numpy as np

from s2a.geometry import finite, pose_error, validate_transform
from s2a.pose_velocity_inverse_osc import PoseVelocityInverseOSC


class CausalArmResidualOSC(PoseVelocityInverseOSC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reset()

    def reset(self):
        self.previous_command = np.clip(np.zeros(6), self.interface.input_min, self.interface.input_max)
        self._pending = None
        self._cycle = 0

    def convert(self, measured, desired, measured_velocity, *, desired_linear_velocity):
        measured, desired = validate_transform(measured), validate_transform(desired)
        velocity = finite(measured_velocity, (6,))
        target_velocity = finite(desired_linear_velocity, (3,))
        if measured.shape != (4, 4) or desired.shape != (4, 4):
            raise ValueError("Individual current and desired source poses required")
        observed = self._pending is not None
        position_residual, velocity_residual = np.zeros(3), np.zeros(3)
        if observed:
            previous_position, nominal_motion, nominal_velocity = self._pending
            position_residual = measured[:3, 3] - previous_position - nominal_motion
            velocity_residual = velocity[:3] - nominal_velocity
        adjusted_desired = desired.copy()
        adjusted_desired[:3, 3] -= position_residual
        result = super().convert(
            measured,
            adjusted_desired,
            velocity,
            desired_linear_velocity=target_velocity - velocity_residual,
        )
        nominal_motion = result["predicted_motion"].copy()
        nominal_velocity = result["predicted_linear_velocity"].copy()
        # History uses the nominal response of the actual constrained command,
        # never the requested delta or the residual-corrected prediction.
        self._pending = measured[:3, 3].copy(), nominal_motion[:3].copy(), nominal_velocity.copy()
        result["endpoint_error"] = pose_error(desired, measured)
        result["desired_linear_velocity"] = target_velocity.copy()
        result["nominal_predicted_motion"] = nominal_motion.copy()
        result["nominal_predicted_linear_velocity"] = nominal_velocity.copy()
        result["predicted_motion"][:3] += position_residual
        result["predicted_linear_velocity"] += velocity_residual
        result["response_residual"] = dict(
            position=position_residual,
            linear_velocity=velocity_residual,
            observed_previous_endpoint=observed,
            cycle=self._cycle,
            previous_command_cycle=self._cycle - 1 if observed else None,
        )
        self._cycle += 1
        return result
