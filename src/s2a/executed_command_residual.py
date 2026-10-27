"""173 bind causal physical-response history to the final executed arm command."""

import numpy as np

from s2a.causal_arm_residual import CausalArmResidualOSC
from s2a.geometry import finite, validate_transform


class ExecutedCommandResidualOSC(CausalArmResidualOSC):
    def reset(self):
        super().reset()
        self._commit_state = None

    def convert(self, measured, desired, measured_velocity, *, desired_linear_velocity):
        result = super().convert(measured, desired, measured_velocity, desired_linear_velocity=desired_linear_velocity)
        self._commit_state = (np.asarray(measured).copy(), np.asarray(measured_velocity).copy())
        return result

    def commit_executed(self, measured, measured_velocity, command):
        """Replace provisional history after blending, before the next observation.

        Uses the physical SOURCE frame of the original083/085 inverse, not a
        SITE displacement or a learned model prediction. No endpoint is invented.
        """
        measured = validate_transform(measured)
        velocity = finite(measured_velocity, (6,))
        command = finite(command, (6,))
        if measured.shape != (4, 4) or self._pending is None or self._commit_state is None:
            raise ValueError("An individual measured pose and current converted candidate required")
        if not np.allclose(measured, self._commit_state[0], atol=1e-12, rtol=0) or not np.allclose(
            velocity, self._commit_state[1], atol=1e-12, rtol=0
        ):
            raise ValueError("Commit must use the same current measured state as the candidate")
        if np.any(command < self.interface.input_min) or np.any(command > self.interface.input_max):
            raise ValueError("Executed command outside actual interface")
        delta = self.interface.decode(command)
        motion = self.response[:3] * delta[:3] + self.momentum[:3] * velocity[:3]
        c = self.kp[:3] * self.momentum[:3]
        d = 1 - self.response[:3] - self.kd[:3] * self.momentum[:3]
        predicted_velocity = c * delta[:3] + d * velocity[:3]
        self._pending = measured[:3, 3].copy(), motion.copy(), predicted_velocity.copy()
        self.previous_command = command.copy()
        self._commit_state = None
        return dict(nominal_motion=motion, nominal_linear_velocity=predicted_velocity)
