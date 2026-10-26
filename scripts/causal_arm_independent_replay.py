"""085 independent scalar-response/least-squares and causal endpoint replay."""

import numpy as np
from pose_velocity_arm_independent_replay import PoseVelocityArmReplay


class CausalArmReplay(PoseVelocityArmReplay):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last = None
        self.cycle = 0

    def convert(self, measured, desired, velocity, *, desired_linear_velocity):
        measured, desired, velocity = np.asarray(measured), np.asarray(desired), np.asarray(velocity)
        observed = self.last is not None
        dp, dv = np.zeros(3), np.zeros(3)
        if observed:
            origin, motion, next_velocity = self.last
            dp = measured[:3, 3] - origin - motion
            dv = velocity[:3] - next_velocity
        target = desired.copy()
        target[:3, 3] -= dp
        result = super().convert(
            measured, target, velocity, desired_linear_velocity=np.asarray(desired_linear_velocity) - dv
        )
        interface = self.interface
        u = result["command"]
        delta = interface.output_min + (u - interface.input_min) * (interface.output_max - interface.output_min) / (
            interface.input_max - interface.input_min
        )
        motion = self.response * delta + self.momentum * velocity
        next_velocity = result["predicted_linear_velocity"].copy()
        self.last = measured[:3, 3].copy(), motion[:3].copy(), next_velocity.copy()
        result["nominal_predicted_motion"] = motion
        result["nominal_predicted_linear_velocity"] = next_velocity
        result["predicted_linear_velocity"] = next_velocity + dv
        result["response_residual"] = dict(
            position=dp,
            linear_velocity=dv,
            observed_previous_endpoint=observed,
            cycle=self.cycle,
            previous_command_cycle=self.cycle - 1 if observed else None,
        )
        self.cycle += 1
        return result
