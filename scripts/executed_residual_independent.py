"""174 scalar-response replay with history bound to the actually sent command."""

import numpy as np
from causal_arm_independent_replay import CausalArmReplay


class ExecutedResidualReplay(CausalArmReplay):
    def commit_executed(self, measured, velocity, command):
        measured, velocity, command = (np.asarray(v, dtype=float) for v in (measured, velocity, command))
        assert measured.shape == (4, 4) and velocity.shape == command.shape == (6,)
        assert np.isfinite(measured).all() and np.isfinite(velocity).all() and np.isfinite(command).all()
        assert self.last is not None
        interface = self.interface
        assert np.all(command >= interface.input_min) and np.all(command <= interface.input_max)
        delta = interface.output_min + (command - interface.input_min) * (
            interface.output_max - interface.output_min
        ) / (interface.input_max - interface.input_min)
        motion = self.response[:3] * delta[:3] + self.momentum[:3] * velocity[:3]
        predicted_velocity = (
            self.kp[:3] * self.momentum[:3] * delta[:3]
            + (1 - self.response[:3] - self.kd[:3] * self.momentum[:3]) * velocity[:3]
        )
        self.last = measured[:3, 3].copy(), motion.copy(), predicted_velocity.copy()
        self.previous = command.copy()
        return dict(nominal_motion=motion, nominal_linear_velocity=predicted_velocity)
