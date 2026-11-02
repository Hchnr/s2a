"""Observe controller torques without forwarding, stepping or changing commands."""

import numpy as np


class ControllerTorqueTrace:
    def __init__(self, robot, observe):
        self.robot = robot
        self.controller = robot.controller
        self.observe = observe
        self.active = False
        self.records = []
        self.original_run = self.controller.run_controller
        self.original_control = robot.control
        self.pending = None
        self.controller.run_controller = self.run
        robot.control = self.control

    def run(self, *args, **kwargs):
        result = self.original_run(*args, **kwargs)
        if self.active:
            if self.pending is not None:
                raise RuntimeError("Unconsumed controller torque observation")
            self.pending = np.array(result, copy=True)
        return result

    def control(self, *args, **kwargs):
        result = self.original_control(*args, **kwargs)
        if self.active:
            if self.pending is None:
                raise RuntimeError("Robot control did not call controller exactly once")
            low, high = (np.array(x, copy=True) for x in self.robot.torque_limits)
            clipped = np.array(self.robot.torques, copy=True)
            np.testing.assert_array_equal(clipped, np.clip(self.pending, low, high))
            record = dict(raw_torque=self.pending.copy(), clipped_torque=clipped, low=low, high=high)
            record.update(self.observe())
            self.records.append(record)
            self.pending = None
        return result

    def close(self):
        self.robot.control = self.original_control
        self.controller.run_controller = self.original_run
