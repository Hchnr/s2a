"""228 completed coupled-model innovation, separate from the old150 nominal history."""

import numpy as np
from scipy.spatial.transform import Rotation

from s2a.current_coupled_response import nominal_prediction
from s2a.geometry import finite
from s2a.osc import OSCInterface
from s2a.runtime_response_history import RuntimeResponseHistory


class CoupledResponseHistory:
    def __init__(self):
        self.reset()

    def reset(self):
        self.innovation = np.zeros(12)
        self.expected_pose = None
        self.expected_twist = None
        self.pending = None
        self.completed_periods = 0

    def current(self, pose, twist):
        pose, twist = RuntimeResponseHistory.endpoint(pose, twist)
        if self.pending is not None:
            raise ValueError("Actual executed period must be observed before replanning")
        if self.expected_pose is not None and (
            not np.allclose(pose, self.expected_pose, atol=1e-9, rtol=0)
            or not np.allclose(twist, self.expected_twist, atol=1e-9, rtol=0)
        ):
            raise ValueError("Current state differs from last completed coupled observation")
        return self.innovation.copy()

    def commit_executed(self, pose, twist, model, translation_command, contract):
        self.current(pose, twist)
        if contract["control_freq"] != 20 or contract["control_frame"] != "MuJoCo world" or model["dt"] != 0.05:
            raise ValueError("Fixed20Hz world model required for coupled history")
        command = finite(translation_command, (3,))
        interface = OSCInterface(*(contract[key] for key in ("input_min", "input_max", "output_min", "output_max")))
        if np.any(command < interface.input_min[:3]) or np.any(command > interface.input_max[:3]):
            raise ValueError("Actual translation command outside fixed interface")
        delta = interface.output_min[:3] + (command - interface.input_min[:3]) * (
            interface.output_max[:3] - interface.output_min[:3]
        ) / (interface.input_max[:3] - interface.input_min[:3])
        prediction = nominal_prediction(model, twist, delta)
        self.pending = (np.asarray(pose).copy(), prediction.copy())
        return dict(executed_translation=command.copy(), physical_delta=delta, nominal_response=prediction.copy())

    def observe_completed(self, next_pose, next_twist):
        if self.pending is None:
            raise ValueError("Actual command must be committed before observing its endpoint")
        next_pose, next_twist = RuntimeResponseHistory.endpoint(next_pose, next_twist)
        pose, prediction = self.pending
        actual_rotation = Rotation.from_matrix(next_pose[:3, :3] @ pose[:3, :3].T)
        self.innovation = np.r_[
            next_pose[:3, 3] - pose[:3, 3] - prediction[:3],
            (actual_rotation * Rotation.from_rotvec(prediction[3:6]).inv()).as_rotvec(),
            0.05 * (next_twist - prediction[6:]),
        ]
        self.expected_pose, self.expected_twist = next_pose.copy(), next_twist.copy()
        self.pending = None
        self.completed_periods += 1
        return self.innovation.copy()
