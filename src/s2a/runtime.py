"""Hardware-free online adapter: available feedback + already generated chunk."""

import numpy as np

from s2a.control import PIDConfig, PosePID
from s2a.geometry import finite, integrate_local, inverse, pose_error, validate_transform
from s2a.interface import DroidReferenceInterface
from s2a.trajectory import BlendedReference, ReferenceTrajectory


class StateToActionAdapter:
    def __init__(self, H_FC, *, dt_command, pid_config=PIDConfig(), interface=None):
        self.H_FC = validate_transform(H_FC).copy()
        self.H_CF = inverse(self.H_FC)
        if not np.isfinite(dt_command) or dt_command <= 0:
            raise ValueError("dt_command must be planned and positive")
        self.dt_command = dt_command
        self.interface = interface or DroidReferenceInterface()
        self.pid = PosePID(pid_config)
        self.reset()

    def reset(self):
        self.pid.reset()
        self.reference = None
        self.chunk_times = None
        self.gripper_openings = None

    def set_chunk(self, anchor_tcp, delta_state, knot_times, *, blend_duration=0.2):
        """Anchor once using current TCP; knot_times define the planned clock.

        delta_state is physical source Action128 (or its first ten slots).
        Caller must mask invalid outputs and de-normalize model output first.
        Measured dt is not accepted as a future execution-time input.
        """
        delta = finite(delta_state)
        times = finite(knot_times)
        if delta.ndim != 2 or delta.shape[1] < 10 or len(times) != len(delta) + 1:
            raise ValueError("chunk must contain pose delta, gripper command and N+1 times")
        openings = delta[:, 9]
        self.interface.gripper_position(openings)  # validate before changing any runtime state
        flange_poses = integrate_local(validate_transform(anchor_tcp), delta[:, :9]) @ self.H_CF
        new = ReferenceTrajectory(times, flange_poses)
        reference = new if self.reference is None else BlendedReference(self.reference, new, times[0], blend_duration)
        if self.reference is not None and (not np.isfinite(blend_duration) or blend_duration <= 0):
            raise ValueError("replanning requires a positive blend duration")
        if self.pid.state.previous_time is not None and times[0] < self.pid.state.previous_time:
            raise ValueError("chunk starts before latest feedback; reset at episode change")
        self.reference, self.chunk_times, self.gripper_openings = reference, times.copy(), openings.copy()

    def update(self, measured_tcp, timestamp, *, now=None):
        if self.reference is None:
            raise ValueError("no generated reference chunk")
        measured = validate_transform(measured_tcp) @ self.H_CF
        desired, ff = self.reference.sample(timestamp)
        index = np.clip(
            np.searchsorted(self.chunk_times, timestamp, side="right") - 1, 0, len(self.gripper_openings) - 1
        )
        gripper = float(self.interface.gripper_position(self.gripper_openings[index]))

        def encode(velocity):
            arm = self.interface.encode_velocity(velocity, measured, self.dt_command)
            state = self.pid.state
            elapsed = self.dt_command if state.previous_time is None else timestamp - state.previous_time
            previous = np.zeros(6) if state.last_result is None else state.last_result["command"][:6]
            arm = self.interface.limit_rate(
                arm,
                previous,
                elapsed,
                self.pid.config.max_translation_acceleration * self.dt_command / self.interface.max_translation_delta,
                self.pid.config.max_rotation_acceleration * self.dt_command / self.interface.max_rotation_delta,
            )
            # Equivalent sent increment accounts for downstream vector saturation.
            sent_target = self.interface.decode_target(arm, measured)
            delivered = pose_error(sent_target, measured) / self.dt_command
            return delivered, np.r_[arm, gripper]

        result = self.pid.update(measured, desired, ff, timestamp, now=now, apply=encode)
        return {
            **result,
            "action_space": "cartesian_velocity",
            "gripper_action_space": "position",
            "dt_command": self.dt_command,
            "reference_pose_flange": desired.copy(),
        }
