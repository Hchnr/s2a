"""Pinned public DROID reference-interface algebra, without robot execution.

cartesian_velocity is dimensionless, NOT SI velocity. Rotation is extrinsic
xyz Euler of the composed base-frame rotation difference. Gripper is position.
"""

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from s2a.geometry import advance, finite, transform, validate_transform


@dataclass(frozen=True)
class DroidReferenceInterface:
    max_translation_delta: float = 0.075
    max_rotation_delta: float = 0.15
    gripper_action_space: str = "position"

    def __post_init__(self):
        if (
            not np.isfinite([self.max_translation_delta, self.max_rotation_delta]).all()
            or min(self.max_translation_delta, self.max_rotation_delta) <= 0
        ):
            raise ValueError("DROID interface scales must be positive")
        if self.gripper_action_space != "position":
            raise ValueError("this adapter requires independent gripper position mode")

    def encode_target(self, desired, measured, limit=True):
        desired, measured = validate_transform(desired), validate_transform(measured)
        diff = desired[..., :3, :3] @ measured[..., :3, :3].swapaxes(-1, -2)
        command = np.concatenate(
            [
                (desired[..., :3, 3] - measured[..., :3, 3]) / self.max_translation_delta,
                Rotation.from_matrix(diff).as_euler("xyz") / self.max_rotation_delta,
            ],
            axis=-1,
        )
        return self.limit(command) if limit else command

    def limit(self, command):
        command = finite(command).copy()
        if command.shape[-1] != 6:
            raise ValueError("expected 6D arm command")
        for part in [slice(0, 3), slice(3, 6)]:
            command[..., part] /= np.maximum(1, np.linalg.norm(command[..., part], axis=-1, keepdims=True))
        return command

    def limit_rate(self, command, previous, elapsed, translation_rate, rotation_rate):
        """Final dimensionless command rate bound, after every inverse/encoder.

        Separate 3D norm limits preserve the unit-ball command constraint.
        """
        if (
            not np.isfinite([elapsed, translation_rate, rotation_rate]).all()
            or min(elapsed, translation_rate, rotation_rate) <= 0
        ):
            raise ValueError("command-rate interval and limits must be positive")
        sent, previous = self.limit(command), self.limit(finite(previous, (6,)))
        change = sent - previous
        for part, rate in [(slice(0, 3), translation_rate), (slice(3, 6), rotation_rate)]:
            norm = np.linalg.norm(change[part])
            if norm > rate * elapsed:
                change[part] *= rate * elapsed / norm
        return previous + change

    def decode_target(self, command, measured):
        command, measured = self.limit(command), validate_transform(measured)
        rotation = Rotation.from_euler("xyz", command[..., 3:] * self.max_rotation_delta).as_matrix()
        return transform(
            measured[..., :3, 3] + command[..., :3] * self.max_translation_delta, rotation @ measured[..., :3, :3]
        )

    def encode_velocity(self, base_velocity, measured, dt_command):
        return self.encode_target(advance(measured, base_velocity, dt_command), measured)

    @staticmethod
    def gripper_position(opening):
        opening = finite(opening)
        if np.any((opening < 0) | (opening > 1)):
            raise ValueError("gripper opening must be in [0,1]")
        return 1 - opening
