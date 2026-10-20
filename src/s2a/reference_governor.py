"""Causal common-clock retiming of already generated world/TCP references.

This module preserves every generated endpoint, including its aperture. It
does not plan, infer a grasp, or use a task predicate. Piecewise linear world
translation and shortest SO(3) interpolation share a retimed clock. A new
chunk has an explicit bridge from the preceding admitted reference to its
measured anchor; that bridge consumes physical time but no model time.
"""

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from s2a.geometry import finite, pose_error, validate_transform


@dataclass(frozen=True)
class ReferenceGovernorConfig:
    max_translation_speed_m_s: float = 0.2
    max_rotation_speed_rad_s: float = 1.0
    max_opening_speed_s: float = 2.0
    position_soft_m: float = 0.01
    position_stop_m: float = 0.04
    rotation_soft_rad: float = 0.1
    rotation_stop_rad: float = 0.4

    def __post_init__(self):
        for name, value in vars(self).items():
            if isinstance(value, (bool, np.bool_)) or not np.isscalar(value) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if self.position_stop_m <= self.position_soft_m or self.rotation_stop_rad <= self.rotation_soft_rad:
            raise ValueError("tracking stop thresholds must exceed soft thresholds")


def interpolate_pose(start, end, fraction):
    result = start.copy()
    result[:3, 3] += fraction * (end[:3, 3] - start[:3, 3])
    relative = Rotation.from_matrix(end[:3, :3] @ start[:3, :3].T).as_rotvec()
    result[:3, :3] = Rotation.from_rotvec(fraction * relative).as_matrix() @ start[:3, :3]
    return result


class ReferenceGovernor:
    """One instance per independent rollout; persist it across model chunks.

    ``set_reference`` receives the current measured anchor and its generated
    absolute world/TCP endpoints. It snapshots arrays and preserves the
    previously admitted command reference. ``advance`` uses only the current
    measured TCP and the newly elapsed control interval. The caller separately
    bounds observation/requery intervals, including when progress is paused.
    """

    def __init__(self, *, config=None):
        self.config = config if config is not None else ReferenceGovernorConfig()
        if not isinstance(self.config, ReferenceGovernorConfig):
            raise ValueError("config must be ReferenceGovernorConfig")
        self._admitted = self._opening = None
        self._poses = None

    def set_reference(self, anchor, targets, openings, *, anchor_opening, model_dt):
        anchor, targets = validate_transform(anchor), validate_transform(targets)
        openings = finite(openings)
        if anchor.shape != (4, 4) or targets.ndim != 3 or len(targets) == 0:
            raise ValueError("requires one anchor and a nonempty sequence of endpoints")
        if openings.shape != (len(targets),) or np.any((openings < 0) | (openings > 1)):
            raise ValueError("endpoint apertures must match poses and lie in [0,1]")
        if not np.isscalar(anchor_opening) or not np.isfinite(anchor_opening) or not 0 <= anchor_opening <= 1:
            raise ValueError("anchor_opening must be in [0,1]")
        if not np.isscalar(model_dt) or not np.isfinite(model_dt) or model_dt <= 0:
            raise ValueError("model_dt must be positive and finite")
        poses = np.concatenate([anchor[None], targets]).copy()
        values = np.r_[anchor_opening, openings]
        nominal = np.arange(len(poses), dtype=float) * model_dt
        if self._admitted is not None:
            poses = np.concatenate([self._admitted[None], poses])
            values = np.r_[self._opening, values]
            nominal = np.r_[0.0, nominal]
        else:
            self._admitted, self._opening = anchor.copy(), float(anchor_opening)
        errors = pose_error(poses[1:], poses[:-1])
        c = self.config
        durations = np.maximum.reduce(
            [
                np.diff(nominal),
                np.linalg.norm(errors[:, :3], axis=1) / c.max_translation_speed_m_s,
                np.linalg.norm(errors[:, 3:], axis=1) / c.max_rotation_speed_rad_s,
                np.abs(np.diff(values)) / c.max_opening_speed_s,
            ]
        )
        self._poses, self._values, self._nominal = poses, values, nominal
        self._times = np.r_[0.0, np.cumsum(durations)]
        self._clock = 0.0
        return dict(retimed_duration_s=float(self._times[-1]), model_duration_s=float(nominal[-1]))

    def advance(self, measured_tcp, dt):
        measured = validate_transform(measured_tcp)
        if measured.shape != (4, 4) or not np.isscalar(dt) or not np.isfinite(dt) or dt <= 0:
            raise ValueError("requires one measured TCP and a positive finite interval")
        if self._poses is None:
            raise ValueError("set_reference must precede advance")
        error = pose_error(self._admitted, measured)
        position, rotation = float(np.linalg.norm(error[:3])), float(np.linalg.norm(error[3:]))
        c = self.config
        scale = float(
            np.clip(
                min(
                    (c.position_stop_m - position) / (c.position_stop_m - c.position_soft_m),
                    (c.rotation_stop_rad - rotation) / (c.rotation_stop_rad - c.rotation_soft_rad),
                ),
                0,
                1,
            )
        )
        self._clock = min(self._clock + dt * scale, self._times[-1])
        index = min(int(np.searchsorted(self._times, self._clock, side="right") - 1), len(self._poses) - 2)
        duration = self._times[index + 1] - self._times[index]
        fraction = (self._clock - self._times[index]) / duration if duration else 1.0
        self._admitted = interpolate_pose(self._poses[index], self._poses[index + 1], fraction)
        self._opening = float((1 - fraction) * self._values[index] + fraction * self._values[index + 1])
        model_time = float((1 - fraction) * self._nominal[index] + fraction * self._nominal[index + 1])
        return dict(
            target_tcp=self._admitted.copy(),
            opening=self._opening,
            model_time_s=model_time,
            retimed_clock_s=float(self._clock),
            tracking_scale=scale,
            preceding_position_error_m=position,
            preceding_rotation_error_rad=rotation,
            bridging=bool(index == 0 and self._nominal[1] == 0),
            complete=bool(self._clock >= self._times[-1]),
        )
