"""Anchored, timed SE(3) references; never reanchor on every measured state."""

import numpy as np
from scipy.interpolate import CubicHermiteSpline
from scipy.spatial.transform import Rotation, RotationSpline

from s2a.geometry import finite, integrate_local, transform, validate_transform


class ReferenceTrajectory:
    def __init__(self, times, poses, *, start_velocity=None):
        self.times, self.poses = finite(times).copy(), validate_transform(poses).copy()
        if self.times.ndim != 1 or len(self.times) < 2 or self.poses.shape != (len(self.times), 4, 4):
            raise ValueError("trajectory needs >=2 matching timed poses")
        if not np.all(np.diff(self.times) > 0):
            raise ValueError("reference times must strictly increase")
        p = self.poses[:, :3, 3]
        slopes = np.gradient(p, self.times, axis=0)
        if start_velocity is not None:
            slopes[0] = finite(start_velocity, (6,))[:3]
        self.position = CubicHermiteSpline(self.times, p, slopes)
        self.rotation = RotationSpline(self.times, Rotation.from_matrix(self.poses[:, :3, :3]))

    @classmethod
    def from_deltas(cls, anchor, deltas, times):
        return cls(times, integrate_local(anchor, deltas))

    def sample(self, timestamp):
        if not np.isfinite(timestamp) or timestamp < self.times[0]:
            raise ValueError("reference requested before generated horizon")
        if timestamp >= self.times[-1]:
            return self.poses[-1].copy(), np.zeros(6)
        rotation = self.rotation(timestamp)
        pose = transform(self.position(timestamp), rotation.as_matrix())
        # RotationSpline angular rate is body-frame; PID uses base-frame.
        velocity = np.r_[self.position(timestamp, 1), rotation.apply(self.rotation(timestamp, 1))]
        return pose, velocity

    def splice(self, timestamp, future_times, future_deltas, blend_duration):
        """C1 switch through old reference at splice; retain PID state externally.

        Future deltas describe a pre-generated chunk anchored at the old reference.
        A smooth SO(3) correction bridges initial angular rate. It does not use
        future measurements, and may alter the new chunk near its start.
        """
        if not np.isfinite(blend_duration) or blend_duration <= 0:
            raise ValueError("positive blend duration required")
        anchor, velocity = self.sample(timestamp)
        future_times = finite(future_times)
        if future_times[0] != timestamp:
            raise ValueError("chunk must start at splice time")
        poses = integrate_local(anchor, future_deltas)
        new = ReferenceTrajectory(future_times, poses, start_velocity=velocity)
        return BlendedReference(self, new, timestamp, blend_duration)


class BlendedReference:
    """C1 blend of two generated references in a small-angle rotation chart."""

    def __init__(self, old, new, start, duration):
        self.old, self.new, self.start, self.duration = old, new, start, duration

    def sample(self, timestamp):
        if timestamp >= self.start + self.duration:
            return self.new.sample(timestamp)
        if timestamp < self.start:
            return self.old.sample(timestamp)
        old_pose, _ = self.old.sample(timestamp)
        new_pose, _ = self.new.sample(timestamp)
        x = (timestamp - self.start) / self.duration
        weight = x * x * (3 - 2 * x)
        delta = Rotation.from_matrix(new_pose[:3, :3] @ old_pose[:3, :3].T).as_rotvec()
        if np.linalg.norm(delta) > np.pi - 0.05:
            raise ValueError("chunk blend exceeds rotation chart")
        pose = transform(
            (1 - weight) * old_pose[:3, 3] + weight * new_pose[:3, 3],
            Rotation.from_rotvec(weight * delta).as_matrix() @ old_pose[:3, :3],
        )
        # Derivative of actual blended pose, computed entirely from generated references.
        eps = 1e-5
        if timestamp == self.start:
            return pose, self.old.sample(timestamp)[1]
        if timestamp - eps < self.start:
            eps = (timestamp - self.start) / 2
        from s2a.geometry import pose_error

        def blend_pose(t):
            a, _ = self.old.sample(t)
            b, _ = self.new.sample(t)
            z = np.clip((t - self.start) / self.duration, 0, 1)
            w = z * z * (3 - 2 * z)
            d = Rotation.from_matrix(b[:3, :3] @ a[:3, :3].T).as_rotvec()
            return transform((1 - w) * a[:3, 3] + w * b[:3, 3], Rotation.from_rotvec(w * d).as_matrix() @ a[:3, :3])

        velocity = pose_error(blend_pose(timestamp + eps), blend_pose(timestamp - eps)) / (2 * eps)
        return pose, velocity
