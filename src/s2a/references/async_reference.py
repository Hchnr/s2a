"""Causal, observation-timed references for a continuously running controller.

Inference does not stop the plant clock. A late response keeps its original
world anchor and timestamps; elapsed endpoints are skipped, never reanchored.
Without a fresh endpoint the last fixed pose/opening is held. The caller must
continue local feedback and preserve the gripper's internal command on hold.
"""

from dataclasses import dataclass

import numpy as np

from s2a.geometry import finite, integrate_local, validate_transform


@dataclass(frozen=True)
class TimedReference:
    sequence: int
    observed_at: float
    period: float
    poses: np.ndarray
    openings: np.ndarray

    def __post_init__(self):
        if isinstance(self.sequence, bool) or not isinstance(self.sequence, int) or self.sequence < 0:
            raise ValueError("sequence must be a nonnegative integer")
        if not np.isfinite(self.observed_at) or not np.isfinite(self.period) or self.period <= 0:
            raise ValueError("invalid observation time or period")
        poses, openings = validate_transform(self.poses).copy(), finite(self.openings).copy()
        if poses.ndim != 3 or len(poses) == 0 or openings.shape != (len(poses),):
            raise ValueError("one opening per nonempty reference endpoint is required")
        if np.any((openings < 0) | (openings > 1)):
            raise ValueError("reference openings must be clipped explicitly before admission")
        poses.flags.writeable = openings.flags.writeable = False
        object.__setattr__(self, "poses", poses)
        object.__setattr__(self, "openings", openings)

    @classmethod
    def from_actions(cls, sequence, observed_at, period, anchor, actions, *, horizon):
        actions = finite(actions)
        if actions.ndim != 2 or actions.shape[1] != 10:
            raise ValueError("expected physical local-delta/opening actions with ten active slots")
        if isinstance(horizon, bool) or not isinstance(horizon, int) or not 1 <= horizon <= len(actions):
            raise ValueError("invalid execution horizon")
        actions = actions[:horizon]
        return cls(
            sequence, observed_at, period, integrate_local(anchor, actions[:, :9])[1:], np.clip(actions[:, 9], 0, 1)
        )


class AsyncReferenceBuffer:
    """One active reference; arrival never changes its observation-time clock."""

    def __init__(self, initial_pose, initial_opening, *, max_age):
        self.held_pose = finite(initial_pose, (4, 4)).copy()
        validate_transform(self.held_pose)
        if not np.isfinite(initial_opening) or not 0 <= initial_opening <= 1:
            raise ValueError("invalid initial opening")
        if not np.isfinite(max_age) or max_age <= 0:
            raise ValueError("max_age must be finite and positive")
        self.held_opening, self.max_age = float(initial_opening), float(max_age)
        self.reference, self.latest_sequence, self.last_time = None, -1, -np.inf

    def _clock(self, now):
        if not np.isfinite(now) or now < self.last_time - 1e-9:
            raise ValueError("controller clock must be finite and monotonic")
        self.last_time = max(self.last_time, float(now))

    def admit(self, reference, *, now):
        self._clock(now)
        if reference.observed_at > now + 1e-9:
            return "future_observation"
        if reference.sequence <= self.latest_sequence:
            return "out_of_order"
        self.latest_sequence = reference.sequence
        if now - reference.observed_at >= self.max_age - 1e-9:
            return "expired"
        if now >= reference.observed_at + len(reference.poses) * reference.period - 1e-9:
            return "exhausted"
        if self.reference is not None and reference.observed_at < self.reference.observed_at - 1e-9:
            return "older_observation"
        self.reference = reference
        return "accepted"

    def sample(self, *, now):
        self._clock(now)
        reference = self.reference
        index, reason, sequence = None, "no_reference", None
        if reference is not None:
            sequence = reference.sequence
            index = int(np.floor((now - reference.observed_at + 1e-9) / reference.period))
            if now - reference.observed_at >= self.max_age - 1e-9:
                reason = "expired"
            elif index >= len(reference.poses):
                reason = "exhausted"
            else:
                self.held_pose = reference.poses[index].copy()
                self.held_opening = float(reference.openings[index])
                reason = "tracking"
        return dict(
            pose=self.held_pose.copy(),
            opening=self.held_opening,
            tracking=reason == "tracking",
            reason=reason,
            sequence=sequence,
            reference_index=index if reason == "tracking" else None,
        )
