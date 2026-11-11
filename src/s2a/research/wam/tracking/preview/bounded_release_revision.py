"""A bounded, causal response to conflicting generated release references.

The selector changes reference consumption, not the robot controller. It has
no object, contact, task, force or future-observation input and provides no
joint-safety or task-success guarantee. The caller must independently gate
every physical dispatch and compare actual state against the latest reference.
"""

from copy import deepcopy
from dataclasses import dataclass
from numbers import Real

import numpy as np
from scipy.spatial.transform import Rotation

from s2a.geometry import pose_error, validate_transform


def _scalar(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite scalar")
    return float(value)


@dataclass(frozen=True)
class ReleaseRevisionConfig:
    stride: int
    bridge_steps: int
    release_clearance: float
    opening_revision_gap: float
    current_position_gate_m: float
    current_rotation_gate_rad: float
    joint_buffer_rad: float

    def __post_init__(self):
        if (
            type(self.stride) is not int
            or type(self.bridge_steps) is not int
            or min(self.stride, self.bridge_steps) < 1
        ):
            raise ValueError("Positive integer stride and bridge length required")
        for name in (
            "release_clearance",
            "opening_revision_gap",
            "current_position_gate_m",
            "current_rotation_gate_rad",
            "joint_buffer_rad",
        ):
            if _scalar(getattr(self, name), name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.release_clearance >= 1 or self.opening_revision_gap >= 1:
            raise ValueError("Opening thresholds must be below one normalized aperture")


@dataclass(frozen=True)
class ReferenceChunk:
    """One already generated trajectory; index j is endpoint step+j+1."""

    step: int
    poses: np.ndarray
    openings: np.ndarray

    def __post_init__(self):
        if type(self.step) is not int or self.step < 0:
            raise ValueError("Reference generation step must be a nonnegative integer")
        poses = validate_transform(self.poses)
        openings = np.asarray(self.openings, dtype=float)
        if (
            poses.ndim != 3
            or not len(poses)
            or openings.shape != (len(poses),)
            or not np.isfinite(openings).all()
            or np.any((openings < 0) | (openings > 1))
        ):
            raise ValueError("A finite paired TCP/opening trajectory is required")
        poses, openings = poses.copy(), openings.copy()
        poses.setflags(write=False)
        openings.setflags(write=False)
        object.__setattr__(self, "poses", poses)
        object.__setattr__(self, "openings", openings)


def _same_chunk(a, b):
    return a.step == b.step and np.array_equal(a.poses, b.poses) and np.array_equal(a.openings, b.openings)


def _remaining(chunk, step):
    offset = step - chunk.step
    if offset < 0:
        raise ValueError("A reference cannot be used before it was generated")
    return chunk.poses[offset:].copy(), chunk.openings[offset:].copy()


def _mix(old_poses, old_openings, new_poses, new_openings, *, hold, bridge):
    """Blend only same-time generated endpoints, using the shortest SO(3) path."""
    poses, openings = new_poses.copy(), new_openings.copy()
    poses[:hold], openings[:hold] = old_poses[:hold], old_openings[:hold]
    for index in range(hold, hold + bridge):
        alpha = (index - hold + 1) / bridge
        if alpha == 1:
            continue
        old, new = old_poses[index], new_poses[index]
        poses[index, :3, 3] = (1 - alpha) * old[:3, 3] + alpha * new[:3, 3]
        relative = old[:3, :3].T @ new[:3, :3]
        poses[index, :3, :3] = (
            old[:3, :3] @ Rotation.from_rotvec(alpha * Rotation.from_matrix(relative).as_rotvec()).as_matrix()
        )
        openings[index] = (1 - alpha) * old_openings[index] + alpha * new_openings[index]
    return poses, openings


class BoundedReleaseRevision:
    """Retain one old pair for one stride, then bridge or return to latest.

    Reference selection is transactional. ``commit`` must only follow the
    actual completion of the entire selected stride; partial execution faults
    the instance until an explicit reset. A closing episode gets at most one
    hold, including when subsequent generated chunks continue to conflict.
    """

    def __init__(self, config):
        if not isinstance(config, ReleaseRevisionConfig):
            raise ValueError("An explicit release revision configuration is required")
        self.config = config
        self.reset()

    def reset(self):
        self._mode = "idle"
        self._held = None
        self._last_step = None
        self._last_latest = None
        self._staged = None
        self._faulted = False

    def state(self):
        return {
            "mode": self._mode,
            "last_completed_step": self._last_step,
            "latest_source_step": None if self._last_latest is None else self._last_latest.step,
            "held_source_step": None if self._held is None else self._held.step,
            "staged": self._staged is not None,
            "faulted": self._faulted,
        }

    def prepare(
        self,
        latest,
        *,
        committed,
        step,
        measured_tcp,
        measured_opening,
        closing_episode,
        previous_inner_margin_rad,
    ):
        if self._faulted:
            raise ValueError("A partially executed stride requires explicit reconstruction or reset")
        if self._staged is not None:
            raise ValueError("The previous selection must be committed or aborted")
        if (
            type(step) is not int
            or step < 0
            or (self._last_step is not None and step != self._last_step + self.config.stride)
        ):
            raise ValueError("Consecutive completed selection steps are required")
        if not isinstance(latest, ReferenceChunk) or (
            committed is not None and not isinstance(committed, ReferenceChunk)
        ):
            raise ValueError("Already generated reference chunks are required")
        if type(closing_episode) is not bool:
            raise ValueError("The completed gripper closing state must be explicit")
        measured = validate_transform(measured_tcp)
        opening = _scalar(measured_opening, "measured_opening")
        if measured.shape != (4, 4) or not 0 <= opening <= 1:
            raise ValueError("One current TCP and normalized opening are required")
        margin = (
            None
            if previous_inner_margin_rad is None
            else _scalar(previous_inner_margin_rad, "previous_inner_margin_rad")
        )
        if margin is not None and margin < self.config.joint_buffer_rad:
            raise ValueError("Observed prior inner joint-buffer loss requires a dispatch stop")
        if committed is not None and committed.step > latest.step:
            raise ValueError("The committed reference cannot be newer than the latest reference")
        if self._mode == "idle" and self._last_latest is not None and committed is not None:
            if not _same_chunk(committed, self._last_latest):
                raise ValueError("The committed reference must match the previously selected reference")
        if self._last_latest is not None:
            if latest.step < self._last_latest.step:
                raise ValueError("Generated reference times cannot go backwards")
            if latest.step == self._last_latest.step and not _same_chunk(latest, self._last_latest):
                raise ValueError("A previously committed generation step cannot change its reference")
        new_p, new_o = _remaining(latest, step)
        if not len(new_o):
            raise ValueError("The latest reference has no current endpoint")
        if committed is not None and committed.step > step:
            raise ValueError("The committed reference must already have been generated")

        c = self.config
        selected_p, selected_o = new_p.copy(), new_o.copy()
        next_mode, next_held = self._mode, None
        reason, source = "no_conflict", "latest"
        conflict, current_position, current_rotation = False, None, None
        old_source = None if committed is None else committed.step
        if not closing_episode:
            next_mode, reason = "idle", "not_closing"
        elif self._mode == "held":
            prior = self._held
            old_source = prior.step
            old_p, old_o = _remaining(prior, step)
            next_mode = "lockout"
            if margin is None:
                reason = "unknown_inner_margin"
            elif min(len(old_o), len(new_o)) < c.bridge_steps:
                reason = "bridge_overlap_expired"
            else:
                selected_p, selected_o = _mix(old_p, old_o, new_p, new_o, hold=0, bridge=c.bridge_steps)
                reason, source = "bounded_bridge", "bridge"
        elif self._mode == "lockout":
            reason = "episode_hold_already_used"
        elif committed is None:
            next_mode, reason = "idle", "no_committed_reference"
        else:
            old_p, old_o = _remaining(committed, step)
            if len(old_o):
                difference = pose_error(old_p[0], measured)
                current_position = float(np.linalg.norm(difference[:3]))
                current_rotation = float(np.linalg.norm(difference[3:]))
                conflict = bool(
                    new_o[0] > opening + c.release_clearance
                    and new_o[0] > old_o[0] + c.opening_revision_gap
                    and old_o[0] <= opening + c.release_clearance
                )
            if conflict and margin is None:
                reason = "unknown_inner_margin"
            elif conflict and (
                current_position > c.current_position_gate_m or current_rotation > c.current_rotation_gate_rad
            ):
                reason = "current_pose_mismatch"
            elif conflict and min(len(old_o), len(new_o)) < c.stride + c.bridge_steps:
                reason = "insufficient_overlap"
            elif conflict:
                selected_p, selected_o = _mix(old_p, old_o, new_p, new_o, hold=c.stride, bridge=c.bridge_steps)
                reason, source, next_mode = "bounded_hold", "committed", "held"
                next_held = ReferenceChunk(committed.step, committed.poses, committed.openings)

        decision = {
            "step": step,
            "source": source,
            "reason": reason,
            "latest_source_step": latest.step,
            "committed_source_step": old_source,
            "absolute_first_endpoint": step + 1,
            "conflict_without_safety": conflict,
            "previous_inner_margin_rad": margin,
            "current_position_error_m": current_position,
            "current_rotation_error_rad": current_rotation,
            "poses": selected_p,
            "openings": selected_o,
        }
        metadata = {key: deepcopy(value) for key, value in decision.items() if key not in ("poses", "openings")}
        latest_copy = ReferenceChunk(latest.step, latest.poses, latest.openings)
        self._staged = (decision, metadata, selected_p.copy(), selected_o.copy(), next_mode, next_held, latest_copy)
        return decision

    def commit(self, decision, *, completed_steps):
        if self._staged is None or decision is not self._staged[0]:
            raise ValueError("The current staged reference selection is required")
        if type(completed_steps) is not int or completed_steps != self.config.stride:
            raise ValueError("Only a completed full stride can commit reference history")
        _, metadata, poses, openings, mode, held, latest = self._staged
        if (
            set(decision) != set(metadata) | {"poses", "openings"}
            or {key: value for key, value in decision.items() if key not in ("poses", "openings")} != metadata
            or not np.array_equal(decision["poses"], poses)
            or not np.array_equal(decision["openings"], openings)
        ):
            raise ValueError("The staged reference selection was changed")
        self._mode, self._held, self._last_step, self._last_latest = mode, held, decision["step"], latest
        self._staged = None

    def abort(self, decision, *, completed_steps=0):
        if self._staged is None or decision is not self._staged[0]:
            raise ValueError("The current staged reference selection is required")
        if type(completed_steps) is not int or not 0 <= completed_steps < self.config.stride:
            raise ValueError("Abort requires a bounded incomplete stride")
        self._staged = None
        if completed_steps:
            self._held, self._faulted = None, True
