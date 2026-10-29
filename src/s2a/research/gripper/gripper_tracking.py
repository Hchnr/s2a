"""Causal aperture tracking through Panda's incremental position actuator.

The native command is -1/open, 0/retain the last motor target, +1/close.
It is neither an absolute observed aperture nor a force command. This adapter
uses only current aperture, the known motor position target, the command used
over the preceding interval and an already generated scalar reference. A
closing-target stall is a diagnostic candidate, never proof of object contact.

This is an experimental interface for the fixed LIBERO Panda contract. Partial
apertures have native command quantization and an explicit closing preload;
neither arbitrary opening tracking nor a calibrated grasp force is certified.
"""

import math
from dataclasses import dataclass
from numbers import Real

import numpy as np


def _scalar(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite scalar")
    return float(value)


def _opening(value, name):
    value = _scalar(value, name)
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be in [0, 1]")
    return value


def panda_opening_from_qpos(qpos):
    """Two measured joint positions in m -> the published clipped aperture."""
    joints = np.asarray(qpos, dtype=float)
    if joints.shape != (2,) or not np.isfinite(joints).all():
        raise ValueError("Panda gripper qpos must contain two finite joint positions")
    return float(np.clip((joints[0] - joints[1] - 0.001) / 0.079, 0, 1))


def panda_opening_from_motor_target(current_action):
    """Read the existing PandaGripper.current_action, without changing it.

    This is the known two-actuator normalized position target, not the scalar
    native direction and not a privileged contact or force observation.
    """
    target = np.asarray(current_action, dtype=float)
    if target.shape != (2,) or not np.isfinite(target).all() or np.any(np.abs(target) > 1):
        raise ValueError("Panda motor target must contain two finite values in [-1, 1]")
    return panda_opening_from_qpos(np.array([0.02, -0.02]) + 0.02 * target)


@dataclass(frozen=True)
class GripperTrackingConfig:
    """Explicit normalized-aperture and physical-time qualification settings."""

    reference_time_constant_s: float = 0.1
    tracking_deadband: float = 0.025
    motor_deadband: float = 0.04
    closing_preload: float = 0.04
    release_margin: float = 0.06
    release_excursion: float = 0.08
    release_confirmation_s: float = 0.15
    stall_target_gap: float = 0.08
    stall_speed_per_s: float = 0.04
    stall_confirmation_s: float = 0.2
    closed_limit: float = 0.03
    max_evidence_interval_s: float = 0.1

    def __post_init__(self):
        for name, value in vars(self).items():
            value = _scalar(value, name)
            if name in ("reference_time_constant_s", "closing_preload"):
                if value < 0:
                    raise ValueError(f"{name} must be nonnegative")
            elif value <= 0:
                raise ValueError(f"{name} must be positive")
        for name in (
            "tracking_deadband",
            "motor_deadband",
            "closing_preload",
            "release_margin",
            "release_excursion",
            "stall_target_gap",
            "closed_limit",
        ):
            if getattr(self, name) >= 1:
                raise ValueError(f"{name} must be below one normalized aperture")
        if self.release_margin < self.tracking_deadband:
            raise ValueError("release_margin must cover the tracking_deadband")


@dataclass(frozen=True)
class GripperTrackingResult:
    command: float
    state: str
    reason: str
    filtered_reference: float
    measured_opening: float
    motor_opening: float
    reference_error: float
    measured_velocity_per_s: float
    closing_episode: bool
    blocked_candidate: bool
    blocked_opening: float | None
    release_level: float | None
    reference_excursion: float
    release_evidence_s: float
    stall_evidence_s: float
    evidence_interval_valid: bool


class GripperTrackingAdapter:
    """A new instance/reset is required for every independent rollout.

    ``previous_command`` is the actual native scalar used over the preceding
    ``dt`` seconds, including any caller override. ``motor_opening`` comes from
    the current position target (helper above), not an estimated future state.
    Keep this instance across reference chunks. Call once per new observation.
    No reference/measurement from after this command is accepted or needed.
    """

    def __init__(self, measured_opening, *, config=None):
        self.config = config if config is not None else GripperTrackingConfig()
        if not isinstance(self.config, GripperTrackingConfig):
            raise ValueError("config must be GripperTrackingConfig")
        self.reset(measured_opening)

    def reset(self, measured_opening):
        measured = _opening(measured_opening, "measured_opening")
        self._filtered = measured
        self._measured = measured
        self._closing = False
        self._reference_min = measured
        self._blocked_opening = None
        self._release_s = self._stall_s = 0.0
        self._release_previous = self._stall_previous = False

    def update(self, desired_opening, measured_opening, *, motor_opening, previous_command, dt):
        # Validate all input before changing state, so a rejected sample is inert.
        desired = _opening(desired_opening, "desired_opening")
        measured = _opening(measured_opening, "measured_opening")
        motor = _opening(motor_opening, "motor_opening")
        previous = _scalar(previous_command, "previous_command")
        elapsed = _scalar(dt, "dt")
        if not -1 <= previous <= 1 or elapsed <= 0:
            raise ValueError("previous_command must be in [-1, 1] and dt must be positive")
        c = self.config
        alpha = 1.0 if c.reference_time_constant_s == 0 else -math.expm1(-elapsed / c.reference_time_constant_s)
        self._filtered += alpha * (desired - self._filtered)
        reference = self._filtered
        velocity = (measured - self._measured) / elapsed
        self._measured = measured
        valid_interval = elapsed <= c.max_evidence_interval_s + 1e-12
        if not valid_interval or (self._closing and previous < 0):
            # A gap does not certify a continuously observed stall/release.
            self._release_s = self._stall_s = 0.0
            self._release_previous = self._stall_previous = False
            self._blocked_opening = None
        if self._closing and previous < 0:
            self._closing = False  # Actual external opening invalidates a held-close inference.

        error = reference - measured
        if not self._closing and error < -c.tracking_deadband and motor > reference + c.motor_deadband:
            self._closing = True
            self._reference_min = reference
        if self._closing:
            self._reference_min = min(self._reference_min, reference)
            if self._blocked_opening is not None and measured < self._blocked_opening - c.tracking_deadband:
                self._blocked_opening = None  # Motion resumed; do not retain a stale width anchor.
                self._stall_s = 0.0
                self._stall_previous = False
            stall = (
                valid_interval
                and previous >= 0
                and measured > c.closed_limit
                and measured - motor >= c.stall_target_gap
                and abs(velocity) <= c.stall_speed_per_s
            )
            self._stall_s = self._stall_s + elapsed if stall and self._stall_previous else 0.0
            self._stall_previous = stall
            if self._stall_s + 1e-12 >= c.stall_confirmation_s:
                self._blocked_opening = measured

            anchor = max(measured, self._blocked_opening or 0.0)
            # Cap requirements by remaining travel: a near-full-open request
            # must still be capable of releasing a very wide aperture.
            release_level = min(
                anchor + min(c.release_margin, (1.0 - anchor) / 2),
                1.0 - c.tracking_deadband / 2,
            )
            excursion = reference - self._reference_min
            excursion_needed = min(c.release_excursion, (1.0 - self._reference_min) / 2)
            release = valid_interval and reference >= release_level and excursion >= excursion_needed
            self._release_s = self._release_s + elapsed if release and self._release_previous else 0.0
            self._release_previous = release
            if self._release_s + 1e-12 >= c.release_confirmation_s:
                self._closing = False
                self._blocked_opening = None
                self._stall_s = 0.0
                self._stall_previous = False
                command, state, reason = -1.0, "opening", "measured_clearance_and_reference_rise_confirmed"
            elif self._blocked_opening is not None:
                command, state, reason = 0.0, "blocked_hold", "retain_existing_closing_motor_target"
            elif release:
                command, state, reason = 0.0, "release_pending", "opening_evidence_not_yet_long_enough"
            elif motor > max(0.0, reference - c.closing_preload) + min(c.motor_deadband, c.closing_preload / 2):
                command, state, reason = 1.0, "closing", "drive_closing_position_target"
            else:
                command, state, reason = 0.0, "hold", "retain_motor_target_without_reversing_on_reference_noise"
        else:
            release_level, excursion = None, 0.0
            self._release_s = self._stall_s = 0.0
            self._release_previous = self._stall_previous = False
            if error > c.tracking_deadband and motor < reference - c.motor_deadband:
                command, state, reason = -1.0, "opening", "drive_opening_position_target"
            else:
                command, state, reason = 0.0, "hold", "position_target_already_covers_reference_or_deadband"
        return GripperTrackingResult(
            command,
            state,
            reason,
            reference,
            measured,
            motor,
            error,
            velocity,
            self._closing,
            self._blocked_opening is not None,
            self._blocked_opening,
            release_level,
            excursion,
            self._release_s,
            self._stall_s,
            valid_interval,
        )
