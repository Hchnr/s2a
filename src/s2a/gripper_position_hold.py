"""Quantized native motor-position tracking with an explicit closing latch.

Only generated aperture, current measured aperture, known motor target and
past control timing enter this fixed Panda candidate. Object/contact/task
labels and future observations are deliberately absent from its interface.
"""

import math
from dataclasses import dataclass
from numbers import Real

from s2a.gripper import OpeningIntent
from s2a.gripper_intent_hold import GripperIntentHoldConfig


def finite(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite scalar")
    return float(value)


def opening(value, name):
    value = finite(value, name)
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be in [0,1]")
    return value


@dataclass(frozen=True)
class GripperPositionHoldConfig(GripperIntentHoldConfig):
    close_threshold: float = 0.60
    motor_deadband: float = 0.07

    def __post_init__(self):
        super().__post_init__()
        if not self.reference_hysteresis < self.close_threshold < 1:
            raise ValueError("close threshold must be above intent hysteresis and below one")
        if not 0 < self.motor_deadband < 0.5:
            raise ValueError("motor deadband must be positive and below half the motor range")


@dataclass(frozen=True)
class GripperPositionHoldResult:
    command: float
    state: str
    reason: str
    reference: float
    measured_opening: float
    motor_opening: float
    normal_intent: float
    closing_episode: bool
    holding_width: float | None
    reference_minimum: float
    reference_excursion: float
    release_level: float | None
    release_evidence_s: float
    evidence_interval_valid: bool
    reference_motor_error: float


class GripperPositionHold:
    """Retain one instance per episode, including across generated chunks.

    The motor deadband is slightly larger than half one native20Hz increment
    (.126582 normalized aperture). Zero retains an existing target. The closed
    target preserves native motor loading; it is not a grasp-force estimate.
    """

    def __init__(self, measured_opening, *, config=None):
        self.config = config if config is not None else GripperPositionHoldConfig()
        if not isinstance(self.config, GripperPositionHoldConfig):
            raise ValueError("config must be GripperPositionHoldConfig")
        self.reset(measured_opening)

    def reset(self, measured_opening):
        measured = opening(measured_opening, "measured_opening")
        self.intent = OpeningIntent(measured, band=self.config.reference_hysteresis)
        self._closing = False
        self._reference_min = measured
        self._holding_width = None
        self._release_s = 0.0
        self._release_previous = False

    def _position_command(self, reference, motor):
        if reference > motor + self.config.motor_deadband:
            return -1.0, "increase_known_motor_position"
        if reference < motor - self.config.motor_deadband:
            return 1.0, "decrease_known_motor_position"
        return 0.0, "retain_quantized_motor_position"

    def update(self, desired_opening, measured_opening, *, motor_opening, previous_command, dt):
        reference = opening(desired_opening, "desired_opening")
        measured = opening(measured_opening, "measured_opening")
        motor = opening(motor_opening, "motor_opening")
        previous, elapsed = finite(previous_command, "previous_command"), finite(dt, "dt")
        if not -1 <= previous <= 1 or elapsed <= 0:
            raise ValueError("previous command must be in [-1,1] and dt positive")
        c = self.config
        normal = self.intent.update(reference)
        valid = elapsed <= c.max_evidence_interval_s + 1e-12
        if not valid or previous < 0:
            self._release_s = 0.0
            self._release_previous = False
            self._holding_width = None
        if not self._closing and reference <= c.close_threshold and normal > 0:
            self._closing = True
            self._reference_min = reference
        if self._closing:
            self._reference_min = min(self._reference_min, reference)
            if motor <= c.motor_limit_tolerance:
                self._holding_width = measured if self._holding_width is None else min(self._holding_width, measured)
            anchor = max(measured, self._holding_width or 0.0)
            release_level = min(anchor + c.release_margin, 1.0 - c.reference_hysteresis / 2)
            excursion = reference - self._reference_min
            needed = min(c.release_excursion, (1 - self._reference_min) / 2)
            release = valid and normal < 0 and reference >= release_level and excursion >= needed
            self._release_s = self._release_s + elapsed if release and self._release_previous else 0.0
            self._release_previous = release
            if self._release_s + 1e-12 >= c.release_confirmation_s:
                self._closing = False
                self._holding_width = None
                command, _ = self._position_command(reference, motor)
                state, reason = "position_tracking_release", "reference_rise_and_measured_clearance_confirmed"
            elif motor > c.motor_limit_tolerance:
                command, state, reason = 1.0, "closing", "continue_native_close_to_motor_limit"
            else:
                command, state, reason = 0.0, "closed_motor_hold", "retain_closed_motor_target_not_observed_aperture"
        else:
            release_level, excursion = None, 0.0
            self._release_s, self._release_previous = 0.0, False
            command, reason = self._position_command(reference, motor)
            state = "position_tracking" if command else "position_hold"
        return GripperPositionHoldResult(
            command,
            state,
            reason,
            reference,
            measured,
            motor,
            normal,
            self._closing,
            self._holding_width,
            self._reference_min,
            excursion,
            release_level,
            self._release_s,
            valid,
            reference - motor,
        )
