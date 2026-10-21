"""Native close-to-limit intent, zero motor hold, measured-clearance release.

This separate candidate preserves the historical closing onset. Observed
aperture is never used as a free motor closing goal. No object/contact, task,
arm pose, future observation or force estimate is accepted.
"""

import math
from dataclasses import dataclass
from numbers import Real

from s2a.gripper import OpeningIntent


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite scalar")
    return float(value)


def _opening(value, name):
    value = _finite(value, name)
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be in [0, 1]")
    return value


@dataclass(frozen=True)
class GripperIntentHoldConfig:
    reference_hysteresis: float = 0.02
    release_margin: float = 0.04
    release_excursion: float = 0.04
    release_confirmation_s: float = 0.05
    motor_limit_tolerance: float = 0.005
    max_evidence_interval_s: float = 0.1

    def __post_init__(self):
        for name, value in vars(self).items():
            if _finite(value, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("reference_hysteresis", "release_margin", "release_excursion", "motor_limit_tolerance"):
            if getattr(self, name) >= 1:
                raise ValueError(f"{name} must be below one normalized aperture")
        if self.release_margin < self.reference_hysteresis:
            raise ValueError("release_margin must cover reference_hysteresis")


@dataclass(frozen=True)
class GripperIntentHoldResult:
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


class GripperIntentHold:
    """Native Panda candidate; retain an instance across generated chunks.

    ``previous_command`` and ``dt`` describe the actual preceding physical
    interval. Zero holds an existing motor goal; it does not command zero force.
    A motor limit is not proof of object contact, sufficient force or grasp.
    """

    def __init__(self, measured_opening, *, config=None):
        self.config = config if config is not None else GripperIntentHoldConfig()
        if not isinstance(self.config, GripperIntentHoldConfig):
            raise ValueError("config must be GripperIntentHoldConfig")
        self.reset(measured_opening)

    def reset(self, measured_opening):
        measured = _opening(measured_opening, "measured_opening")
        self.intent = OpeningIntent(measured, band=self.config.reference_hysteresis)
        self._closing = False
        self._reference_min = measured
        self._holding_width = None
        self._release_s = 0.0
        self._release_previous = False

    def update(self, desired_opening, measured_opening, *, motor_opening, previous_command, dt):
        reference = _opening(desired_opening, "desired_opening")
        measured = _opening(measured_opening, "measured_opening")
        motor = _opening(motor_opening, "motor_opening")
        previous = _finite(previous_command, "previous_command")
        elapsed = _finite(dt, "dt")
        if not -1 <= previous <= 1 or elapsed <= 0:
            raise ValueError("previous_command must be in [-1, 1] and dt positive")
        c = self.config
        normal = self.intent.update(reference)
        valid = elapsed <= c.max_evidence_interval_s + 1e-12
        if not valid or previous < 0:
            self._release_s = 0.0
            self._release_previous = False
            self._holding_width = None
        if normal > 0 and not self._closing:
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
                command, state, reason = -1.0, "opening", "reference_rise_and_measured_clearance_confirmed"
            elif motor > c.motor_limit_tolerance:
                command, state, reason = 1.0, "closing", "continue_native_close_to_motor_limit"
            else:
                command, state, reason = 0.0, "closed_motor_hold", "retain_closed_motor_target_not_observed_aperture"
        else:
            release_level, excursion = None, 0.0
            self._release_s = 0.0
            self._release_previous = False
            if motor < 1 - c.motor_limit_tolerance:
                command, state, reason = -1.0, "opening", "continue_native_open_to_motor_limit"
            else:
                command, state, reason = 0.0, "open_motor_hold", "retain_open_motor_target"
        return GripperIntentHoldResult(
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
        )
