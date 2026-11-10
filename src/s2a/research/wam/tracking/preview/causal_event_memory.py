"""Experimental causal memory of an already generated gripper phase."""

from copy import deepcopy
from dataclasses import dataclass

import numpy as np

from s2a.geometry import finite, pose_error, validate_transform


@dataclass(frozen=True)
class EventMemoryConfig:
    horizon: int
    stride: int
    closing_below: float
    opening_above: float
    reference_position_m: float
    reference_rotation_rad: float
    current_position_m: float
    current_rotation_rad: float

    def __post_init__(self):
        if (
            type(self.horizon) is not int
            or type(self.stride) is not int
            or self.stride < 1
            or self.horizon < 2 * self.stride
            or self.horizon % self.stride
        ):
            raise ValueError("A bounded horizon with at least two complete strides is required")
        values = np.asarray(
            [
                self.closing_below,
                self.opening_above,
                self.reference_position_m,
                self.reference_rotation_rad,
                self.current_position_m,
                self.current_rotation_rad,
            ],
            dtype=float,
        )
        if (
            not np.isfinite(values).all()
            or not 0 < self.closing_below < self.opening_above < 1
            or np.any(values[2:] <= 0)
        ):
            raise ValueError("Explicit normalized phase thresholds and positive finite pose scales required")


class CausalEventMemory:
    """Stage a generated event and commit memory only after a complete stride.

    This experimental reference selector supplies no joint-safety guarantee.
    The caller must separately qualify and execute each selected pose/opening
    pair and score actual errors against both the selected and latest reference.
    """

    def __init__(self, config):
        if not isinstance(config, EventMemoryConfig):
            raise ValueError("An explicit event-memory configuration is required")
        self.config = config
        self.reset()

    def reset(self):
        self._pending = None
        self._staged = None
        self._last_step = None
        self._faulted = False

    def state(self):
        pending = self._pending
        return {
            "last_completed_replan_step": self._last_step,
            "faulted": self._faulted,
            "staged": self._staged is not None,
            "pending_event": None
            if pending is None
            else {
                "direction": pending["direction"],
                "source_step": pending["source_step"],
                "event_index": pending["event_index"],
                "absolute_endpoint": pending["source_step"] + pending["event_index"] + 1,
            },
        }

    def _phase(self, opening):
        if opening > self.config.opening_above:
            return "closure"
        if opening < self.config.closing_below:
            return "opening"
        return None

    def _first(self, openings, direction):
        indices = np.flatnonzero(
            openings < self.config.closing_below if direction == "closure" else openings > self.config.opening_above
        )
        return int(indices[0]) if len(indices) else None

    def _arm(self, step, poses, openings, phase):
        if phase is None:
            return None
        event_index = self._first(openings, phase)
        if event_index is None or event_index < self.config.stride:
            return None
        return {
            "source_step": step,
            "event_index": event_index,
            "direction": phase,
            "poses": poses.copy(),
            "openings": openings.copy(),
        }

    def prepare(self, poses, openings, *, step, measured_tcp, measured_opening):
        if self._faulted:
            raise ValueError("A partially executed stride requires an explicit reset")
        if self._staged is not None:
            raise ValueError("The previous selection must be committed or aborted")
        if (
            type(step) is not int
            or step < 0
            or (self._last_step is not None and step != self._last_step + self.config.stride)
        ):
            raise ValueError("Consecutive completed replan steps are required")
        poses, current = validate_transform(poses), validate_transform(measured_tcp)
        openings = finite(openings, (self.config.horizon,))
        opening = float(finite(measured_opening, ()))
        if poses.shape != (self.config.horizon, 4, 4) or current.shape != (4, 4):
            raise ValueError("One generated trajectory and current TCP pose are required")
        if np.any((openings < 0) | (openings > 1)) or not 0 <= opening <= 1:
            raise ValueError("Generated and measured openings must be normalized")
        phase = self._phase(opening)
        chosen, chosen_o = poses[: self.config.stride].copy(), openings[: self.config.stride].copy()
        source_step, offset, source, reason = step, 0, "latest", "no_pending_event"
        retained, activated, guards = None, False, None
        prior = self._pending
        if prior is not None:
            offset = step - prior["source_step"]
            remaining = prior["event_index"] - offset
            if phase != prior["direction"]:
                reason = "measured_phase_changed"
            elif offset <= 0 or offset + self.config.stride > self.config.horizon or remaining < 0:
                reason = "event_expired"
            else:
                latest_first = self._first(openings, phase)
                if latest_first is not None and latest_first <= remaining:
                    reason = "latest_covers_event"
                else:
                    old_prefix = prior["poses"][offset : offset + self.config.stride]
                    difference = pose_error(old_prefix, poses[: self.config.stride])
                    event_difference = pose_error(prior["poses"][prior["event_index"]], poses[remaining])
                    current_difference = pose_error(old_prefix[0], current)
                    guards = {
                        "prefix_position_m": float(np.max(np.linalg.norm(difference[:, :3], axis=1))),
                        "prefix_rotation_rad": float(np.max(np.linalg.norm(difference[:, 3:], axis=1))),
                        "event_position_m": float(np.linalg.norm(event_difference[:3])),
                        "event_rotation_rad": float(np.linalg.norm(event_difference[3:])),
                        "current_position_m": float(np.linalg.norm(current_difference[:3])),
                        "current_rotation_rad": float(np.linalg.norm(current_difference[3:])),
                    }
                    agreement = (
                        guards["prefix_position_m"] <= self.config.reference_position_m
                        and guards["prefix_rotation_rad"] <= self.config.reference_rotation_rad
                        and guards["event_position_m"] <= self.config.reference_position_m
                        and guards["event_rotation_rad"] <= self.config.reference_rotation_rad
                        and guards["current_position_m"] <= self.config.current_position_m
                        and guards["current_rotation_rad"] <= self.config.current_rotation_rad
                    )
                    if not agreement:
                        reason = "pose_agreement_rejected"
                    elif remaining >= self.config.stride:
                        reason, retained = "event_retained", prior
                    else:
                        reason, source, source_step, activated = "event_due", "remembered", prior["source_step"], True
                        chosen = old_prefix.copy()
                        chosen_o = prior["openings"][offset : offset + self.config.stride].copy()
        armed = False
        if retained is None and not activated:
            retained = self._arm(step, poses, openings, phase)
            armed = retained is not None
        event = prior if prior is not None else retained
        selected_offset = offset if activated else 0
        decision = {
            "step": step,
            "source": source,
            "source_step": source_step,
            "source_indices": list(range(selected_offset, selected_offset + self.config.stride)),
            "poses": chosen,
            "openings": chosen_o,
            "reason": reason,
            "armed_new_event": armed,
            "event": None
            if event is None
            else {
                "direction": event["direction"],
                "source_step": event["source_step"],
                "event_index": event["event_index"],
                "absolute_endpoint": event["source_step"] + event["event_index"] + 1,
            },
            "guards": guards,
        }
        metadata = {key: deepcopy(value) for key, value in decision.items() if key not in ("poses", "openings")}
        self._staged = (decision, retained, chosen.copy(), chosen_o.copy(), metadata)
        return decision

    def commit(self, decision, *, completed_steps):
        if self._staged is None or decision is not self._staged[0]:
            raise ValueError("The current staged selection is required")
        if type(completed_steps) is not int or completed_steps != self.config.stride:
            raise ValueError("Only an actually completed full stride can commit memory")
        _, pending, poses, openings, metadata = self._staged
        if (
            set(decision) != set(metadata) | {"poses", "openings"}
            or {key: value for key, value in decision.items() if key not in ("poses", "openings")} != metadata
            or not np.array_equal(decision["poses"], poses)
            or not np.array_equal(decision["openings"], openings)
        ):
            raise ValueError("The staged pose/opening selection was changed")
        self._pending, self._last_step, self._staged = pending, decision["step"], None

    def abort(self, decision, *, completed_steps=0):
        if self._staged is None or decision is not self._staged[0]:
            raise ValueError("The current staged selection is required")
        if type(completed_steps) is not int or not 0 <= completed_steps < self.config.stride:
            raise ValueError("Abort requires a bounded incomplete stride")
        self._staged = None
        if completed_steps:
            self._pending, self._faulted = None, True
