"""Stage a loaded-gripper decision without committing unexecuted history."""

import copy
import math
from dataclasses import asdict
from numbers import Real

from s2a.research.gripper.causal_loaded_history import loaded_state
from s2a.research.joint_control.tracking.loaded_velocity_control import LoadedVelocityControl


class LoadedGripperTransaction:
    """Commit state only after the matching physical command completes."""

    def __init__(self, controller):
        if not isinstance(controller, LoadedVelocityControl):
            raise ValueError("A reconstructed loaded-gripper controller is required")
        self._controller = copy.deepcopy(controller)
        self._staged = None
        self._faulted = False

    def state(self):
        return {
            "latch": copy.deepcopy(loaded_state(self._controller)),
            "staged": self._staged is not None,
            "faulted": self._faulted,
        }

    def prepare(self, *args, **kwargs):
        if self._faulted:
            raise ValueError("Reconstruct history after an uncertain or partial execution")
        if self._staged is not None:
            raise ValueError("Commit or abort the previous gripper selection")
        candidate = copy.deepcopy(self._controller)
        result = candidate.update(*args, **kwargs)
        command = float(result.command)
        if not math.isfinite(command) or not -1 <= command <= 1:
            raise ValueError("Finite bounded gripper command required")
        decision = {"command": command, "detail": asdict(result)}
        self._staged = (decision, candidate, copy.deepcopy(decision))
        return decision

    def commit(self, decision, *, actual_command, completed_steps):
        if self._staged is None or decision is not self._staged[0]:
            raise ValueError("The current staged gripper selection is required")
        if type(completed_steps) is not int or completed_steps != 1:
            raise ValueError("A full completed physical period is required")
        if (
            isinstance(actual_command, bool)
            or not isinstance(actual_command, Real)
            or not math.isfinite(float(actual_command))
            or float(actual_command) != self._staged[2]["command"]
            or decision != self._staged[2]
        ):
            raise ValueError("Completed command or staged decision differs")
        self._controller = self._staged[1]
        self._staged = None

    def abort(self, decision, *, partial_execution=False):
        if self._staged is None or decision is not self._staged[0]:
            raise ValueError("The current staged gripper selection is required")
        if type(partial_execution) is not bool:
            raise ValueError("Partial execution must be explicit")
        self._staged = None
        if partial_execution:
            self._faulted = True
