"""Explicit stricter pose acceptance and inherited original search geometry."""

from dataclasses import dataclass

import numpy as np

from s2a.geometry import finite
from s2a.research.trajectory_search.state_vector_dual_recovery import initial_global_envelope

RESERVE_POLICY = "pose_reserve_at_least_original_entry_fraction"
DOMAIN_POLICY = "inherited_original_root_command_and_pose_envelope"


class ReserveEntryAcceptance:
    policy = RESERVE_POLICY

    @staticmethod
    def allows(actual, config, mode):
        return mode != "pose" or actual["reserve"] >= config["reserve_entry_fraction"]


@dataclass(frozen=True)
class InheritedSearchDomain:
    reference_command: np.ndarray
    reference_squared: np.ndarray

    @classmethod
    def from_entry(cls, entry):
        return cls(
            finite(entry["commands"], (12, 6)).copy(),
            finite(entry["normalized_squared_errors"], (12, 2)).ravel().copy(),
        )

    def bounds(self, previous, input_min, input_max, config):
        root = finite(self.reference_command, (12, 6))
        lower = np.maximum(np.tile(input_min, (12, 1)), root - config["global_displacement_limit"])
        upper = np.minimum(np.tile(input_max, (12, 1)), root + config["global_displacement_limit"])
        lower[0] = np.maximum(lower[0], previous - 0.25)
        upper[0] = np.minimum(upper[0], previous + 0.25)
        return lower, upper

    def ceiling(self, config):
        return initial_global_envelope(finite(self.reference_squared, (24,)), config)

    def metadata(self):
        return dict(
            policy=DOMAIN_POLICY,
            reference_command=finite(self.reference_command, (12, 6)).copy(),
            reference_squared=finite(self.reference_squared, (24,)).copy(),
        )
