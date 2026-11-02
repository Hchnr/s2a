"""Independent reconstruction of the stricter pose reserve constraint."""

import numpy as np

from scripts.tracking.trajectory_search.cap_distance_audit_policy import CapDistanceAuditPolicy
from scripts.verification.trajectory_search.directional_derivative_audit import array

POLICY = "cap_distance_gn_with_reserve_entry_floor"


class ReserveAlignedCapAuditPolicy(CapDistanceAuditPolicy):
    policy = POLICY

    @staticmethod
    def problem(commands, safety, squared, error, Js, Jq, Je, previous, lower, upper, ceiling, radius, mode, config):
        B, g, A, b = CapDistanceAuditPolicy.problem(
            commands, safety, squared, error, Js, Jq, Je, previous, lower, upper, ceiling, radius, mode, config
        )
        if mode == "pose":
            s = array(safety, "All original safety rows", (4369,))
            weights = np.r_[np.ones(4368), config["terminal_time_scale"]]
            reserve = float(np.min(weights * (s - config["safety_buffer"]) / config["reserve_target"]))
            level = max(min(0.5 * reserve, 0.5), config["reserve_entry_fraction"])
            b[:4369] = s - (config["safety_buffer"] + level * config["reserve_target"] / weights)
        return B, g, A, b
