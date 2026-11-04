"""Cap-distance GN with a pose safety floor aligned to the reserve entry gate."""

import numpy as np

from s2a.research.trajectory_search.cap_distance_objective import CapDistanceObjective

POLICY = "cap_distance_gn_with_reserve_entry_floor"


class ReserveAlignedCapObjective(CapDistanceObjective):
    policy = POLICY

    @staticmethod
    def problem(anchor, Js, Jq, Je, previous, lower, upper, ceiling, radius, mode, config):
        problem = CapDistanceObjective.problem(
            anchor, Js, Jq, Je, previous, lower, upper, ceiling, radius, mode, config
        )
        if mode == "pose":
            weights = np.ones(4369)
            weights[-1] = config["terminal_time_scale"]
            required = config["safety_buffer"] + config["reserve_entry_fraction"] * config["reserve_target"] / weights
            floor = np.maximum(problem["safety_floor"], required)
            first, last = problem["row_blocks"]["safety"]
            problem["b"][first:last] = np.asarray(anchor["safety_slack"]) - floor
            problem["safety_floor"] = floor
        return problem
