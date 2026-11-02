"""Select an already charged safe reserve rejection for safety-only calibration."""

import numpy as np

from s2a.geometry import finite
from s2a.research.trajectory_search.interior_trajectory_search import _outputs, metrics

POLICY = "safe_reserve_rejection_minimum_alpha_safety_only"


class ReserveSafetyFeedback:
    policy = POLICY

    @staticmethod
    def choose(stage, entries, ceiling, config):
        if stage["accepted_response_index"] is not None or stage["mode"] != "pose":
            return None
        eligible = []
        for attempt in stage["attempts"]:
            index = attempt["response_index"]
            if (
                index is None
                or attempt["accepted"]
                or attempt["acceptance_reason"] != "pose_reserve_below_entry"
                or not (attempt["predicted_metrics"]["reserve"] >= config["reserve_entry_fraction"])
            ):
                continue
            observed = entries[index]
            safety, squared = _outputs(observed)
            step = finite(attempt["actual_step"], (12, 6))
            if (
                observed["audit"]["feasible"]
                and np.all(safety >= config["safety_buffer"])
                and np.all(squared <= ceiling)
                and np.any(step != 0)
                and np.sum(step**2) >= config["secant_min_squared_step"]
                and metrics(observed, config)["reserve"] < config["reserve_entry_fraction"]
            ):
                eligible.append(attempt)
        return min(eligible, key=lambda t: (t["alpha"], t["response_index"])) if eligible else None
