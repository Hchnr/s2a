"""Opt-in closest cached plan within the registered future-quality bound."""

import copy
from types import FunctionType

import numpy as np

from s2a.geometry import finite
from s2a.research.joint_control.preview import bounded_seed_preference as bounded
from s2a.research.joint_control.preview import cached_seed_preference as unbounded
from s2a.research.joint_control.preview.shared_shift_fallback import SharedShiftFallbackAudit
from s2a.research.trajectory_search import margin_two_step_search as margin


def choose_nearest_bounded_seed(candidates, seed):
    """Select an already evaluated plan closest to the current complete seed."""
    seed = finite(seed, (8, 6)).copy()
    bounded_choice, info = bounded.choose_bounded_seed(candidates, seed)
    default, _ = margin.select_evaluated_margin_candidate(candidates, prefer_refresh=False)
    allowed = [
        row
        for row in candidates
        if row[0] <= default[0] + bounded.MAX_FUTURE_SLACK_INCREASE + margin.SELECTION_TOLERANCE
    ]
    if not allowed:
        raise ValueError("The original candidate must remain inside its own future-quality bound")

    def distance(row):
        delta = finite(row[3], (8, 6)) - seed
        return float(np.sum(delta * delta))

    selected = min(allowed, key=lambda row: (distance(row), row[0], row[1], row[2]))
    selected_distance, bounded_distance, default_distance = map(distance, (selected, bounded_choice, default))
    if selected_distance > bounded_distance + 1e-12:
        raise ValueError("The selected cached plan must be no further from the seed than the bounded baseline")
    info.update(
        selected_seed=bool(np.array_equal(selected[3], seed)),
        selected_changed=not np.array_equal(selected[3], default[3]),
        selected_future_slack=float(selected[0]),
        future_slack_increase=float(selected[0] - default[0]),
        bounded_candidates=len(allowed),
        selected_seed_distance_squared=selected_distance,
        bounded_seed_distance_squared=bounded_distance,
        default_seed_distance_squared=default_distance,
        changed_from_bounded=not np.array_equal(selected[3], bounded_choice[3]),
        distance_metric="squared_euclidean_8x6_normalized_arm_commands",
    )
    return selected, info


def solve_nearest_bounded_seed(*args, **kwargs):
    """Reuse the shared search and replace only its isolated cache selector."""
    solver = unbounded.solve_prefer_cached_seed
    namespace = solver.__globals__.copy()
    namespace["choose_cached_seed"] = choose_nearest_bounded_seed
    isolated = FunctionType(solver.__code__, namespace, solver.__name__, solver.__defaults__, solver.__closure__)
    isolated.__kwdefaults__ = dict(solver.__kwdefaults__ or {})
    result = isolated(*args, **kwargs)
    trace = result["seed_preference"]
    # A search without a margin-qualified cache keeps the original fallback.
    trace.setdefault("max_future_slack_increase", bounded.MAX_FUTURE_SLACK_INCREASE)
    trace.setdefault("unbounded_seed_future_slack_increase", None)
    trace.setdefault("seed_blocked_by_future_slack", False)
    trace.setdefault("bounded_candidates", 0)
    trace.setdefault("selected_seed_distance_squared", None)
    trace.setdefault("bounded_seed_distance_squared", None)
    trace.setdefault("default_seed_distance_squared", None)
    trace.setdefault("changed_from_bounded", False)
    trace.setdefault("distance_metric", "squared_euclidean_8x6_normalized_arm_commands")
    return result


class NearestBoundedSeedAudit(SharedShiftFallbackAudit):
    """Keep the original search, independent witness audit and fallback."""

    def __init__(self, receipt):
        if not isinstance(receipt, dict) or "seed_preference_traces" in receipt:
            raise ValueError("A fresh branch receipt is required")
        self.seed_traces = receipt["seed_preference_traces"] = []

        def solver(*args, **kwargs):
            result = solve_nearest_bounded_seed(*args, **kwargs)
            self.seed_traces.append(copy.deepcopy(result["seed_preference"]))
            return result

        super().__init__(receipt, solver=solver)
