"""Opt-in preference for an already evaluated, margin-qualified refiner seed."""

import copy
from types import FunctionType

import numpy as np

from s2a.geometry import finite
from s2a.research.joint_control.preview.shared_shift_fallback import SharedShiftFallbackAudit
from s2a.research.trajectory_search import margin_guarded_fallback as shared
from s2a.research.trajectory_search import margin_two_step_search as margin


def choose_cached_seed(candidates, seed):
    """Change only ranking within the already joint-safe, margin-qualified cache."""
    seed = finite(seed, (8, 6))
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("A nonempty previously accepted margin candidate cache is required")
    for row in candidates:
        if len(row) != 7 or not np.isfinite([row[0], row[1]]).all() or row[0] < 0:
            raise ValueError("Each cached candidate must retain its finite future slack and objective")
        finite(row[3], (8, 6))
        if not margin.margin_prefix_satisfied(row[5]):
            raise ValueError("Every selectable candidate must satisfy the original two-point margins")
    default, _ = margin.select_evaluated_margin_candidate(candidates, prefer_refresh=False)
    matches = [row for row in candidates if np.array_equal(row[3], seed)]
    if len(matches) > 1:
        raise ValueError("The same cached seed cannot occur twice")
    selected = matches[0] if matches else default
    info = dict(
        seed_in_eligible_cache=bool(matches),
        selected_seed=bool(matches),
        selected_changed=not np.array_equal(selected[3], default[3]),
        eligible_candidates=len(candidates),
        default_future_slack=float(default[0]),
        selected_future_slack=float(selected[0]),
        future_slack_increase=float(selected[0] - default[0]),
        extra_model_responses=0,
    )
    return selected, info


def solve_prefer_cached_seed(
    model, velocity, seed, previous, interface, site, tool, targets, twists, lower, upper, **kwargs
):
    """Run the unchanged shared search and replace only its cached margin ranking."""
    seed = finite(seed, (8, 6)).copy()
    if kwargs.get("prefer_refresh", False) is not False:
        raise ValueError("Seed and refresh preferences cannot be combined")
    recorded = {}

    def isolated_margin(capture):
        solver = shared.isolated_margin_solver(capture)
        namespace = solver.__globals__.copy()

        def select(candidates, *, prefer_refresh=False):
            if prefer_refresh is not False or recorded:
                raise ValueError("Exactly one uncombined cached selection is allowed")
            selected, info = choose_cached_seed(candidates, seed)
            recorded.update(info)
            return selected, None

        namespace["select_evaluated_margin_candidate"] = select
        function = FunctionType(solver.__code__, namespace, solver.__name__, solver.__defaults__, solver.__closure__)
        function.__kwdefaults__ = dict(solver.__kwdefaults__ or {})
        return function

    solver = shared.solve_margin_guarded_fallback
    namespace = solver.__globals__.copy()
    namespace["isolated_margin_solver"] = isolated_margin
    function = FunctionType(solver.__code__, namespace, solver.__name__, solver.__defaults__, solver.__closure__)
    function.__kwdefaults__ = dict(solver.__kwdefaults__ or {})
    result = function(model, velocity, seed, previous, interface, site, tool, targets, twists, lower, upper, **kwargs)
    if not recorded:
        recorded = dict(
            seed_in_eligible_cache=False,
            selected_seed=False,
            selected_changed=False,
            eligible_candidates=0,
            default_future_slack=None,
            selected_future_slack=None,
            future_slack_increase=None,
            extra_model_responses=0,
        )
    result["seed_preference"] = recorded
    return result


class CachedSeedPreferredAudit(SharedShiftFallbackAudit):
    """Keep the original shared fallback and independent witness audit."""

    def __init__(self, receipt):
        if not isinstance(receipt, dict) or "seed_preference_traces" in receipt:
            raise ValueError("A fresh branch receipt is required")
        self.seed_traces = receipt["seed_preference_traces"] = []

        def solver(*args, **kwargs):
            result = solve_prefer_cached_seed(*args, **kwargs)
            self.seed_traces.append(copy.deepcopy(result["seed_preference"]))
            return result

        super().__init__(receipt, solver=solver)
