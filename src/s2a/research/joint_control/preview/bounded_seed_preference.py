"""Opt-in cached-seed preference with a fixed future-quality tradeoff."""

import copy
from types import FunctionType

import numpy as np

from s2a.research.joint_control.preview import cached_seed_preference as unbounded
from s2a.research.joint_control.preview.shared_shift_fallback import SharedShiftFallbackAudit
from s2a.research.trajectory_search import margin_two_step_search as margin

MAX_FUTURE_SLACK_INCREASE = 0.3


def choose_bounded_seed(candidates, seed):
    """Keep the original ranking when the eligible seed costs too much future slack."""
    chosen, info = unbounded.choose_cached_seed(candidates, seed)
    requested = info["future_slack_increase"] if info["seed_in_eligible_cache"] else None
    blocked = bool(requested is not None and requested > MAX_FUTURE_SLACK_INCREASE + margin.SELECTION_TOLERANCE)
    if blocked:
        chosen, _ = margin.select_evaluated_margin_candidate(candidates, prefer_refresh=False)
        info.update(
            selected_seed=bool(np.array_equal(chosen[3], seed)),
            selected_changed=False,
            selected_future_slack=float(chosen[0]),
            future_slack_increase=0.0,
        )
    info.update(
        max_future_slack_increase=MAX_FUTURE_SLACK_INCREASE,
        unbounded_seed_future_slack_increase=requested,
        seed_blocked_by_future_slack=blocked,
    )
    return chosen, info


def solve_bounded_seed(*args, **kwargs):
    """Run the original bounded search with an isolated, bounded cache selector."""
    solver = unbounded.solve_prefer_cached_seed
    namespace = solver.__globals__.copy()
    namespace["choose_cached_seed"] = choose_bounded_seed
    isolated = FunctionType(solver.__code__, namespace, solver.__name__, solver.__defaults__, solver.__closure__)
    isolated.__kwdefaults__ = dict(solver.__kwdefaults__ or {})
    result = isolated(*args, **kwargs)
    trace = result["seed_preference"]
    # Fallbacks without an eligible margin cache never invoke the selector.
    trace.setdefault("max_future_slack_increase", MAX_FUTURE_SLACK_INCREASE)
    trace.setdefault("unbounded_seed_future_slack_increase", None)
    trace.setdefault("seed_blocked_by_future_slack", False)
    return result


class BoundedSeedPreferredAudit(SharedShiftFallbackAudit):
    """Retain the shared search and independent witness audit for this variant."""

    def __init__(self, receipt):
        if not isinstance(receipt, dict) or "seed_preference_traces" in receipt:
            raise ValueError("A fresh branch receipt is required")
        self.seed_traces = receipt["seed_preference_traces"] = []

        def solver(*args, **kwargs):
            result = solve_bounded_seed(*args, **kwargs)
            self.seed_traces.append(copy.deepcopy(result["seed_preference"]))
            return result

        super().__init__(receipt, solver=solver)
