"""Isolate a scaled first-stage objective without changing the frozen solver."""

from types import FunctionType

import numpy as np

from s2a.geometry import finite
from s2a.research.trajectory_search import two_step_viability as original

SEARCH_GUARD = 1e-4
OBJECTIVE_SCALE = 1e4
_ORIGINAL_SOLVER = original.solve_two_step_viability
_FROZEN_GLOBALS = {
    name: _ORIGINAL_SOLVER.__globals__[name]
    for name in (
        "prefix_violation_objective",
        "split_prefix_constraints",
        "rollout_joint_preview",
        "preview_objective",
        "pose_quality_constraints",
        "linear_domain",
        "nearest_linear_plan",
        "domain_check",
        "minimize",
    )
}


def guarded_prefix_objective(values, jacobian):
    """Scale a tiny interior search target without changing acceptance caps."""
    values, jacobian = finite(values), finite(jacobian)
    if values.shape != (16,) or jacobian.ndim != 2 or jacobian.shape[0] != 16:
        raise ValueError("Eight position/rotation rows and matching Jacobian required")
    violation = np.minimum(values[:4] - SEARCH_GUARD, 0.0)
    return 0.5 * OBJECTIVE_SCALE * float(violation @ violation), OBJECTIVE_SCALE * (violation @ jacobian[:4])


def isolated_solver():
    """Copy the original solver's namespace and replace only its first objective."""
    if original.solve_two_step_viability is not _ORIGINAL_SOLVER or any(
        _ORIGINAL_SOLVER.__globals__.get(name) is not value for name, value in _FROZEN_GLOBALS.items()
    ):
        raise ValueError("The original solver and its frozen dependencies must remain unchanged")
    namespace = _ORIGINAL_SOLVER.__globals__.copy()
    namespace["prefix_violation_objective"] = guarded_prefix_objective
    candidate = FunctionType(
        _ORIGINAL_SOLVER.__code__,
        namespace,
        _ORIGINAL_SOLVER.__name__,
        _ORIGINAL_SOLVER.__defaults__,
        _ORIGINAL_SOLVER.__closure__,
    )
    candidate.__kwdefaults__ = dict(_ORIGINAL_SOLVER.__kwdefaults__ or {})
    return candidate


def solve_guarded_two_step(*args, **kwargs):
    """Run the original safety and selection contract with one changed objective."""
    result = isolated_solver()(*args, **kwargs)
    result["first_stage_search_objective"] = dict(
        normalized_guard=SEARCH_GUARD,
        scale=OBJECTIVE_SCALE,
        acceptance_caps_changed=False,
        future_stage_changed=False,
    )
    return result
