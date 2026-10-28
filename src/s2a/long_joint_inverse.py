"""Long action-space optimization with fixed frozen joint dynamics."""

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize

from s2a.geometry import finite
from s2a.joint_limit_preview_objective import preview_objective
from s2a.long_joint_preview import rollout_joint_preview


def refine_long_plan(model, velocity, seed, previous, interface, site, tool, targets, twists, lower, upper):
    """H<=40 model-only diagnostic; allows anticipation to trade endpoint errors."""
    seed, previous = finite(seed), finite(previous, (6,))
    if seed.ndim != 2 or seed.shape[1] != 6 or not 1 <= len(seed) <= 40:
        raise ValueError("One to forty seed commands required")
    horizon, width = len(seed), seed.size
    lo, hi = np.tile(interface.input_min, horizon), np.tile(interface.input_max, horizon)
    difference = np.eye(width) - np.eye(width, k=-6)
    center = np.r_[previous, np.zeros(width - 6)]

    def feasible(x):
        return bool(
            np.all(x >= lo - 1e-12) and np.all(x <= hi + 1e-12) and np.max(abs(difference @ x - center)) <= 0.25 + 1e-12
        )

    if not feasible(seed.ravel()) or np.any(previous < interface.input_min) or np.any(previous > interface.input_max):
        raise ValueError("Feasible seed and previous required")
    scale = np.tile(
        (interface.output_max - interface.output_min) / (interface.input_max - interface.input_min), horizon
    )
    cache = {}

    class Stop(RuntimeError):
        pass

    def evaluate(x):
        key = tuple(x)
        if key not in cache:
            if len(cache) >= 160:
                raise Stop("response_budget")
            deltas = np.asarray([interface.decode(c) for c in np.asarray(x).reshape(horizon, 6)])
            if np.any(np.all(deltas[:, 3:] == 0, axis=1)):
                raise Stop("unsupported_zero_rotation_latch")
            rollout = rollout_joint_preview(model, velocity, deltas, site, lower, upper, enforce_limits=True)
            cache[key] = dict(rollout=rollout, score=preview_objective(rollout, tool, targets, twists))
        return cache[key]

    original = evaluate(seed.ravel())
    chosen, commands = original, seed.copy()
    status, success, iterations = "weak_seed", False, 0

    def objective(x):
        item = evaluate(x)
        if np.any(item["rollout"]["weak_projection_steps"]):
            raise Stop("weak_trial")
        return item["score"]["objective"], item["score"]["gradient"] * scale

    if not np.any(original["rollout"]["weak_projection_steps"]):
        try:
            result = minimize(
                objective,
                seed.ravel(),
                jac=True,
                method="SLSQP",
                bounds=Bounds(lo, hi),
                constraints=[LinearConstraint(difference, center - 0.25, center + 0.25)],
                options=dict(maxiter=60, ftol=1e-9),
            )
            status, success, iterations = str(result.message), bool(result.success), int(result.nit)
        except Stop as error:
            status = str(error)
        for key, item in cache.items():
            if (
                feasible(np.asarray(key))
                and not np.any(item["rollout"]["weak_projection_steps"])
                and item["score"]["objective"] < chosen["score"]["objective"]
            ):
                commands, chosen = np.asarray(key).reshape(horizon, 6).copy(), item
    return dict(
        planned_commands=commands,
        rollout=chosen["rollout"],
        seed_rollout=original["rollout"],
        objective=chosen["score"]["objective"],
        seed_objective=original["score"]["objective"],
        response_evaluations=len(cache),
        optimizer_success=success,
        optimizer_message=status,
        iterations=iterations,
        accepted=not np.array_equal(seed, commands),
        scope="long frozen M/J ideal-limit action-space diagnostic; no control qualification",
    )
