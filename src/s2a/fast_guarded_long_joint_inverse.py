"""Long action preview with first-executed TCP pose protection."""

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize

from s2a.fast_long_joint_preview import rollout_joint_preview
from s2a.geometry import finite
from s2a.joint_limit_preview_objective import preview_objective


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
        divisor = np.array([100.0] * 3 + [20.0] * 3)
        first_seed = original["score"]["residual"][:6] / divisor
        seed_norms = np.array([np.linalg.norm(first_seed[:3]), np.linalg.norm(first_seed[3:])])

        def first_errors(x):
            score = evaluate(x)["score"]
            return score["residual"][:6] / divisor, score["jacobian"][:6] * scale[None, :] / divisor[:, None]

        constraints = [LinearConstraint(difference, center - 0.25, center + 0.25)]
        for group, (a, b) in enumerate(((0, 3), (3, 6))):
            if seed_norms[group] <= 1e-12:
                constraints.append(
                    dict(
                        type="eq",
                        fun=lambda x, a=a, b=b: first_errors(x)[0][a:b] - first_seed[a:b],
                        jac=lambda x, a=a, b=b: first_errors(x)[1][a:b],
                    )
                )
            else:

                def value(x, a=a, b=b, group=group):
                    return 1 - np.linalg.norm(first_errors(x)[0][a:b]) / seed_norms[group]

                def derivative(x, a=a, b=b, group=group):
                    error, jac = first_errors(x)
                    norm = np.linalg.norm(error[a:b])
                    return -(error[a:b] / norm if norm > 1e-15 else np.zeros(3)) @ jac[a:b] / seed_norms[group]

                constraints.append(dict(type="ineq", fun=value, jac=derivative))

        def guarded(x):
            error, _ = first_errors(x)
            return bool(
                np.linalg.norm(error[:3]) <= seed_norms[0] + 1e-10
                and np.linalg.norm(error[3:]) <= seed_norms[1] + 1e-10
            )

        try:
            result = minimize(
                objective,
                seed.ravel(),
                jac=True,
                method="SLSQP",
                bounds=Bounds(lo, hi),
                constraints=constraints,
                options=dict(maxiter=60, ftol=1e-9),
            )
            status, success, iterations = str(result.message), bool(result.success), int(result.nit)
        except Stop as error:
            status = str(error)
        for key, item in cache.items():
            if (
                feasible(np.asarray(key))
                and not np.any(item["rollout"]["weak_projection_steps"])
                and guarded(np.asarray(key))
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
        scope="long frozen M/J ideal-limit action-space diagnostic; first TCP guard; no control qualification",
    )
