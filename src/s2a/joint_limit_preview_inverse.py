"""Bounded four-period joint preview inverse with analytic local derivatives."""

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize

from s2a.geometry import finite
from s2a.joint_limit_preview import rollout_joint_preview
from s2a.joint_limit_preview_objective import preview_objective


class _StopSearch(RuntimeError):
    pass


def refine_joint_plan(
    model,
    velocity,
    seed,
    previous,
    interface,
    site,
    site_from_tcp,
    desired_tcp,
    desired_twists,
    joint_lower,
    joint_upper,
    *,
    pose_guard=True,
):
    """Optimize all available H<=4 commands; execute none in this pure function.

    A weak projection face, unsupported zero-rotation trial or evaluation budget
    exhaustion returns the original feasible seed without retry. No old model's
    innovation is implicitly injected into the frozen joint dynamics.
    """
    seed = finite(seed)
    previous = finite(previous, (6,))
    if seed.ndim != 2 or seed.shape[1] != 6 or not 1 <= len(seed) <= 4 or type(pose_guard) is not bool:
        raise ValueError("One to four seed commands and explicit pose guard required")
    horizon, width = len(seed), seed.size
    lo, hi = np.tile(interface.input_min, horizon), np.tile(interface.input_max, horizon)
    difference = np.eye(width) - np.eye(width, k=-6)
    center = np.r_[previous, np.zeros(width - 6)]
    rate_lo, rate_hi = center - 0.25, center + 0.25

    def feasible(x):
        return bool(
            np.all(x >= lo - 1e-12)
            and np.all(x <= hi + 1e-12)
            and np.all(difference @ x >= rate_lo - 1e-12)
            and np.all(difference @ x <= rate_hi + 1e-12)
        )

    if np.any(previous < interface.input_min) or np.any(previous > interface.input_max) or not feasible(seed.ravel()):
        raise ValueError("Feasible previous and complete box/rate seed plan required")
    scale = np.tile(
        (interface.output_max - interface.output_min) / (interface.input_max - interface.input_min), horizon
    )
    cache = {}

    def evaluate(x):
        key = tuple(np.asarray(x))
        if key not in cache:
            if len(cache) >= 80:
                raise _StopSearch("response_budget_seed")
            commands = np.asarray(x).reshape(horizon, 6)
            deltas = np.asarray([interface.decode(c) for c in commands])
            if np.any(np.all(deltas[:, 3:] == 0, axis=1)):
                raise _StopSearch("unsupported_latch_trial_seed")
            rollout = rollout_joint_preview(
                model, velocity, deltas, site, joint_lower, joint_upper, enforce_limits=True
            )
            score = preview_objective(rollout, site_from_tcp, desired_tcp, desired_twists)
            score["jacobian"] *= scale[None, :]
            score["gradient"] *= scale
            errors = score["residual"].reshape(horizon, 12)[:, :6] / np.array([100.0] * 3 + [20.0] * 3)
            error_jac = (
                score["jacobian"].reshape(horizon, 12, width)[:, :6] / np.array([100.0] * 3 + [20.0] * 3)[None, :, None]
            )
            cache[key] = dict(score=score, rollout=rollout, errors=errors, error_jac=error_jac)
        return cache[key]

    try:
        original = evaluate(seed.ravel())
    except _StopSearch as error:
        raise ValueError("Initial seed needs explicit nonzero reset representatives") from error
    limits = np.c_[np.sum(original["errors"][:, :3] ** 2, axis=1), np.sum(original["errors"][:, 3:] ** 2, axis=1)]
    constraint_scales = np.array([100.0**2, 20.0**2])
    nonzero_groups, zero_groups = [], []
    for k in range(horizon):
        for group, (start, end) in enumerate(((0, 3), (3, 6))):
            entry = (k, group, start, end)
            (zero_groups if np.sqrt(limits[k, group]) <= 1e-12 else nonzero_groups).append(entry)

    def guard_values(x):
        e = evaluate(x)["errors"]
        return np.asarray(
            [
                (limits[k, g] - e[k, start:end] @ e[k, start:end]) * constraint_scales[g]
                for k, g, start, end in nonzero_groups
            ]
        )

    def guard_jacobian(x):
        item = evaluate(x)
        rows = []
        for k, group, start, end in nonzero_groups:
            rows.append(-2 * constraint_scales[group] * item["errors"][k, start:end] @ item["error_jac"][k, start:end])
        return np.asarray(rows)

    def zero_values(x):
        e = evaluate(x)["errors"]
        return np.concatenate(
            [
                (e[k, start:end] - original["errors"][k, start:end]) * np.sqrt(constraint_scales[g])
                for k, g, start, end in zero_groups
            ]
        )

    def zero_jacobian(x):
        e = evaluate(x)["error_jac"]
        return np.vstack([e[k, start:end] * np.sqrt(constraint_scales[g]) for k, g, start, end in zero_groups])

    def guarded(x):
        if not pose_guard:
            return True
        e = evaluate(x)["errors"]
        return bool(
            np.all(np.linalg.norm(e[:, :3], axis=1) <= np.sqrt(limits[:, 0]) + 1e-10)
            and np.all(np.linalg.norm(e[:, 3:], axis=1) <= np.sqrt(limits[:, 1]) + 1e-10)
        )

    commands, chosen = seed.copy(), original
    message, success = "weak_face_seed", False
    iterations = 0
    if not np.any(original["rollout"]["weak_projection_steps"]):
        constraints = [LinearConstraint(difference, rate_lo, rate_hi)]
        if pose_guard:
            if nonzero_groups:
                constraints.append(dict(type="ineq", fun=guard_values, jac=guard_jacobian))
            if zero_groups:
                # A squared zero norm has zero gradient. Preserve the tiny seed
                # error vector as an equality instead; the final norm gate stays.
                constraints.append(dict(type="eq", fun=zero_values, jac=zero_jacobian))

        def smooth(x):
            item = evaluate(x)
            if np.any(item["rollout"]["weak_projection_steps"]):
                raise _StopSearch("weak_trial_seed")
            return item["score"]["objective"], item["score"]["gradient"]

        try:
            result = minimize(
                smooth,
                seed.ravel(),
                jac=True,
                method="SLSQP",
                bounds=Bounds(lo, hi),
                constraints=constraints,
                options=dict(maxiter=20, ftol=1e-9),
            )
            success, message, iterations = bool(result.success), str(result.message), int(result.nit)
            candidate = np.clip(result.x, lo, hi)
            if feasible(candidate) and guarded(candidate):
                item = evaluate(candidate)
                if (
                    not np.any(item["rollout"]["weak_projection_steps"])
                    and item["score"]["objective"] < original["score"]["objective"]
                ):
                    commands, chosen = candidate.reshape(horizon, 6).copy(), item
        except _StopSearch as error:
            message = str(error)
    return dict(
        planned_commands=commands,
        command=commands[0].copy(),
        seed_objective=original["score"]["objective"],
        objective=chosen["score"]["objective"],
        rollout=chosen["rollout"],
        response_evaluations=len(cache),
        iterations=iterations,
        optimizer_success=success,
        optimizer_message=message,
        accepted=bool(not np.array_equal(commands, seed)),
        pose_guard=pose_guard,
        zero_error_equality_groups=len(zero_groups) if pose_guard else 0,
        scope="four-period ideal joint model optimization only; no native control qualification",
    )
