"""Minimize future quality shortfall while keeping first-step quality hard."""

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, minimize

from s2a.geometry import finite
from s2a.research.joint_control.limits.joint_limit_preview_objective import preview_objective
from s2a.research.joint_control.preview.interior_joint_preview import rollout_joint_preview
from s2a.research.trajectory_search.full_horizon_quality import pose_quality_constraints


def split_quality_constraints(values, jacobian, slack):
    """Keep the first position/rotation rows hard and relax later rows only."""
    values, jacobian = finite(values), finite(jacobian)
    if values.shape != (16,) or jacobian.ndim != 2 or jacobian.shape[0] != 16:
        raise ValueError("Eight position/rotation rows and matching Jacobian required")
    if not np.isfinite(slack) or slack < 0:
        raise ValueError("A nonnegative finite future slack is required")
    hard_jacobian = np.c_[jacobian[:2], np.zeros(2)]
    future_jacobian = np.c_[jacobian[2:], np.ones(14)]
    return values[:2].copy(), hard_jacobian, values[2:] + slack, future_jacobian


def solve_first_hard_restoration(
    model,
    velocity,
    seed,
    previous,
    interface,
    site,
    tool,
    targets,
    twists,
    lower,
    upper,
    *,
    initial_guess=None,
    past_inner_error_rad=None,
    terminal_normals=None,
    braking_time_s=0.1,
    feasibility_restoration=True,
    enforce_braking=True,
):
    """Return a first-pose-safe model candidate, never a full-horizon claim."""
    if feasibility_restoration is not True or enforce_braking is not True:
        raise ValueError("First-hard restoration and braking must remain enabled")
    if not np.isclose(braking_time_s, 0.1, atol=0, rtol=0):
        raise ValueError("The original 0.1 s braking constraint is required")
    seed, previous = finite(seed, (8, 6)), finite(previous, (6,))
    lower, upper = finite(lower, (7,)), finite(upper, (7,))
    error = np.zeros((25, 7)) if past_inner_error_rad is None else finite(past_inner_error_rad, (25, 7))
    if np.any(error < 0) or np.any(np.diff(error, axis=0) < 0):
        raise ValueError("Only a nonnegative completed inner-error profile is accepted")
    peak = error.max(axis=0)
    if np.any(lower + 0.05 + peak >= upper - 0.05 - peak):
        raise ValueError("Joint ranges must contain the original tightened interior")
    margin = np.tile(0.05 + peak, (200, 1))
    margin[:25] = 0.05 + error
    lo, hi = np.tile(interface.input_min, 8), np.tile(interface.input_max, 8)
    rate = np.eye(48) - np.eye(48, k=-6)
    center = np.r_[previous, np.zeros(42)]

    def check_command(candidate):
        x = finite(candidate, (8, 6)).ravel()
        if np.any(x < lo - 1e-12) or np.any(x > hi + 1e-12) or np.max(abs(rate @ x - center)) > 0.25 + 1e-12:
            raise ValueError("Seed and warm plan must satisfy the original command box and rate")
        if np.any(np.all(np.asarray([interface.decode(row) for row in x.reshape(8, 6)])[:, 3:] == 0, axis=1)):
            raise ValueError("Native rotation latch is unsupported in the frozen model")
        return x.copy()

    seed_x = check_command(seed)
    start = seed_x.copy() if initial_guess is None else check_command(initial_guess)
    scale = np.tile((interface.output_max - interface.output_min) / (interface.input_max - interface.input_min), 8)
    cache = {}

    class Stop(RuntimeError):
        pass

    def evaluate(x):
        key = tuple(x)
        if key not in cache:
            if len(cache) >= 320:
                raise Stop("response_budget")
            delta = np.asarray([interface.decode(row) for row in np.asarray(x).reshape(8, 6)])
            if np.any(np.all(delta[:, 3:] == 0, axis=1)):
                raise Stop("unsupported_rotation_latch")
            rollout = rollout_joint_preview(model, velocity, delta, site, lower, upper, enforce_limits=False)
            cache[key] = dict(rollout=rollout, score=preview_objective(rollout, tool, targets, twists))
        return cache[key]

    original = evaluate(seed_x)
    rollout = original["rollout"]
    q = rollout["inner_joint_position"].ravel()
    joint_a = rollout["inner_joint_position_jacobian"].reshape(len(q), 48) * scale
    joint_offset = q - joint_a @ seed_x
    joint_lo, joint_hi = (lower + margin).ravel() - joint_offset, (upper - margin).ravel() - joint_offset
    brake = rollout["joint_position"] + 0.1 * rollout["joint_velocity"]
    brake_jac = rollout["joint_position_jacobian"] + 0.1 * rollout["joint_velocity_jacobian"]
    brake_a = brake_jac.reshape(brake.size, 48) * scale
    brake_offset = brake.ravel() - brake_a @ seed_x
    joint_a = np.vstack((joint_a, brake_a))
    joint_lo = np.r_[joint_lo, np.tile(lower + 0.05, 8) - brake_offset]
    joint_hi = np.r_[joint_hi, np.tile(upper - 0.05, 8) - brake_offset]
    if terminal_normals is None:
        inner, identity = rollout["inner_joint_position"], np.eye(7)
        terminal_normals = np.vstack(
            (identity[np.any(inner >= upper - 0.05, axis=0)], -identity[np.any(inner <= lower + 0.05, axis=0)])
        )
    normals = finite(terminal_normals)
    if normals.ndim != 2 or normals.shape[1] != 7:
        raise ValueError("Finite terminal joint direction rows required")
    terminal_a = normals @ rollout["joint_velocity_jacobian"][-1] * scale[None, :]
    terminal_offset = normals @ rollout["joint_velocity"][-1] - terminal_a @ seed_x
    base = dict(
        seed_commands=seed.copy(),
        terminal_normals=normals.copy(),
        past_inner_error_rad=error.copy(),
        joint_margin_rad=0.05,
        braking_time_s=0.1,
        braking_constraint_enforced=True,
        position_cap_m=0.01,
        rotation_cap_rad=0.05,
        pose_horizon=8,
        initial_margin_rad=float(np.min(np.minimum(model["q"] - lower, upper - model["q"]))),
        initial_guess_used=initial_guess is not None,
        first_pose_optimized_hard=True,
        scope="first-pose-hard frozen-model candidate; future quality gap is not a full-horizon witness",
    )
    if base["initial_margin_rad"] < 0.05 - 1e-12:
        return dict(**base, status="initial_outside_requested_interior", witness_found=False, response_evaluations=1)
    linear = linprog(
        np.zeros(48),
        A_ub=np.vstack((joint_a, -joint_a, rate, -rate, terminal_a)),
        b_ub=np.r_[joint_hi, -joint_lo, center + 0.25, -center + 0.25, -terminal_offset],
        bounds=list(zip(lo, hi, strict=True)),
        method="highs",
    )
    base.update(linear_status=int(linear.status), linear_message=str(linear.message))
    if not linear.success:
        return dict(**base, status="linear_solver_no_witness", witness_found=False, response_evaluations=len(cache))

    def pose(x):
        return pose_quality_constraints(evaluate(x)["score"], 8, scale)

    def objective(y):
        delta = y[:-1] - start
        return float(y[-1] + 1e-9 * (delta @ delta)), np.r_[2e-9 * delta, 1.0]

    def constraints(y):
        values, jacobian, _ = pose(y[:-1])
        return split_quality_constraints(values, jacobian, y[-1])

    if initial_guess is not None:
        try:
            evaluate(start)
        except Stop as exc:
            raise ValueError("Warm plan uses unsupported latch mode") from exc
    start_slack = max(0.0, -float(np.min(pose(start)[0][2:]))) + 1e-6
    linear_constraints = [
        LinearConstraint(np.c_[rate, np.zeros(48)], center - 0.25, center + 0.25),
        LinearConstraint(np.c_[joint_a, np.zeros(len(joint_a))], joint_lo, joint_hi),
        {"type": "ineq", "fun": lambda y: constraints(y)[0], "jac": lambda y: constraints(y)[1]},
        {"type": "ineq", "fun": lambda y: constraints(y)[2], "jac": lambda y: constraints(y)[3]},
    ]
    if len(normals):
        linear_constraints.append(
            LinearConstraint(np.c_[terminal_a, np.zeros(len(normals))], -np.inf, -terminal_offset)
        )
    success, iterations, message, optimizer_slack = False, 0, "not_run", None
    try:
        optimized = minimize(
            objective,
            np.r_[start, start_slack],
            jac=True,
            method="SLSQP",
            bounds=Bounds(np.r_[lo, 0.0], np.r_[hi, np.inf]),
            constraints=linear_constraints,
            options={"maxiter": 120, "ftol": 1e-9},
        )
        success, iterations, message, optimizer_slack = (
            bool(optimized.success),
            int(optimized.nit),
            str(optimized.message),
            float(optimized.x[-1]),
        )
    except Stop as exc:
        message = str(exc)

    best, rejected, joint_safe_count, first_safe_count = None, None, 0, 0
    for key, item in list(cache.items()):
        x = np.asarray(key)
        values, _, errors = pose_quality_constraints(item["score"], 8, scale)
        joint_safe = (
            np.all(x >= lo - 1e-12)
            and np.all(x <= hi + 1e-12)
            and np.max(abs(rate @ x - center)) <= 0.25 + 1e-12
            and np.all(joint_a @ x >= joint_lo - 1e-10)
            and np.all(joint_a @ x <= joint_hi + 1e-10)
            and np.all(terminal_a @ x + terminal_offset <= 1e-10)
        )
        if not joint_safe:
            continue
        joint_safe_count += 1
        first_position, first_rotation = float(np.linalg.norm(errors[0, :3])), float(np.linalg.norm(errors[0, 3:]))
        first_violation = max(0.0, first_position / 0.01 - 1, first_rotation / 0.05 - 1)
        future_slack = max(0.0, float(-np.min(values[2:])))
        if first_position > 0.01 + 1e-10 or first_rotation > 0.05 + 1e-10:
            if rejected is None or first_violation < rejected["first_normalized_violation"]:
                rejected = dict(
                    first_tcp_error=errors[0].copy(),
                    first_normalized_violation=first_violation,
                    future_quality_slack=future_slack,
                    planned_commands=x.reshape(8, 6).copy(),
                )
            continue
        first_safe_count += 1
        if best is None or (future_slack, item["score"]["objective"]) < (best[3], best[1]["score"]["objective"]):
            best = (x, item, errors, future_slack, max(0.0, float(-np.min(values))))
    base.update(
        optimizer_success=success,
        iterations=iterations,
        optimizer_message=message,
        optimizer_future_slack=optimizer_slack,
        response_evaluations=len(cache),
        joint_safe_evaluations=joint_safe_count,
        first_hard_evaluations=first_safe_count,
        closest_first_rejected=rejected,
        witness_found=False,
    )
    if best is None:
        return dict(**base, status="first_hard_search_no_witness")
    x, item, errors, future_slack, overall_slack = best
    base["best_joint_safe"] = dict(
        witness_found=True,
        planned_commands=x.reshape(8, 6).copy(),
        terminal_normals=normals.copy(),
        rollout=item["rollout"],
        first_tcp_error=errors[0].copy(),
        tcp_errors=errors.copy(),
        quality_slack=overall_slack,
        future_quality_slack=future_slack,
        first_pose_optimized_hard=True,
    )
    return dict(**base, status="first_hard_restoration_candidate")
