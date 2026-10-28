"""Constrained preview with FK-consistent Cartesian states and checked LP starts."""

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, minimize

from s2a.fk_consistent_joint_preview import fk_consistent_rollout
from s2a.geometry import finite
from s2a.interior_joint_preview import rollout_joint_preview
from s2a.joint_limit_preview_objective import preview_objective


def solve_interior_plan(
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
    fk_jacobian,
    initial_guess=None,
    terminal_normals=None,
    braking_time_s=0.1,
    past_inner_error_rad=None,
):
    """Find a frozen-model witness with original pose caps at every horizon point.

    Joint positions are affine in actions for frozen free M/J, including anchor
    resets. All 2ms endpoints, not only 50ms control endpoints, are constrained.
    This diagnostic does not model contacts, torque saturation or nonlinear M/J.
    """
    if not np.isfinite(braking_time_s) or braking_time_s <= 0:
        raise ValueError("Positive finite braking lookahead required")
    seed, previous = finite(seed), finite(previous, (6,))
    if seed.ndim != 2 or seed.shape[1] != 6 or not 1 <= len(seed) <= 16:
        raise ValueError("One to sixteen seed commands required")
    initial_pose, initial_jacobian = fk_jacobian(model["q"])
    np.testing.assert_allclose(initial_pose, site, atol=1e-10, rtol=0)
    np.testing.assert_allclose(initial_jacobian, model["jacobian"], atol=1e-8, rtol=0)
    lower, upper = finite(lower, (len(model["q"]),)), finite(upper, (len(model["q"]),))
    error = (
        np.zeros((25, len(lower))) if past_inner_error_rad is None else finite(past_inner_error_rad, (25, len(lower)))
    )
    if np.any(error < 0):
        raise ValueError("Nonnegative past position error required")
    if np.any(np.diff(error, axis=0) < 0):
        raise ValueError("Nondecreasing inner-time error profile required")
    peak = error.max(axis=0)
    margin = 0.05 + peak
    if np.any(lower + margin >= upper - margin):
        raise ValueError("Joint ranges must contain requested .05rad interior")
    h, width = len(seed), seed.size
    position_margin = np.tile(margin, (25 * h, 1))
    position_margin[:25] = 0.05 + error
    lo, hi = np.tile(interface.input_min, h), np.tile(interface.input_max, h)
    d = np.eye(width) - np.eye(width, k=-6)
    center = np.r_[previous, np.zeros(width - 6)]
    if (
        np.any(seed.ravel() < lo - 1e-12)
        or np.any(seed.ravel() > hi + 1e-12)
        or np.max(abs(d @ seed.ravel() - center)) > 0.25 + 1e-12
    ):
        raise ValueError("Box/rate feasible seed required")
    scale = np.tile((interface.output_max - interface.output_min) / (interface.input_max - interface.input_min), h)
    cache = {}

    class Stop(RuntimeError):
        pass

    def evaluate(x):
        key = tuple(x)
        if key not in cache:
            if len(cache) >= 160:
                raise Stop("response_budget")
            delta = np.asarray([interface.decode(c) for c in np.asarray(x).reshape(h, 6)])
            if np.any(np.all(delta[:, 3:] == 0, axis=1)):
                raise Stop("unsupported_rotation_latch")
            rollout = rollout_joint_preview(model, velocity, delta, site, lower, upper, enforce_limits=False)
            rollout = fk_consistent_rollout(rollout, fk_jacobian)
            cache[key] = dict(rollout=rollout, score=preview_objective(rollout, tool, targets, twists))
        return cache[key]

    original = evaluate(seed.ravel())
    q = original["rollout"]["inner_joint_position"].ravel()
    a = original["rollout"]["inner_joint_position_jacobian"].reshape(len(q), width) * scale
    offset = q - a @ seed.ravel()
    joint_lo = (lower + position_margin).ravel() - offset
    joint_hi = (upper - position_margin).ravel() - offset
    braking_q = original["rollout"]["joint_position"] + braking_time_s * original["rollout"]["joint_velocity"]
    braking_jac = (
        original["rollout"]["joint_position_jacobian"] + braking_time_s * original["rollout"]["joint_velocity_jacobian"]
    )
    braking_a = braking_jac.reshape(braking_q.size, width) * scale
    braking_offset = braking_q.ravel() - braking_a @ seed.ravel()
    a = np.vstack((a, braking_a))
    offset = np.r_[offset, braking_offset]
    joint_lo = np.r_[joint_lo, np.tile(lower + 0.05, h) - braking_offset]
    joint_hi = np.r_[joint_hi, np.tile(upper - 0.05, h) - braking_offset]
    first = original["score"]["residual"][:6] / np.array([100.0] * 3 + [20.0] * 3)
    seed_comparison_caps = np.maximum([np.linalg.norm(first[:3]), np.linalg.norm(first[3:])], [0.01, 0.05])
    caps = np.array([0.01, 0.05])
    if terminal_normals is None:
        inner = original["rollout"]["inner_joint_position"]
        identity = np.eye(len(lower))
        terminal_normals = np.vstack(
            (identity[np.any(inner >= upper - 0.05, axis=0)], -identity[np.any(inner <= lower + 0.05, axis=0)])
        )
    normals = finite(terminal_normals)
    if normals.ndim != 2 or normals.shape[1] != len(lower):
        raise ValueError("Finite terminal joint direction rows required")
    terminal_jac = normals @ original["rollout"]["joint_velocity_jacobian"][-1] * scale[None, :]
    terminal_offset = normals @ original["rollout"]["joint_velocity"][-1] - terminal_jac @ seed.ravel()
    base = dict(
        seed_commands=seed.copy(),
        first_pose_caps=caps,
        seed_comparison_caps=seed_comparison_caps,
        tracking_caps_enforced=True,
        all_pose_caps_enforced=True,
        pose_caps_horizon=h,
        joint_margin_rad=0.05,
        braking_time_s=float(braking_time_s),
        past_inner_error_rad=error.copy(),
        requested_position_margin_rad=position_margin.copy(),
        error_scope="Past inner-time profile tightens first period; past peak retained for all later periods; no future guarantee",
        braking_constraint_scope="q+tau*v at all control endpoints plus all inner position constraints",
        terminal_normals=normals.copy(),
        search_policy="First minimize all-horizon pose-cap violation under original linear constraints; then optimize full preview with all-horizon pose caps",
        response_budget=160,
        iteration_budget=60,
        feasibility_iteration_budget=30,
        initial_margin_rad=float(np.min(np.minimum(model["q"] - lower, upper - model["q"]))),
        scope="frozen free joint dynamics with nonlinear FK(q)/J(q) qdot; no physical control qualification",
    )
    if base["initial_margin_rad"] < 0.05 - 1e-12:
        return dict(
            **base,
            status="initial_outside_requested_interior",
            witness_found=False,
            response_evaluations=len(cache),
            optimizer_success=False,
        )
    lp_options = dict(primal_feasibility_tolerance=1e-10, dual_feasibility_tolerance=1e-10)
    base["linear_solver_options"] = lp_options
    lp = linprog(
        np.zeros(width),
        A_ub=np.vstack((a, -a, d, -d, terminal_jac)),
        b_ub=np.r_[joint_hi, -joint_lo, center + 0.25, -center + 0.25, -terminal_offset],
        bounds=list(zip(lo, hi, strict=True)),
        method="highs",
        options=lp_options,
    )
    base.update(linear_status=int(lp.status), linear_message=str(lp.message))
    if not lp.success:
        return dict(
            **base,
            status="linear_solver_no_witness",
            witness_found=False,
            response_evaluations=len(cache),
            optimizer_success=False,
        )
    base["linear_witness_commands"] = lp.x.reshape(h, 6).copy()

    def pose_errors(x):
        return evaluate(x)["score"]["residual"].reshape(h, 12)[:, :6] / np.array([100.0] * 3 + [20.0] * 3)

    def pose_values(x):
        errors = pose_errors(x)
        return np.column_stack(
            (1 - np.linalg.norm(errors[:, :3], axis=1) / caps[0], 1 - np.linalg.norm(errors[:, 3:], axis=1) / caps[1])
        ).ravel()

    def pose_jac(x):
        item = evaluate(x)["score"]
        errors = pose_errors(x)
        jac = (
            item["jacobian"].reshape(h, 12, width)[:, :6]
            * scale[None, None, :]
            / np.array([100.0] * 3 + [20.0] * 3)[None, :, None]
        )
        rows = []
        for period in range(h):
            for group, (start, end) in enumerate(((0, 3), (3, 6))):
                error = errors[period, start:end]
                norm = np.linalg.norm(error)
                rows.append(-(error / norm if norm > 1e-15 else np.zeros(3)) @ jac[period, start:end] / caps[group])
        return np.asarray(rows)

    def objective(x):
        score = evaluate(x)["score"]
        return score["objective"], score["gradient"] * scale

    start = seed.ravel().copy()
    if initial_guess is not None:
        guess = finite(initial_guess, seed.shape)
        x = guess.ravel()
        if np.any(x < lo - 1e-12) or np.any(x > hi + 1e-12) or np.max(abs(d @ x - center)) > 0.25 + 1e-12:
            raise ValueError("Warm start must satisfy current command box/rate")
        try:
            evaluate(x)  # Preserve a checked feasible warm candidate even if later search exhausts budget.
        except Stop as error:
            raise ValueError("Warm start uses unsupported latch mode") from error
        start = x.copy()
    base["initial_guess_used"] = initial_guess is not None

    def linear_feasible(x):
        return bool(
            np.all(x >= lo - 1e-12)
            and np.all(x <= hi + 1e-12)
            and np.max(abs(d @ x - center)) <= 0.25 + 1e-12
            and np.all(a @ x >= joint_lo - 1e-10)
            and np.all(a @ x <= joint_hi + 1e-10)
            and np.all(terminal_jac @ x + terminal_offset <= 1e-10)
        )

    base["provided_start_linear_feasible"] = linear_feasible(start)
    base["nonlinear_start_policy"] = "provided_start"
    if not base["provided_start_linear_feasible"]:
        candidate = lp.x.copy()
        base["linear_witness_residuals"] = dict(
            box_excess=float(max(0.0, np.max(lo - candidate), np.max(candidate - hi))),
            rate_excess=float(max(0.0, np.max(abs(d @ candidate - center)) - 0.25)),
            joint_braking_excess=float(max(0.0, np.max(joint_lo - a @ candidate), np.max(a @ candidate - joint_hi))),
            terminal_excess=float(max(0.0, np.max(terminal_jac @ candidate + terminal_offset)))
            if len(normals)
            else 0.0,
        )
        if not linear_feasible(candidate):
            return dict(
                **base,
                status="linear_witness_not_certified",
                witness_found=False,
                response_evaluations=len(cache),
                optimizer_success=False,
                selected_start_linear_feasible=False,
                nonlinear_start_policy="rejected_uncertified_linear_witness",
            )
        try:
            evaluate(candidate)
        except Stop as error:
            # The fixed free model excludes exact-zero rotation latch actions.
            # Keep the declared original start, report this exception, and let
            # the unchanged final witness checks decide; no constraint relaxation.
            base["nonlinear_start_policy"] = "original_start_due_to_unsupported_lp_latch"
            base["linear_start_error"] = str(error)
        else:
            start = candidate
            base["nonlinear_start_policy"] = "verified_linear_witness"
    base["selected_start_linear_feasible"] = linear_feasible(start)

    def feasibility_objective(x):
        violation = np.maximum(-pose_values(x), 0.0)
        return 0.5 * float(violation @ violation), -violation @ pose_jac(x)

    linear_constraints = [
        LinearConstraint(d, center - 0.25, center + 0.25),
        LinearConstraint(a, joint_lo, joint_hi),
    ] + ([LinearConstraint(terminal_jac, -np.inf, -terminal_offset)] if len(normals) else [])
    success, iterations, message = False, 0, "not_run"
    phases = []
    try:
        first_result = minimize(
            feasibility_objective,
            start,
            jac=True,
            method="SLSQP",
            bounds=Bounds(lo, hi),
            constraints=linear_constraints,
            options=dict(maxiter=30, ftol=1e-12),
        )
        iterations = int(first_result.nit)
        phases.append(
            dict(
                phase="pose_feasibility",
                iterations=iterations,
                success=bool(first_result.success),
                message=str(first_result.message),
                violation_objective=float(first_result.fun),
                response_evaluations=len(cache),
            )
        )
        result = minimize(
            objective,
            first_result.x,
            jac=True,
            method="SLSQP",
            bounds=Bounds(lo, hi),
            constraints=linear_constraints + [dict(type="ineq", fun=pose_values, jac=pose_jac)],
            options=dict(maxiter=max(1, 60 - iterations), ftol=1e-9),
        )
        iterations += int(result.nit)
        success, message = bool(result.success), str(result.message)
        phases.append(
            dict(
                phase="full_preview",
                iterations=int(result.nit),
                success=success,
                message=message,
                response_evaluations=len(cache),
            )
        )
    except Stop as error:
        message = str(error)
        phases.append(dict(phase="shared_budget_or_latch_stop", message=message, response_evaluations=len(cache)))
    base["search_phases"] = phases
    best = None
    closest = None
    for key, item in list(cache.items()):
        x = np.asarray(key)
        if (
            np.all(x >= lo - 1e-12)
            and np.all(x <= hi + 1e-12)
            and np.max(abs(d @ x - center)) <= 0.25 + 1e-12
            and np.all(a @ x >= joint_lo - 1e-10)
            and np.all(a @ x <= joint_hi + 1e-10)
            and np.all(terminal_jac @ x + terminal_offset <= 1e-10)
        ):
            errors = item["score"]["residual"].reshape(h, 12)[:, :6] / np.array([100.0] * 3 + [20.0] * 3)
            first_error = errors[0]
            violation = max(
                float(np.max(np.linalg.norm(errors[:, :3], axis=1) / caps[0] - 1)),
                float(np.max(np.linalg.norm(errors[:, 3:], axis=1) / caps[1] - 1)),
                0.0,
            )
            if closest is None or violation < closest["maximum_normalized_pose_cap_violation"]:
                closest = dict(
                    maximum_normalized_pose_cap_violation=float(violation),
                    first_tcp_error=first_error.copy(),
                    all_tcp_errors=errors.copy(),
                    commands=x.reshape(h, 6).copy(),
                    scope="Best checked linear-feasible all-horizon attempt; not a global lower bound or physical guarantee",
                )
            if np.all(np.linalg.norm(errors[:, :3], axis=1) <= caps[0] + 1e-10) and np.all(
                np.linalg.norm(errors[:, 3:], axis=1) <= caps[1] + 1e-10
            ):
                if best is None or item["score"]["objective"] < best[1]["score"]["objective"]:
                    best = (x, item)
    base["closest_linear_feasible_attempt"] = closest
    base.update(
        optimizer_success=success,
        iterations=iterations,
        optimizer_message=message,
        response_evaluations=len(cache),
        witness_found=best is not None,
    )
    if best is None:
        return dict(**base, status="nonlinear_search_no_witness")
    x, item = best
    actual = item["rollout"]["inner_joint_position"]
    braking_actual = item["rollout"]["joint_position"] + braking_time_s * item["rollout"]["joint_velocity"]
    np.testing.assert_allclose(np.r_[actual.ravel(), braking_actual.ravel()], a @ x + offset, atol=1e-10, rtol=0)
    return dict(
        **base,
        status="model_witness",
        planned_commands=x.reshape(h, 6).copy(),
        first_tcp_error=item["score"]["residual"][:6] / np.array([100.0] * 3 + [20.0] * 3),
        all_tcp_errors=item["score"]["residual"].reshape(h, 12)[:, :6] / np.array([100.0] * 3 + [20.0] * 3),
        original_first_pose_caps_satisfied=bool(
            np.linalg.norm(item["score"]["residual"][:3]) / 100.0 <= caps[0] + 1e-10
            and np.linalg.norm(item["score"]["residual"][3:6]) / 20.0 <= caps[1] + 1e-10
        ),
        first_pose_caps_enforced=True,
        objective=item["score"]["objective"],
        seed_objective=original["score"]["objective"],
        minimum_tightened_inner_slack_rad=float(
            np.min(np.minimum(actual - lower - position_margin, upper - position_margin - actual))
        ),
        minimum_braking_margin_rad=float(np.min(np.minimum(braking_actual - lower, upper - braking_actual))),
        minimum_inner_margin_rad=float(np.min(np.minimum(actual - lower, upper - actual))),
        terminal_signed_velocities_rad_s=normals @ item["rollout"]["joint_velocity"][-1],
        rollout=item["rollout"],
    )
