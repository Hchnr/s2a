"""Find first-pose feasibility before optimizing future tracking quality."""

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize

from s2a.geometry import finite
from s2a.research.joint_control.limits.joint_limit_preview_objective import preview_objective
from s2a.research.joint_control.preview.interior_joint_preview import rollout_joint_preview
from s2a.research.trajectory_search.feasible_start_restoration import domain_check, linear_domain, nearest_linear_plan
from s2a.research.trajectory_search.first_hard_restoration import split_quality_constraints
from s2a.research.trajectory_search.full_horizon_quality import pose_quality_constraints


def first_violation_objective(values, jacobian):
    """Squared normalized first-position and first-rotation shortfall."""
    values, jacobian = finite(values), finite(jacobian)
    if values.shape != (16,) or jacobian.ndim != 2 or jacobian.shape[0] != 16:
        raise ValueError("Eight position/rotation rows and matching Jacobian required")
    violation = np.minimum(values[:2], 0.0)
    return 0.5 * float(violation @ violation), violation @ jacobian[:2]


def solve_staged_first_feasibility(
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
    """Share one finite response cache across first-feasibility and future optimization."""
    if initial_guess is not None or feasibility_restoration is not True or enforce_braking is not True:
        raise ValueError("Only registered restoration with an internally chosen start is supported")
    if not np.isclose(braking_time_s, 0.1, atol=0, rtol=0):
        raise ValueError("The original 0.1 s braking constraint is required")
    seed, previous, velocity = finite(seed, (8, 6)), finite(previous, (6,)), finite(velocity, (7,))
    lower, upper = finite(lower, (7,)), finite(upper, (7,))
    error = np.zeros((25, 7)) if past_inner_error_rad is None else finite(past_inner_error_rad, (25, 7))
    lo, hi = np.tile(interface.input_min, 8), np.tile(interface.input_max, 8)
    rate, center = np.eye(48) - np.eye(48, k=-6), np.r_[previous, np.zeros(42)]
    seed_x = seed.ravel().copy()
    if np.any(seed_x < lo - 1e-12) or np.any(seed_x > hi + 1e-12) or np.max(abs(rate @ seed_x - center)) > 0.25 + 1e-12:
        raise ValueError("The original seed must satisfy command box and rate")
    if np.any(np.all(np.asarray([interface.decode(row) for row in seed])[:, 3:] == 0, axis=1)):
        raise ValueError("The original seed uses an unsupported rotation latch")
    scale = np.tile((interface.output_max - interface.output_min) / (interface.input_max - interface.input_min), 8)
    cache = {}
    stage = "seed"

    class Stop(RuntimeError):
        pass

    def evaluate(x):
        key = tuple(x)
        if key not in cache:
            if len(cache) >= 320:
                raise Stop("response_budget")
            physical = np.asarray([interface.decode(row) for row in np.asarray(x).reshape(8, 6)])
            if np.any(np.all(physical[:, 3:] == 0, axis=1)):
                raise Stop("unsupported_rotation_latch")
            rollout = rollout_joint_preview(model, velocity, physical, site, lower, upper, enforce_limits=False)
            cache[key] = dict(
                rollout=rollout, score=preview_objective(rollout, tool, targets, twists), first_seen=stage
            )
        return cache[key]

    original = evaluate(seed_x)
    domain = linear_domain(model, seed, previous, interface, lower, upper, error, original["rollout"], terminal_normals)
    normals = domain["terminal_normals"]
    initial_margin = float(np.min(np.minimum(model["q"] - lower, upper - model["q"])))
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
        initial_margin_rad=initial_margin,
        scope="staged frozen-model first-pose candidate; future gap is not a full-horizon witness",
    )
    if initial_margin < 0.05 - 1e-12:
        info = dict(
            seed_linear_safe=False,
            start_lp_attempts=0,
            latch_repair=False,
            warm_start_used=False,
            reason="initial_outside_requested_interior",
            seed_rollout_reused=True,
            actual_model_rollouts=1,
            solver_lp_attempted=False,
        )
        return dict(
            **base,
            status="initial_outside_requested_interior",
            witness_found=False,
            response_evaluations=1,
            initialization=info,
            search_phases=[],
        )
    start, info = nearest_linear_plan(seed, domain, interface)
    if start is None:
        info.update(seed_rollout_reused=True, actual_model_rollouts=1, solver_lp_attempted=False)
        return dict(
            **base,
            status="feasible_start_unavailable",
            witness_found=False,
            response_evaluations=1,
            initialization=info,
            search_phases=[],
        )
    stage = "initialization"
    evaluate(start.ravel())

    def pose(x):
        return pose_quality_constraints(evaluate(x)["score"], 8, scale)

    def first_objective(x):
        values, jacobian, _ = pose(x)
        return first_violation_objective(values, jacobian)

    joint_a, joint_lo, joint_hi = (domain[k] for k in ("joint_a", "joint_lo", "joint_hi"))
    terminal_a, terminal_offset = domain["terminal_a"], domain["terminal_offset"]
    joint_constraints = [
        LinearConstraint(rate, center - 0.25, center + 0.25),
        LinearConstraint(joint_a, joint_lo, joint_hi),
    ]
    if len(normals):
        joint_constraints.append(LinearConstraint(terminal_a, -np.inf, -terminal_offset))

    def summarize_candidates():
        safe, first_safe, closest = [], [], None
        for key, item in list(cache.items()):
            x = np.asarray(key)
            if not domain_check(x.reshape(8, 6), domain, interface)["accepted"]:
                continue
            values, _, errors = pose_quality_constraints(item["score"], 8, scale)
            first_position, first_rotation = float(np.linalg.norm(errors[0, :3])), float(np.linalg.norm(errors[0, 3:]))
            violation = max(0.0, first_position / 0.01 - 1, first_rotation / 0.05 - 1)
            future = max(0.0, float(-np.min(values[2:])))
            row = (x, item, errors, future, max(0.0, float(-np.min(values))), violation)
            safe.append(row)
            if closest is None or violation < closest[5]:
                closest = row
            if first_position <= 0.01 + 1e-10 and first_rotation <= 0.05 + 1e-10:
                first_safe.append(row)
        return safe, first_safe, closest

    phases = []
    stage = "first_feasibility"
    first_iterations, interrupted = None, False
    try:
        first = minimize(
            first_objective,
            start.ravel(),
            jac=True,
            method="SLSQP",
            bounds=Bounds(lo, hi),
            constraints=joint_constraints,
            options={"maxiter": 40, "ftol": 1e-12},
        )
        first_iterations = int(first.nit)
        if not 0 <= first_iterations <= 40:
            raise ValueError("First-stage iteration budget exceeded")
        phases.append(
            dict(
                phase=stage,
                iterations=first_iterations,
                success=bool(first.success),
                message=str(first.message),
                violation_objective=float(first.fun),
                response_evaluations=len(cache),
            )
        )
    except Stop as exc:
        interrupted = True
        phases.append(
            dict(phase=stage, iterations=None, success=False, message=str(exc), response_evaluations=len(cache))
        )

    safe, first_safe, closest = summarize_candidates()
    if not interrupted and closest is not None:
        phase_start = min(first_safe, key=lambda row: (row[3], row[1]["score"]["objective"])) if first_safe else closest
        start_x = phase_start[0].copy()
        start_slack = phase_start[3] + 1e-6

        def future_objective(y):
            difference = y[:-1] - start_x
            return float(y[-1] + 1e-9 * (difference @ difference)), np.r_[2e-9 * difference, 1.0]

        def split(y):
            values, jacobian, _ = pose(y[:-1])
            return split_quality_constraints(values, jacobian, y[-1])

        future_constraints = [
            LinearConstraint(np.c_[rate, np.zeros(48)], center - 0.25, center + 0.25),
            LinearConstraint(np.c_[joint_a, np.zeros(len(joint_a))], joint_lo, joint_hi),
            {"type": "ineq", "fun": lambda y: split(y)[0], "jac": lambda y: split(y)[1]},
            {"type": "ineq", "fun": lambda y: split(y)[2], "jac": lambda y: split(y)[3]},
        ]
        if len(normals):
            future_constraints.append(
                LinearConstraint(np.c_[terminal_a, np.zeros(len(normals))], -np.inf, -terminal_offset)
            )
        stage = "future_quality"
        remaining = 120 - first_iterations
        try:
            second = minimize(
                future_objective,
                np.r_[start_x, start_slack],
                jac=True,
                method="SLSQP",
                bounds=Bounds(np.r_[lo, 0.0], np.r_[hi, np.inf]),
                constraints=future_constraints,
                options={"maxiter": remaining, "ftol": 1e-9},
            )
            second_iterations = int(second.nit)
            if not 0 <= second_iterations <= remaining:
                raise ValueError("Shared iteration budget exceeded")
            phases.append(
                dict(
                    phase=stage,
                    iterations=second_iterations,
                    success=bool(second.success),
                    message=str(second.message),
                    optimizer_future_slack=float(second.x[-1]),
                    response_evaluations=len(cache),
                )
            )
        except Stop as exc:
            phases.append(
                dict(phase=stage, iterations=None, success=False, message=str(exc), response_evaluations=len(cache))
            )
    safe, first_safe, closest = summarize_candidates()
    rejected = None
    if closest is not None and closest[5] > 0:
        rejected = dict(
            first_tcp_error=closest[2][0].copy(),
            first_normalized_violation=closest[5],
            future_quality_slack=closest[3],
            planned_commands=closest[0].reshape(8, 6).copy(),
        )
    info.update(seed_rollout_reused=True, actual_model_rollouts=len(cache), solver_lp_attempted=False)
    base.update(
        response_evaluations=len(cache),
        initialization=info,
        search_phases=phases,
        joint_safe_evaluations=len(safe),
        first_hard_evaluations=len(first_safe),
        closest_first_rejected=rejected,
        witness_found=False,
    )
    if not first_safe:
        return dict(**base, status="staged_first_no_witness")
    best = min(first_safe, key=lambda row: (row[3], row[1]["score"]["objective"]))
    x, item, errors, future_slack, overall_slack, _ = best
    base["best_joint_safe"] = dict(
        witness_found=True,
        planned_commands=x.reshape(8, 6).copy(),
        terminal_normals=normals.copy(),
        rollout=item["rollout"],
        first_tcp_error=errors[0].copy(),
        tcp_errors=errors.copy(),
        quality_slack=overall_slack,
        future_quality_slack=future_slack,
        first_pose_optimized_hard=item["first_seen"] == "future_quality",
        first_pose_selection_hard=True,
    )
    return dict(**base, status="staged_first_restoration_candidate")
