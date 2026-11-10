"""Find two hard pose points before optimizing a softer six-point suffix."""

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize

from s2a.geometry import finite
from s2a.research.joint_control.limits.joint_limit_preview_objective import preview_objective
from s2a.research.joint_control.preview.interior_joint_preview import rollout_joint_preview
from s2a.research.trajectory_search.feasible_start_restoration import domain_check, linear_domain, nearest_linear_plan
from s2a.research.trajectory_search.full_horizon_quality import pose_quality_constraints


def prefix_violation_objective(values, jacobian):
    """Squared normalized position and rotation shortfalls for two points."""
    values, jacobian = finite(values), finite(jacobian)
    if values.shape != (16,) or jacobian.ndim != 2 or jacobian.shape[0] != 16:
        raise ValueError("Eight position/rotation rows and matching Jacobian required")
    violation = np.minimum(values[:4], 0.0)
    return 0.5 * float(violation @ violation), violation @ jacobian[:4]


def split_prefix_constraints(values, jacobian, slack):
    """Keep the first two pose points hard while relaxing only later points."""
    values, jacobian = finite(values), finite(jacobian)
    if values.shape != (16,) or jacobian.ndim != 2 or jacobian.shape[0] != 16 or not np.isfinite(slack) or slack < 0:
        raise ValueError("Eight quality points and one nonnegative future slack required")
    return (
        values[:4],
        np.c_[jacobian[:4], np.zeros(4)],
        values[4:] + slack,
        np.c_[jacobian[4:], np.ones(12)],
    )


def solve_two_step_viability(
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
    past_inner_error_rad=None,
    terminal_normals=None,
    on_response=None,
):
    """Return only evaluated joint-safe plans with both first poses within caps."""
    seed, previous, velocity = finite(seed, (8, 6)), finite(previous, (6,)), finite(velocity, (7,))
    lower, upper = finite(lower, (7,)), finite(upper, (7,))
    error = np.zeros((25, 7)) if past_inner_error_rad is None else finite(past_inner_error_rad, (25, 7))
    lo, hi = np.tile(interface.input_min, 8), np.tile(interface.input_max, 8)
    rate, center = np.eye(48) - np.eye(48, k=-6), np.r_[previous, np.zeros(42)]
    seed_x = seed.ravel().copy()
    if np.any(seed_x < lo - 1e-12) or np.any(seed_x > hi + 1e-12) or np.max(abs(rate @ seed_x - center)) > 0.25 + 1e-12:
        raise ValueError("The seed must satisfy the original command box and rate")
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
            if on_response is not None:
                on_response("attempted")
            rollout = rollout_joint_preview(model, velocity, physical, site, lower, upper, enforce_limits=False)
            if on_response is not None:
                on_response("completed")
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
        seed_tcp_errors=pose_quality_constraints(original["score"], 8, scale)[2],
        terminal_normals=normals.copy(),
        past_inner_error_rad=error.copy(),
        initial_margin_rad=initial_margin,
        joint_margin_rad=0.05,
        braking_time_s=0.1,
        position_cap_m=0.01,
        rotation_cap_rad=0.05,
        hard_pose_prefix=2,
        pose_horizon=8,
        witness_found=False,
        scope="two hard frozen-model poses; future gap is not a full-horizon witness",
    )
    if initial_margin < 0.05 - 1e-12:
        return dict(
            **base,
            status="initial_outside_requested_interior",
            response_evaluations=1,
            initialization=dict(start_lp_attempts=0, reason="initial_outside_requested_interior"),
            search_phases=[],
        )
    start, info = nearest_linear_plan(seed, domain, interface)
    if start is None:
        return dict(
            **base, status="feasible_start_unavailable", response_evaluations=1, initialization=info, search_phases=[]
        )
    stage = "initialization"
    evaluate(start.ravel())

    def pose(x):
        return pose_quality_constraints(evaluate(x)["score"], 8, scale)

    def prefix_objective(x):
        values, jacobian, _ = pose(x)
        return prefix_violation_objective(values, jacobian)

    joint_a, joint_lo, joint_hi = (domain[k] for k in ("joint_a", "joint_lo", "joint_hi"))
    terminal_a, terminal_offset = domain["terminal_a"], domain["terminal_offset"]
    joint_constraints = [
        LinearConstraint(rate, center - 0.25, center + 0.25),
        LinearConstraint(joint_a, joint_lo, joint_hi),
    ]
    if len(normals):
        joint_constraints.append(LinearConstraint(terminal_a, -np.inf, -terminal_offset))

    def summarize_candidates():
        safe, prefix_safe, closest = [], [], None
        for key, item in list(cache.items()):
            x = np.asarray(key)
            if not domain_check(x.reshape(8, 6), domain, interface)["accepted"]:
                continue
            values, _, errors = pose_quality_constraints(item["score"], 8, scale)
            position = np.linalg.norm(errors[:2, :3], axis=1)
            rotation = np.linalg.norm(errors[:2, 3:], axis=1)
            prefix_violation = max(0.0, float(np.max(position / 0.01 - 1)), float(np.max(rotation / 0.05 - 1)))
            future = max(0.0, float(-np.min(values[4:])))
            row = (x, item, errors, future, max(0.0, float(-np.min(values))), prefix_violation)
            safe.append(row)
            if closest is None or prefix_violation < closest[5]:
                closest = row
            if np.all(position <= 0.01 + 1e-10) and np.all(rotation <= 0.05 + 1e-10):
                prefix_safe.append(row)
        return safe, prefix_safe, closest

    phases = []
    stage = "two_point_feasibility"
    first_iterations, interrupted = None, False
    try:
        first = minimize(
            prefix_objective,
            start.ravel(),
            jac=True,
            method="SLSQP",
            bounds=Bounds(lo, hi),
            constraints=joint_constraints,
            options={"maxiter": 60, "ftol": 1e-12},
        )
        first_iterations = int(first.nit)
        if not 0 <= first_iterations <= 60:
            raise ValueError("Two-point feasibility exceeds its iteration budget")
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

    safe, prefix_safe, closest = summarize_candidates()
    if not interrupted and prefix_safe:
        phase_start = min(prefix_safe, key=lambda row: (row[3], row[1]["score"]["objective"]))
        start_x = phase_start[0].copy()

        def future_objective(y):
            difference = y[:-1] - start_x
            return float(y[-1] + 1e-9 * (difference @ difference)), np.r_[2e-9 * difference, 1.0]

        def split(y):
            values, jacobian, _ = pose(y[:-1])
            return split_prefix_constraints(values, jacobian, y[-1])

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
                np.r_[start_x, phase_start[3] + 1e-6],
                jac=True,
                method="SLSQP",
                bounds=Bounds(np.r_[lo, 0.0], np.r_[hi, np.inf]),
                constraints=future_constraints,
                options={"maxiter": remaining, "ftol": 1e-9},
            )
            second_iterations = int(second.nit)
            if not 0 <= second_iterations <= remaining:
                raise ValueError("Two stages exceed the shared iteration budget")
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

    safe, prefix_safe, closest = summarize_candidates()
    rejected = None
    if closest is not None and closest[5] > 0:
        rejected = dict(
            prefix_tcp_errors=closest[2][:2].copy(),
            prefix_normalized_violation=closest[5],
            future_quality_slack=closest[3],
            planned_commands=closest[0].reshape(8, 6).copy(),
        )
    info.update(seed_rollout_reused=True, actual_model_rollouts=len(cache))
    base.update(
        response_evaluations=len(cache),
        initialization=info,
        search_phases=phases,
        joint_safe_evaluations=len(safe),
        prefix_hard_evaluations=len(prefix_safe),
        closest_prefix_rejected=rejected,
    )
    if not prefix_safe:
        return dict(**base, status="two_step_no_witness")
    x, item, errors, future_slack, overall_slack, _ = min(
        prefix_safe, key=lambda row: (row[3], row[1]["score"]["objective"])
    )
    base["best_joint_safe"] = dict(
        witness_found=True,
        planned_commands=x.reshape(8, 6).copy(),
        terminal_normals=normals.copy(),
        rollout=item["rollout"],
        first_tcp_error=errors[0].copy(),
        tcp_errors=errors.copy(),
        quality_slack=overall_slack,
        future_quality_slack=future_slack,
        hard_pose_prefix=2,
        prefix_selection_hard=True,
        first_seen=item["first_seen"],
    )
    return dict(**base, status="two_step_candidate")
