"""Finite-budget trajectory inverse search using causal nonlinear robot rollouts."""

import numpy as np
from scipy.optimize import LinearConstraint, minimize

from s2a.geometry import finite, pose_error, validate_transform
from s2a.joint_preview_constraint_audit import audit_joint_preview_constraints


def solve_trajectory_response(
    predict,
    previous,
    warm,
    interface,
    lower,
    upper,
    desired,
    site_from_tcp,
    *,
    past_inner_error_rad,
    terminal_normals,
    max_evaluations=4000,
    max_iterations=60,
    finite_difference_step=1e-4,
):
    """Require original pose caps at every supplied future point, including H1.

    The predictor receives only an H-by-six command array and must propagate
    its own states and controller goals. Each unique full trajectory, including
    derivative probes, counts against one shared budget. Box bounds and linear
    inter-period rate limits are explicit; probes respect box bounds but may
    violate rate limits, and can never become witnesses without the full audit.
    A safety epigraph first restores inner/braking/terminal feasibility; the
    remaining iterations minimize the maximum normalized squared pose excess.
    A converged optimizer with positive excess is not a feasible action plan.
    This local numerical search supplies neither physical nor global guarantees.
    """
    previous, warm = finite(previous, (6,)), finite(warm)
    if warm.ndim != 2 or warm.shape[1] != 6 or not 1 <= len(warm) <= 16:
        raise ValueError("One to sixteen explicit six-axis warm commands required")
    h, width = len(warm), warm.size
    lower, upper = finite(lower), finite(upper)
    desired, tool = validate_transform(desired), validate_transform(site_from_tcp)
    profile, normals = finite(past_inner_error_rad), finite(terminal_normals)
    if (
        lower.ndim != 1
        or not len(lower)
        or upper.shape != lower.shape
        or np.any(lower >= upper)
        or profile.shape != (25, len(lower))
        or np.any(profile < 0)
        or np.any(np.diff(profile, axis=0) < 0)
        or normals.ndim != 2
        or normals.shape[1] != len(lower)
        or desired.shape != (h, 4, 4)
        or tool.shape != (4, 4)
        or type(max_evaluations) is not int
        or max_evaluations < width + 1
        or type(max_iterations) is not int
        or max_iterations < 2
        or not np.isfinite(finite_difference_step)
        or finite_difference_step <= 0
    ):
        raise ValueError("Valid trajectory/profile/limits and explicit search budgets required")
    lo = np.tile(finite(interface.input_min, (6,)), h)
    hi = np.tile(finite(interface.input_max, (6,)), h)
    lo[:6], hi[:6] = np.maximum(lo[:6], previous - 0.25), np.minimum(hi[:6], previous + 0.25)
    difference = np.eye(width) - np.eye(width, k=-6)
    center = np.r_[previous, np.zeros(width - 6)]
    if (
        np.any(hi - lo <= finite_difference_step)
        or np.any(warm.ravel() < lo - 1e-12)
        or np.any(warm.ravel() > hi + 1e-12)
        or np.any(np.abs(difference @ warm.ravel() - center) > 0.25 + 1e-12)
    ):
        raise ValueError("Box/rate feasible warm trajectory and nondegenerate command bounds required")
    start = np.clip(warm.ravel(), lo, hi)
    rate = LinearConstraint(np.c_[difference, np.zeros(width)], center - 0.25, center + 0.25)
    safety_buffer = 1e-8
    cache, phases = {}, []
    derivative_key, derivative_value = None, None
    safety_names = ("inner_lower", "inner_upper", "braking_lower", "braking_upper", "terminal")

    class BudgetExhausted(Exception):
        pass

    def evaluate(x):
        x = finite(x, (width,))
        key = x.tobytes()
        if key in cache:
            return cache[key]
        if len(cache) >= max_evaluations:
            raise BudgetExhausted
        if np.any(x < lo - 1e-12) or np.any(x > hi + 1e-12):
            raise ValueError("Solver attempted an out-of-box response")
        commands = x.reshape(h, 6).copy()
        rollout = predict(commands.copy())
        audit = audit_joint_preview_constraints(
            rollout,
            commands,
            previous,
            interface,
            lower,
            upper,
            past_inner_error_rad=profile,
            terminal_normals=normals,
            braking_time_s=0.1,
        )
        poses = validate_transform(rollout["pose"])
        if poses.shape != (h, 4, 4):
            raise ValueError("Every predicted endpoint requires its own FK pose")
        errors = pose_error(desired, poses @ tool)
        normalized = np.stack(
            (np.sum(errors[:, :3] ** 2, axis=1) / 0.01**2, np.sum(errors[:, 3:] ** 2, axis=1) / 0.05**2), axis=1
        )
        safety = np.concatenate([audit["groups"][name]["slack"].ravel() for name in safety_names])
        record = dict(
            commands=commands,
            rollout=rollout,
            audit=audit,
            tcp_errors=errors,
            normalized_squared_errors=normalized,
            safety_slack=safety,
            maximum_pose_cap_violation=float(max(0.0, np.max(np.sqrt(normalized)) - 1.0)),
            pose_loss=float(np.sum(normalized)),
            feasible=bool(audit["feasible"] and np.all(normalized <= 1.0)),
        )
        cache[key] = record
        return record

    def derivatives(x):
        nonlocal derivative_key, derivative_value
        key = np.asarray(x).tobytes()
        if key != derivative_key:
            base = evaluate(x)
            ds, dp = [], []
            for axis in range(width):
                delta = (
                    finite_difference_step if x[axis] + finite_difference_step <= hi[axis] else -finite_difference_step
                )
                other = np.asarray(x).copy()
                other[axis] += delta
                changed = evaluate(other)
                ds.append((changed["safety_slack"] - base["safety_slack"]) / delta)
                dp.append(
                    (changed["normalized_squared_errors"].ravel() - base["normalized_squared_errors"].ravel()) / delta
                )
            derivative_key, derivative_value = key, (np.asarray(ds).T, np.asarray(dp).T)
        return derivative_value

    def run_phase(x, phase, iterations):
        base = evaluate(x)
        initial_slack = (
            max(0.0, safety_buffer - float(np.min(base["safety_slack"])))
            if phase == "safety"
            else max(0.0, float(np.max(base["normalized_squared_errors"])) - 1.0)
        )
        x0 = np.r_[x, initial_slack + 1e-6]

        def objective(z):
            return float(z[width] + (1e-4 * evaluate(z[:width])["pose_loss"] if phase == "pose" else 0.0))

        def gradient(z):
            return np.r_[1e-4 * derivatives(z[:width])[1].sum(axis=0) if phase == "pose" else np.zeros(width), 1.0]

        def constraint(z):
            entry = evaluate(z[:width])
            safety = entry["safety_slack"] - safety_buffer
            return (
                safety + z[width]
                if phase == "safety"
                else np.r_[safety, 1.0 + z[width] - entry["normalized_squared_errors"].ravel()]
            )

        def jacobian(z):
            ds, dp = derivatives(z[:width])
            return (
                np.c_[ds, np.ones(len(ds))]
                if phase == "safety"
                else np.vstack((np.c_[ds, np.zeros(len(ds))], np.c_[-dp, np.ones(2 * h)]))
            )

        record = dict(
            phase=phase,
            max_iterations=iterations,
            initial_commands=x.reshape(h, 6).copy(),
            initial_epigraph=float(x0[width]),
            initial_safety_min=float(np.min(base["safety_slack"])),
            evaluations_before=len(cache),
            iterations=0,
        )
        phases.append(record)

        def callback(_):
            record["iterations"] += 1

        try:
            answer = minimize(
                objective,
                x0,
                jac=gradient,
                method="SLSQP",
                bounds=[*zip(lo, hi, strict=True), (0.0, None)],
                constraints=[rate, dict(type="ineq", fun=constraint, jac=jacobian)],
                callback=callback,
                options=dict(maxiter=iterations, ftol=1e-12, disp=False),
            )
            record.update(
                success=bool(answer.success),
                status=int(answer.status),
                message=str(answer.message),
                iterations=int(answer.nit),
                final_commands=answer.x[:width].reshape(h, 6),
                final_epigraph=float(answer.x[width]),
            )
            evaluate(answer.x[:width])
        except BudgetExhausted:
            record.update(success=False, status="response_budget", message="Shared full-trajectory budget exhausted")
        record["evaluations_after"] = len(cache)

    initial = evaluate(start)
    if np.min(initial["safety_slack"]) < safety_buffer:
        run_phase(start, "safety", min(20, max(1, max_iterations // 3)))

    def rank(r):
        return r["maximum_pose_cap_violation"], r["pose_loss"]

    safe = [r for r in cache.values() if r["audit"]["feasible"]]
    used = sum(p["iterations"] for p in phases)
    if safe and len(cache) < max_evaluations and used < max_iterations:
        run_phase(min(safe, key=rank)["commands"].ravel(), "pose", max_iterations - used)
    safe = [r for r in cache.values() if r["audit"]["feasible"]]
    witnesses = [r for r in safe if r["feasible"]]
    selected = min(witnesses or safe, key=rank) if safe else None

    def safety_rank(r):
        worst = min(g["minimum"] for g in r["audit"]["groups"].values() if g["minimum"] is not None)
        return (max(0.0, -worst), r["pose_loss"])

    return dict(
        witness_found=bool(witnesses),
        selected=selected,
        closest_safety_attempt=min(cache.values(), key=safety_rank),
        initial=initial,
        supplied_warm=warm.copy(),
        used_start=start.reshape(h, 6),
        start_roundoff_clip=start.reshape(h, 6) - warm,
        lower_command=lo.reshape(h, 6),
        upper_command=hi.reshape(h, 6),
        horizon=h,
        pose_caps_horizon=h,
        all_pose_caps_enforced=True,
        terminal_normals=normals.copy(),
        phases=phases,
        response_evaluations=len(cache),
        total_iterations=sum(p["iterations"] for p in phases),
        max_evaluations=max_evaluations,
        max_iterations=max_iterations,
        finite_difference_step=finite_difference_step,
        optimization_safety_buffer=safety_buffer,
        past_inner_error_rad=profile.copy(),
        evaluated_candidates=[
            dict(
                commands=r["commands"],
                safety_min=float(np.min(r["safety_slack"])),
                joint_feasible=r["audit"]["feasible"],
                tcp_errors=r["tcp_errors"],
                feasible=r["feasible"],
            )
            for r in cache.values()
        ],
        status="witness" if witnesses else "finite_search_without_witness",
    )
