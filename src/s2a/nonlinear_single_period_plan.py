"""Bounded one-period inverse search using an explicit nonlinear robot response."""

import numpy as np
from scipy.optimize import minimize

from s2a.geometry import finite, pose_error, validate_transform
from s2a.joint_preview_constraint_audit import audit_joint_preview_constraints


def solve_single_period_response(
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
    max_evaluations=600,
    max_iterations=60,
    finite_difference_step=1e-4,
):
    """Search six arm commands; scalar epigraphs only measure infeasibility.

    ``predict(command)`` owns one causal 25x2ms rollout and returns the native
    response fields. Phase1 (at most20 iterations) restores joint safety when
    needed; phase2 minimizes the larger squared normalized pose-cap violation,
    preserving safety. Both phases share the total response/iteration budget.
    Finite differences stay inside the command box/rate intersection. Every
    returned witness is selected from independently audited cached responses,
    never from the optimizer's success flag. This is no global infeasibility
    certificate and does not establish actual control safety.
    """
    previous, warm = finite(previous, (6,)), finite(warm, (6,))
    lower, upper = finite(lower), finite(upper)
    desired, site_from_tcp = validate_transform(desired), validate_transform(site_from_tcp)
    profile = finite(past_inner_error_rad)
    if (
        lower.ndim != 1
        or upper.shape != lower.shape
        or profile.shape != (25, len(lower))
        or np.any(lower >= upper)
        or np.any(profile < 0)
        or np.any(np.diff(profile, axis=0) < 0)
        or desired.shape != (4, 4)
        or site_from_tcp.shape != (4, 4)
        or type(max_evaluations) is not int
        or max_evaluations < 7
        or type(max_iterations) is not int
        or max_iterations < 2
        or not np.isfinite(finite_difference_step)
        or finite_difference_step <= 0
    ):
        raise ValueError("Valid joint/profile/pose data and positive explicit search budgets required")
    lo = np.maximum(finite(interface.input_min, (6,)), previous - 0.25)
    hi = np.minimum(finite(interface.input_max, (6,)), previous + 0.25)
    if np.any(hi - lo <= finite_difference_step) or np.any(warm < lo - 1e-12) or np.any(warm > hi + 1e-12):
        raise ValueError("Warm command within the original box/rate tolerance and nondegenerate search box required")
    start = np.clip(warm, lo, hi)
    cache, derivative_cache, phases = {}, {}, []
    safety_buffer = 1e-8
    n = len(lower)

    class BudgetExhausted(Exception):
        pass

    def evaluate(command):
        command = finite(command, (6,))
        key = command.tobytes()
        if key in cache:
            return cache[key]
        if len(cache) >= max_evaluations:
            raise BudgetExhausted
        if np.any(command < lo - 1e-12) or np.any(command > hi + 1e-12):
            raise ValueError("Solver attempted a command outside registered bounds")
        response = predict(command.copy())
        q = finite(response["joint_position"], (25, n))
        velocity = finite(response["joint_velocity"], (25, n))
        audit = audit_joint_preview_constraints(
            dict(inner_joint_position=q, joint_position=q[-1:], joint_velocity=velocity[-1:]),
            command[None, :],
            previous,
            interface,
            lower,
            upper,
            past_inner_error_rad=profile,
            terminal_normals=np.empty((0, n)),
            braking_time_s=0.1,
        )
        safety = np.concatenate(
            [
                audit["groups"][name]["slack"].ravel()
                for name in ("inner_lower", "inner_upper", "braking_lower", "braking_upper")
            ]
        )
        tcp = validate_transform(response["endpoint_pose"]) @ site_from_tcp
        error = pose_error(desired, tcp)
        normalized = np.array([np.sum(error[:3] ** 2) / 0.01**2, np.sum(error[3:] ** 2) / 0.05**2])
        entry = dict(
            command=command.copy(),
            response=response,
            audit=audit,
            tcp_error=error,
            normalized_squared_errors=normalized,
            safety_slack=safety,
            maximum_pose_cap_violation=float(max(0.0, np.max(np.sqrt(normalized)) - 1.0)),
            pose_loss=float(np.sum(normalized)),
            feasible=bool(audit["feasible"] and np.all(normalized <= 1.0)),
        )
        cache[key] = entry
        return entry

    def derivatives(command):
        key = np.asarray(command).tobytes()
        if key not in derivative_cache:
            base = evaluate(command)
            ds, dp = [], []
            for axis in range(6):
                delta = (
                    finite_difference_step
                    if command[axis] + finite_difference_step <= hi[axis]
                    else -finite_difference_step
                )
                other = np.asarray(command).copy()
                other[axis] += delta
                changed = evaluate(other)
                ds.append((changed["safety_slack"] - base["safety_slack"]) / delta)
                dp.append((changed["normalized_squared_errors"] - base["normalized_squared_errors"]) / delta)
            derivative_cache[key] = (np.asarray(ds).T, np.asarray(dp).T)
        return derivative_cache[key]

    def run_phase(command, phase, iterations):
        base = evaluate(command)
        initial_slack = (
            max(0.0, safety_buffer - float(np.min(base["safety_slack"])))
            if phase == "safety"
            else max(0.0, float(np.max(base["normalized_squared_errors"])) - 1.0)
        )
        x0 = np.r_[command, initial_slack + 1e-6]

        def objective(x):
            return float(x[6] + (1e-4 * evaluate(x[:6])["pose_loss"] if phase == "pose" else 0.0))

        def gradient(x):
            return np.r_[1e-4 * derivatives(x[:6])[1].sum(axis=0) if phase == "pose" else np.zeros(6), 1.0]

        def constraint(x):
            record = evaluate(x[:6])
            safety = record["safety_slack"] - safety_buffer
            return (
                safety + x[6] if phase == "safety" else np.r_[safety, 1.0 + x[6] - record["normalized_squared_errors"]]
            )

        def jacobian(x):
            ds, dp = derivatives(x[:6])
            return (
                np.c_[ds, np.ones(len(ds))]
                if phase == "safety"
                else np.vstack((np.c_[ds, np.zeros(len(ds))], np.c_[-dp, np.ones(2)]))
            )

        record = dict(
            phase=phase,
            max_iterations=iterations,
            initial_command=command.copy(),
            initial_epigraph=float(x0[6]),
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
                constraints=dict(type="ineq", fun=constraint, jac=jacobian),
                callback=callback,
                options=dict(maxiter=iterations, ftol=1e-12, disp=False),
            )
            record.update(
                success=bool(answer.success),
                status=int(answer.status),
                message=str(answer.message),
                iterations=int(answer.nit),
                final_command=answer.x[:6],
                final_epigraph=float(answer.x[6]),
            )
            evaluate(answer.x[:6])
        except BudgetExhausted:
            record.update(success=False, status="response_budget", message="Shared response budget exhausted")
        record["evaluations_after"] = len(cache)

    initial = evaluate(start)
    if np.min(initial["safety_slack"]) < safety_buffer:
        run_phase(start, "safety", min(20, max(1, max_iterations // 3)))
    safe = [entry for entry in cache.values() if entry["audit"]["feasible"]]
    used_iterations = sum(phase["iterations"] for phase in phases)
    if safe and len(cache) < max_evaluations and used_iterations < max_iterations:
        selected = min(safe, key=lambda r: (r["maximum_pose_cap_violation"], r["pose_loss"]))
        run_phase(selected["command"], "pose", max_iterations - used_iterations)
    safe = [entry for entry in cache.values() if entry["audit"]["feasible"]]
    witnesses = [entry for entry in safe if entry["feasible"]]
    selected = min(witnesses or safe, key=lambda r: (r["maximum_pose_cap_violation"], r["pose_loss"])) if safe else None
    closest = min(cache.values(), key=lambda r: (max(0.0, -float(np.min(r["safety_slack"]))), r["pose_loss"]))
    return dict(
        witness_found=bool(witnesses),
        selected=selected,
        closest_safety_attempt=closest,
        initial=initial,
        supplied_warm=warm.copy(),
        used_start=start,
        start_roundoff_clip=start - warm,
        lower_command=lo,
        upper_command=hi,
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
                command=r["command"],
                safety_min=float(np.min(r["safety_slack"])),
                joint_feasible=r["audit"]["feasible"],
                tcp_error=r["tcp_error"],
                feasible=r["feasible"],
            )
            for r in cache.values()
        ],
        status="witness" if witnesses else "finite_search_without_witness",
    )
