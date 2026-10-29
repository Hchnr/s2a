"""Repair a complete nonlinear command plan while preserving attained pose caps."""

import numpy as np
from scipy.optimize import LinearConstraint, minimize

from s2a.geometry import finite, pose_error, validate_transform
from s2a.joint_preview_constraint_audit import audit_joint_preview_constraints


def solve_prefix_preserving_trajectory_response(
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
    """Keep attained prefix caps hard while repairing the next failing point.

    All H-by-six commands remain free. A stage's single nonnegative epigraph
    relaxes only its target cursor's two pose caps; previous caps, every inner
    joint constraint, braking, terminal velocity and command rate remain hard.
    Stages change only at completed optimizer callback boundaries or after an
    optimizer returns. They share command-byte response and derivative caches,
    a response budget and an iteration budget; a stalled stage is not restarted.

    The warm trajectory must already satisfy the full joint/command audit and
    the 1e-8 nonlinear optimization buffer unless it is a complete witness, in
    which case one response is sufficient. A longer prefix alone never permits
    dispatch: all H original pose caps and the full audit define a witness.
    This finite local model search is neither a physical safety qualification
    nor a certificate of global infeasibility.
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

    class PrefixAdvanced(Exception):
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
        failures = np.flatnonzero(~np.all(normalized <= 1.0, axis=1))
        prefix = int(failures[0]) if len(failures) else h
        safety = np.concatenate([audit["groups"][name]["slack"].ravel() for name in safety_names])
        record = dict(
            commands=commands,
            rollout=rollout,
            audit=audit,
            tcp_errors=errors,
            normalized_squared_errors=normalized,
            safety_slack=safety,
            prefix_length=prefix,
            maximum_normalized_squared_error=float(np.max(normalized)),
            maximum_pose_cap_violation=float(max(0.0, np.max(np.sqrt(normalized)) - 1.0)),
            pose_loss=float(np.sum(normalized)),
            feasible=bool(audit["feasible"] and prefix == h),
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
            # These full-window derivatives do not depend on the active prefix.
            derivative_key, derivative_value = key, (np.asarray(ds).T, np.asarray(dp).T)
        return derivative_value

    def witness_rank(record):
        return record["maximum_normalized_squared_error"], record["pose_loss"]

    def prefix_rank(record):
        prefix = record["prefix_length"]
        next_error = float(np.max(record["normalized_squared_errors"][prefix])) if prefix < h else 0.0
        return -prefix, next_error, record["pose_loss"]

    def select(records):
        witnesses = [record for record in records if record["feasible"]]
        # Dict insertion order makes exact ties select the first evaluation.
        return min(witnesses, key=witness_rank) if witnesses else min(records, key=prefix_rank)

    def safe_candidates(minimum_prefix=0):
        return [
            record
            for record in cache.values()
            if record["audit"]["feasible"] and record["prefix_length"] >= minimum_prefix
        ]

    def run_phase(base, iterations):
        cursor = base["prefix_length"]
        x = base["commands"].ravel()
        initial_slack = max(0.0, float(np.max(base["normalized_squared_errors"][cursor])) - 1.0)
        x0 = np.r_[x, initial_slack + 1e-6]

        def objective(z):
            return float(z[width] + 1e-4 * evaluate(z[:width])["pose_loss"])

        def gradient(z):
            return np.r_[1e-4 * derivatives(z[:width])[1].sum(axis=0), 1.0]

        def constraint(z):
            entry = evaluate(z[:width])
            normalized = entry["normalized_squared_errors"]
            return np.r_[
                entry["safety_slack"] - safety_buffer,
                1.0 - normalized[:cursor].ravel(),
                1.0 + z[width] - normalized[cursor],
            ]

        def jacobian(z):
            ds, dp = derivatives(z[:width])
            return np.vstack(
                (
                    np.c_[ds, np.zeros(len(ds))],
                    np.c_[-dp[: 2 * cursor], np.zeros(2 * cursor)],
                    np.c_[-dp[2 * cursor : 2 * cursor + 2], np.ones(2)],
                )
            )

        record = dict(
            phase="prefix",
            prefix_before=cursor,
            target_cursor=cursor,
            cursor=cursor,
            max_iterations=iterations,
            initial_commands=base["commands"].copy(),
            initial_epigraph=float(x0[width]),
            initial_safety_min=float(np.min(base["safety_slack"])),
            evaluations_before=len(cache),
            response_budget_remaining_before=max_evaluations - len(cache),
            iterations=0,
            callback_count=0,
            optimizer_returned=False,
        )
        phases.append(record)

        def callback(z):
            record["callback_count"] += 1
            record["iterations"] = record["callback_count"]
            record["last_callback_commands"] = np.asarray(z[:width]).reshape(h, 6).copy()
            record["last_callback_epigraph"] = float(z[width])
            if safe_candidates(cursor + 1):
                raise PrefixAdvanced

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
                optimizer_returned=True,
                optimizer_iterations=int(answer.nit),
                iterations=max(record["callback_count"], int(answer.nit)),
                final_commands=answer.x[:width].reshape(h, 6).copy(),
                final_epigraph=float(answer.x[width]),
            )
            evaluate(answer.x[:width])
        except PrefixAdvanced:
            record.update(
                success=False,
                status="prefix_advanced",
                message="A fully audited cached trajectory extended the prefix at a callback boundary",
            )
        except BudgetExhausted:
            record.update(success=False, status="response_budget", message="Shared full-trajectory budget exhausted")
        next_base = select(safe_candidates(cursor))
        record.update(
            evaluations_after=len(cache),
            response_budget_remaining_after=max_evaluations - len(cache),
            prefix_after=next_base["prefix_length"],
            selected_commands=next_base["commands"].copy(),
        )
        return next_base

    initial = evaluate(start)
    if not initial["feasible"] and (
        not initial["audit"]["feasible"] or np.min(initial["safety_slack"]) < safety_buffer
    ):
        raise ValueError("Incomplete warm trajectory requires the full joint/command audit and 1e-8 safety buffer")
    current = initial
    termination_reason = "witness" if initial["feasible"] else None
    while not current["feasible"]:
        used = sum(phase["iterations"] for phase in phases)
        if len(cache) >= max_evaluations:
            termination_reason = "response_budget"
            break
        if used >= max_iterations:
            termination_reason = "iteration_budget"
            break
        next_base = run_phase(current, max_iterations - used)
        if next_base["feasible"]:
            current, termination_reason = next_base, "witness"
            break
        if len(cache) >= max_evaluations:
            termination_reason = "response_budget"
            break
        if next_base["prefix_length"] <= current["prefix_length"]:
            termination_reason = "no_prefix_progress"
            break
        current = next_base

    safe = safe_candidates()
    selected = select(safe)
    witnesses = [record for record in safe if record["feasible"]]

    def safety_rank(record):
        worst = min(group["minimum"] for group in record["audit"]["groups"].values() if group["minimum"] is not None)
        return max(0.0, -worst), record["pose_loss"]

    return dict(
        witness_found=bool(witnesses),
        selected=selected,
        closest_safety_attempt=min(cache.values(), key=safety_rank),
        initial=initial,
        initial_prefix_length=initial["prefix_length"],
        selected_prefix_length=selected["prefix_length"],
        supplied_warm=warm.copy(),
        used_start=start.reshape(h, 6),
        start_roundoff_clip=start.reshape(h, 6) - warm,
        lower_command=lo.reshape(h, 6),
        upper_command=hi.reshape(h, 6),
        horizon=h,
        pose_caps_horizon=h,
        all_pose_caps_enforced=True,
        pose_cap_enforcement_scope="All H caps are required for a witness; stages hard-constrain only the attained prefix",
        terminal_normals=normals.copy(),
        phases=phases,
        response_evaluations=len(cache),
        total_iterations=sum(phase["iterations"] for phase in phases),
        max_evaluations=max_evaluations,
        max_iterations=max_iterations,
        finite_difference_step=finite_difference_step,
        optimization_safety_buffer=safety_buffer,
        past_inner_error_rad=profile.copy(),
        evaluated_candidates=[
            dict(
                commands=record["commands"],
                safety_min=float(np.min(record["safety_slack"])),
                joint_feasible=record["audit"]["feasible"],
                tcp_errors=record["tcp_errors"],
                prefix_length=record["prefix_length"],
                feasible=record["feasible"],
            )
            for record in cache.values()
        ],
        termination_reason="witness" if witnesses else termination_reason,
        status="witness" if witnesses else "finite_search_without_witness",
    )
