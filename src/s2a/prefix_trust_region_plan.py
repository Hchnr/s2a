"""Finite prefix repair with explicit, fully audited trust-region anchors."""

import numpy as np
from scipy.optimize import LinearConstraint, minimize

from s2a.geometry import finite, pose_error, validate_transform
from s2a.joint_preview_constraint_audit import audit_joint_preview_constraints

RADII = (0.01, 0.005, 0.0025)


def solve_prefix_trust_region_response(
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
    max_stages=60,
):
    """Recenter only after complete nonlinear prefix/safety/descent acceptance.

    Mathematical responses, audit tolerances, pose caps, epigraph objective,
    rate constraint, and final candidate rankings match the frozen prefix
    solver. Each stage has at most five iterations and its own derivative
    cache. The exact-byte response cache and all budgets span every stage.

    Every evaluate request is recorded, including cache hits and denied new
    responses. A complete witness may stop at a callback or survive a later
    budget denial; it needs the original audit/caps, not the extra buffer.
    Ordinary prefix-preserving descent can recenter without increasing the
    prefix. Final selection still uses all safe responses, so it need not be
    an accepted anchor. This is a finite model search, not physical control.
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
        or type(max_stages) is not int
        or max_stages < 1
        or not np.isfinite(finite_difference_step)
        or finite_difference_step <= 0
        or finite_difference_step >= RADII[-1]
    ):
        raise ValueError(
            "Valid trajectory/profile/limits, stage-compatible difference step and explicit budgets required"
        )
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
    cache, cache_indices, phases, ledger, accepted_anchors = {}, {}, [], [], []
    active_stage = None
    safety_names = ("inner_lower", "inner_upper", "braking_lower", "braking_upper", "terminal")

    class BudgetExhausted(Exception):
        pass

    class WitnessFound(Exception):
        pass

    def evaluate(x, role, *, axis=None, nominal_delta=None):
        x = finite(x, (width,))
        key = x.tobytes()
        request = dict(
            request_index=len(ledger),
            stage_index=active_stage["stage_index"] if active_stage is not None else None,
            role=role,
            command_key=key.hex(),
            response_index=None,
            new_response_index=None,
            outcome="pending",
            response_evaluations_before=len(cache),
            response_evaluations_after=len(cache),
            axis=axis,
            nominal_delta=nominal_delta,
        )
        ledger.append(request)
        bound_lo = lo if active_stage is None else active_stage["stage_lower_command"].ravel()
        bound_hi = hi if active_stage is None else active_stage["stage_upper_command"].ravel()
        if np.any(x < bound_lo - 1e-12) or np.any(x > bound_hi + 1e-12):
            request["outcome"] = "bounds_rejected"
            raise ValueError("Solver attempted a response outside its current command bounds")
        if key in cache:
            request.update(outcome="cache_hit", response_index=cache_indices[key])
        else:
            if len(cache) >= max_evaluations:
                request["outcome"] = "budget_rejected"
                raise BudgetExhausted
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
            normalized = finite(
                np.stack(
                    (np.sum(errors[:, :3] ** 2, axis=1) / 0.01**2, np.sum(errors[:, 3:] ** 2, axis=1) / 0.05**2),
                    axis=1,
                )
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
            index = len(cache)
            cache_indices[key], cache[key] = index, record
            request.update(
                outcome="new_response",
                response_index=index,
                new_response_index=index,
                response_evaluations_after=len(cache),
            )
        if active_stage is not None:
            active_stage["touched_response_indices"].append(request["response_index"])
        return cache[key]

    def witness_rank(record):
        return record["maximum_normalized_squared_error"], record["pose_loss"]

    def prefix_rank(record):
        prefix = record["prefix_length"]
        next_error = float(np.max(record["normalized_squared_errors"][prefix])) if prefix < h else 0.0
        return -prefix, next_error, record["pose_loss"]

    def select(records):
        witnesses = [record for record in records if record["feasible"]]
        return min(witnesses, key=witness_rank) if witnesses else min(records, key=prefix_rank)

    def index_of(record):
        return cache_indices[record["commands"].ravel().tobytes()]

    def anchor_record(record, before, stage=None):
        cursor = (
            stage["target_cursor"] if stage is not None else (None if record["feasible"] else record["prefix_length"])
        )
        before_max = float(np.max(before["normalized_squared_errors"][cursor])) if cursor is not None else None
        after_max = float(np.max(record["normalized_squared_errors"][cursor])) if cursor is not None else None
        return dict(
            is_initial=stage is None,
            stage_index=stage["stage_index"] if stage is not None else None,
            response_index=index_of(record),
            previous_response_index=index_of(before) if stage is not None else None,
            commands=record["commands"].copy(),
            prefix_length=record["prefix_length"],
            joint_feasible=record["audit"]["feasible"],
            safety_min=float(np.min(record["safety_slack"])),
            feasible=record["feasible"],
            audit=record["audit"],
            tcp_errors=record["tcp_errors"],
            normalized_squared_errors=record["normalized_squared_errors"],
            stage_target_cursor=cursor,
            stage_target_max_before=before_max,
            stage_target_max_after=after_max,
            target_improvement=before_max - after_max if cursor is not None else None,
            radius=stage["radius"] if stage is not None else None,
            acceptance_reason=stage["acceptance_reason"]
            if stage is not None
            else ("initial_witness" if record["feasible"] else "warm"),
        )

    def run_phase(base, radius_index, iterations):
        nonlocal active_stage
        cursor, radius = base["prefix_length"], RADII[radius_index]
        x = base["commands"].ravel()
        stage_lo, stage_hi = np.maximum(lo, x - radius), np.minimum(hi, x + radius)
        if np.any(stage_hi - stage_lo <= finite_difference_step):
            raise ValueError("Every stage requires a nondegenerate finite-difference interval")
        initial_slack = max(0.0, float(np.max(base["normalized_squared_errors"][cursor])) - 1.0)
        x0 = np.r_[x, initial_slack + 1e-6]
        record = dict(
            phase="prefix_trust",
            stage_index=len(phases),
            prefix_before=cursor,
            target_cursor=cursor,
            cursor=cursor,
            radius_index=radius_index,
            radius=radius,
            anchor_before_index=index_of(base),
            stage_lower_command=stage_lo.reshape(h, 6).copy(),
            stage_upper_command=stage_hi.reshape(h, 6).copy(),
            max_iterations=iterations,
            initial_commands=base["commands"].copy(),
            initial_epigraph=float(x0[width]),
            initial_safety_min=float(np.min(base["safety_slack"])),
            evaluations_before=len(cache),
            response_budget_remaining_before=max_evaluations - len(cache),
            requests_before=len(ledger),
            iterations=0,
            callback_count=0,
            optimizer_returned=False,
            touched_response_indices=[],
        )
        phases.append(record)
        active_stage = record
        # Deliberately local: identical x in another stage may use another FD sign.
        derivative_key, derivative_value = None, None

        def derivatives(point, role):
            nonlocal derivative_key, derivative_value
            key = np.asarray(point).tobytes()
            if key != derivative_key:
                base_response = evaluate(point, role + "_base")
                ds, dp = [], []
                for axis in range(width):
                    delta = (
                        finite_difference_step
                        if point[axis] + finite_difference_step <= stage_hi[axis]
                        else -finite_difference_step
                    )
                    other = np.asarray(point).copy()
                    other[axis] += delta
                    changed = evaluate(other, role + "_probe", axis=axis, nominal_delta=delta)
                    ds.append((changed["safety_slack"] - base_response["safety_slack"]) / delta)
                    dp.append(
                        (
                            changed["normalized_squared_errors"].ravel()
                            - base_response["normalized_squared_errors"].ravel()
                        )
                        / delta
                    )
                derivative_key, derivative_value = key, (np.asarray(ds).T, np.asarray(dp).T)
            return derivative_value

        def objective(z):
            return float(z[width] + 1e-4 * evaluate(z[:width], "objective")["pose_loss"])

        def gradient(z):
            return np.r_[1e-4 * derivatives(z[:width], "gradient")[1].sum(axis=0), 1.0]

        def constraint(z):
            entry = evaluate(z[:width], "constraint")
            normalized = entry["normalized_squared_errors"]
            return np.r_[
                entry["safety_slack"] - safety_buffer,
                1.0 - normalized[:cursor].ravel(),
                1.0 + z[width] - normalized[cursor],
            ]

        def jacobian(z):
            ds, dp = derivatives(z[:width], "jacobian")
            return np.vstack(
                (
                    np.c_[ds, np.zeros(len(ds))],
                    np.c_[-dp[: 2 * cursor], np.zeros(2 * cursor)],
                    np.c_[-dp[2 * cursor : 2 * cursor + 2], np.ones(2)],
                )
            )

        def callback(z):
            record["callback_count"] += 1
            record["iterations"] = record["callback_count"]
            record["last_callback_commands"] = np.asarray(z[:width]).reshape(h, 6).copy()
            record["last_callback_epigraph"] = float(z[width])
            evaluate(z[:width], "callback")
            values = list(cache.values())
            if any(values[index]["feasible"] for index in record["touched_response_indices"]):
                raise WitnessFound

        try:
            evaluate(x, "stage_anchor")
            answer = minimize(
                objective,
                x0,
                jac=gradient,
                method="SLSQP",
                bounds=[*zip(stage_lo, stage_hi, strict=True), (0.0, None)],
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
            evaluate(answer.x[:width], "final")
        except WitnessFound:
            record.update(
                success=False,
                status="witness_found",
                message="A fully audited full-horizon witness reached a callback boundary",
            )
        except BudgetExhausted:
            record.update(success=False, status="response_budget", message="Shared full-trajectory budget exhausted")
        if not 0 <= record["iterations"] <= iterations:
            raise ValueError("Optimizer iteration accounting exceeded the explicit stage budget")
        record["touched_response_indices"] = sorted(set(record["touched_response_indices"]))
        values = list(cache.values())
        touched = [values[index] for index in record["touched_response_indices"]]
        eligible, decisions = [], []
        base_target = float(np.max(base["normalized_squared_errors"][cursor]))
        for candidate in touched:
            reasons = []
            if not candidate["feasible"]:
                if not candidate["audit"]["feasible"]:
                    reasons.append("joint_or_command")
                if candidate["prefix_length"] < cursor:
                    reasons.append("prefix_lost")
                if float(np.min(candidate["safety_slack"])) < safety_buffer:
                    reasons.append("safety_buffer")
                change = base_target - float(np.max(candidate["normalized_squared_errors"][cursor]))
                if not (candidate["prefix_length"] > cursor or change > 1e-12):
                    reasons.append("no_target_descent")
            if not reasons:
                eligible.append(candidate)
            decisions.append(
                dict(
                    response_index=index_of(candidate), eligible=not reasons, selected=False, rejection_reasons=reasons
                )
            )
        chosen = select(eligible) if eligible else None
        next_base = chosen if chosen is not None else base
        chosen_index = index_of(chosen) if chosen is not None else None
        for decision in decisions:
            decision["selected"] = decision["response_index"] == chosen_index
            if decision["eligible"] and not decision["selected"]:
                decision["rejection_reasons"] = ["not_selected_by_rank"]
        reason = (
            None
            if chosen is None
            else (
                "witness"
                if chosen["feasible"]
                else "prefix_growth"
                if chosen["prefix_length"] > cursor
                else "target_descent"
            )
        )
        record.update(
            evaluations_after=len(cache),
            response_budget_remaining_after=max_evaluations - len(cache),
            requests_after=len(ledger),
            prefix_after=next_base["prefix_length"],
            selected_commands=next_base["commands"].copy(),
            anchor_after_index=index_of(next_base),
            accepted=chosen is not None,
            acceptance_reason=reason,
            radius_after=RADII[0]
            if chosen is not None
            else (RADII[radius_index + 1] if radius_index + 1 < len(RADII) else None),
            candidate_decisions=decisions,
            rejected_candidates=[item for item in decisions if not item["selected"]],
        )
        if chosen is not None:
            accepted_anchors.append(anchor_record(chosen, base, record))
        active_stage = None
        return next_base, chosen is not None

    initial = evaluate(start, "warm")
    if not initial["feasible"] and (
        not initial["audit"]["feasible"] or np.min(initial["safety_slack"]) < safety_buffer
    ):
        raise ValueError("Incomplete warm trajectory requires the full joint/command audit and1e-8 safety buffer")
    accepted_anchors.append(anchor_record(initial, initial))
    current, radius_index = initial, 0
    termination_reason = "witness" if initial["feasible"] else None
    while not current["feasible"]:
        used = sum(phase["iterations"] for phase in phases)
        if len(cache) >= max_evaluations:
            termination_reason = "response_budget"
            break
        if used >= max_iterations:
            termination_reason = "iteration_budget"
            break
        if len(phases) >= max_stages:
            termination_reason = "stage_budget"
            break
        if radius_index >= len(RADII):
            termination_reason = "trust_region_stalled"
            break
        current, accepted = run_phase(current, radius_index, min(5, max_iterations - used))
        radius_index = 0 if accepted else radius_index + 1
    safe = [record for record in cache.values() if record["audit"]["feasible"]]
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
        request_ledger=ledger,
        accepted_anchors=accepted_anchors,
        max_stages=max_stages,
        total_stages=len(phases),
    )
