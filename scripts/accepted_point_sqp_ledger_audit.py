"""Audit saved accepted-point SQP algebra and control-flow bookkeeping.

The caller independently reconstructs every trajectory and strict classification.
Differentiation here uses the saved, checked response values: differentiating a
different floating-point replay would not reproduce the algorithm's local model.
No producer, production optimizer, QP solver, or dynamics evaluator is imported.
Full ray/anchor rollout and audit arrays remain the numerical checker's duty.
"""

import numpy as np
from accepted_point_sqp_math_audit import audit_curvature_update, audit_qp_certificate
from check_tracking_v5_prefix_trust_control import (
    TRACE_FIELDS,
    command_key,
    compare_contract,
    numerical_compare,
    trace_ranks,
)
from prefix_callback_backtracking_audit import (
    ALPHAS,
    RADII,
    STEP,
    _audit_requests,
    _ordinary_rejections,
    integer,
    require,
)


def _saved_trace_state(answer, classifications, compare):
    """Preserve the frozen strict classification checks, plus full safety data."""
    trace = answer["evaluated_candidates"]
    require(len(trace) == len(classifications) >= 1, "A classification per unique response")
    safety = []
    projected = []
    for saved, metrics in zip(trace, classifications, strict=True):
        require(set(saved) == TRACE_FIELDS | {"safety_slack"}, "Complete SQP trace fields")
        values = np.asarray(saved["safety_slack"], dtype=float)
        require(
            values.ndim == 1 and values.size > 0 and np.isfinite(values).all(),
            "Finite complete saved nonlinear safety vector",
        )
        require(not safety or values.shape == safety[0].shape, "Unchanged safety row order and dimension")
        require(float(np.min(values)) == saved["safety_min"], "Safety minimum is from the full saved vector")
        compare("trace_safety_slack", metrics["safety_slack"], values)
        safety.append(values)
        projected.append({name: saved[name] for name in TRACE_FIELDS})
    # Do not alter globals or the archived answer when adapting the old schema.
    legacy = dict(answer, evaluated_candidates=projected)
    squared, keys, prefixes, best = trace_ranks(legacy, classifications, compare=compare)
    return squared, keys, prefixes, best, safety


def _exact_array(actual, saved, message):
    actual, saved = np.asarray(actual), np.asarray(saved)
    require(actual.shape == saved.shape and np.array_equal(actual, saved), message)


def _audit_generation(answer, phase, current, squared, safety, keys, B, pending, compare):
    """Rebuild one complete or budget-truncated FD group and its local QP."""
    trace, requests = answer["evaluated_candidates"], answer["request_ledger"]
    horizon, width = answer["horizon"], 6 * answer["horizon"]
    start, end = phase["requests_before"], phase["generation_requests_after"]
    rows = requests[start:end]
    require(len(rows) >= 3, "Stage anchor, cached FD base, and at least one probe")
    require(
        [row["role"] for row in rows[:2]] == ["stage_anchor", "gradient_base"]
        and all(row["role"] == "gradient_probe" for row in rows[2:])
        and all(row["response_index"] == current and row["outcome"] == "cache_hit" for row in rows[:2]),
        "Exactly one anchor-based finite difference group generates the direction",
    )
    probes = rows[2:]
    completed = [row for row in probes if row["response_index"] is not None]
    complete = len(completed) == width
    require(
        len(probes) == width if complete else len(probes) == len(completed) + 1 <= width,
        "One response or final budget denial per ordered derivative axis",
    )
    require(
        complete or probes[-1]["outcome"] == "budget_rejected",
        "Incomplete derivative is explicitly budget denied",
    )
    expected = dict(
        complete=complete,
        base_response_index=current,
        probe_response_indices=[row["response_index"] for row in completed],
        nominal_deltas=[row["nominal_delta"] for row in completed],
    )
    _exact_array(B, phase["hessian_before"], "Curvature enters from the preceding stage exactly")
    require(
        phase["pending_pair_source_stage"] == (None if pending is None else pending["stage_index"]),
        "Pending pair belongs to the last unconsumed accepted ray",
    )
    cursor = phase["target_cursor"]
    x = np.asarray(trace[current]["commands"], dtype=float).ravel()
    qp_check = curvature_check = None
    g = nonlinear = None
    available = requests[end - 1]["response_evaluations_after"]
    witness_available = any(row["feasible"] for row in trace[:available])
    if complete:
        Js = np.asarray(
            [(safety[row["response_index"]] - safety[current]) / row["nominal_delta"] for row in completed]
        ).T
        Jq = np.asarray(
            [
                (squared[row["response_index"]].ravel() - squared[current].ravel()) / row["nominal_delta"]
                for row in completed
            ]
        ).T
        require(np.isfinite(Js).all() and np.isfinite(Jq).all(), "Finite archived-value Jacobians")
        g = 1e-4 * np.sum(Jq, axis=0)
        nonlinear = np.vstack(
            (
                np.column_stack((Js, np.zeros(len(Js)))),
                np.column_stack((-Jq[: 2 * cursor], np.zeros(2 * cursor))),
                np.column_stack((-Jq[2 * cursor : 2 * cursor + 2], np.ones(2))),
            )
        )
        offset = np.concatenate(
            (safety[current] - 1e-8, 1.0 - 1e-4 - squared[current][:cursor].ravel(), 1.0 - squared[current][cursor])
        )
        expected.update(objective_gradient=g, nonlinear_row_count=len(offset))
    compare_contract(expected, phase["linearization"], compare=compare, name="linearization")
    if not complete or witness_available:
        require(phase["qp"] is None and phase["qp_direction"] is None, "No QP before complete nonwitness stencil")
        integer(phase["qp_iterations"], 0, 0, "No fabricated QP iterations")
        require("hessian_condition_number" not in phase, "No curvature-use diagnostic before QP generation")
        expected_update = dict(updated=False, reason="no_pending_pair" if complete else "incomplete_stencil")
        compare_contract(expected_update, phase["curvature_update"], compare=compare, name="curvature_update")
        require(
            phase["status"] == ("witness_found" if complete else "response_budget"),
            "Stencil interruption reason is recorded without a fabricated optimizer return",
        )
        _exact_array(B, phase["hessian_after_update"], "No B update from an interrupted or witness stencil")
        return B, pending, g, nonlinear, qp_check, curvature_check

    if pending is not None:
        require(
            pending["accepted_response_index"] == current and pending["cursor"] == cursor,
            "A secant terminates at its actual accepted point and unchanged cursor",
        )
        new_gradient = g - nonlinear[:, :width].T @ pending["multipliers"]
        s = x - pending["base_commands"]
        y = new_gradient - pending["lagrangian_gradient"]
        record = phase["curvature_update"]
        binding = dict(
            source_stage=pending["stage_index"],
            source_response_index=pending["base_response_index"],
            accepted_response_index=current,
            displacement=s,
            gradient_change=y,
            previous_lagrangian_gradient=pending["lagrangian_gradient"],
            current_lagrangian_gradient=new_gradient,
            step_inf_over_fd=float(np.max(np.abs(s)) / STEP),
        )
        compare_contract(binding, {name: record[name] for name in binding}, compare=compare, name="secant_binding")
        curvature_check = audit_curvature_update(B, s, y, record)
        # Carry the checked archived B, keeping its exact bits for the next QP.
        B = np.asarray(record["matrix"], dtype=float).copy()
        pending = None
    else:
        compare_contract(
            dict(updated=False, reason="no_pending_pair"),
            phase["curvature_update"],
            compare=compare,
            name="curvature_update",
        )
    _exact_array(B, phase["hessian_after_update"], "Only the checked accepted-point update changes B")
    condition = float(np.linalg.cond(B))
    expected_condition = condition if np.isfinite(condition) else None
    compare_contract(expected_condition, phase["hessian_condition_number"], compare=compare, name="condition_number")
    difference = np.eye(width) - np.eye(width, k=-6)
    previous = np.asarray(answer["previous_command"], dtype=float)
    center = np.concatenate((previous, np.zeros(width - 6)))
    rate = np.column_stack((difference, np.zeros(width)))
    box = np.column_stack((np.eye(width), np.zeros(width)))
    A = np.vstack((nonlinear, rate, -rate, box, -box, np.r_[np.zeros(width), 1.0][None]))
    b = np.concatenate(
        (
            offset,
            difference @ x - center + 0.25,
            0.25 - difference @ x + center,
            x - np.asarray(phase["stage_lower_command"]).ravel(),
            np.asarray(phase["stage_upper_command"]).ravel() - x,
            [0.0],
        )
    )
    qp = phase["qp"]
    require(isinstance(qp, dict), "One saved explicit QP attempt after a complete nonwitness stencil")
    qp_check = audit_qp_certificate(qp, B, g, A, b)
    require(phase["qp_iterations"] == qp["solver_nit"], "Actual algebraic QP iteration count")
    direction = None
    if not qp["qualified"]:
        status = "qp_certificate_failed"
    else:
        point = np.asarray(qp["point"], dtype=float)
        trial = x + point[:width]
        lower, upper = (np.asarray(phase[name]).ravel() for name in ("stage_lower_command", "stage_upper_command"))
        if np.any(trial < lower - 1e-12) or np.any(trial > upper + 1e-12):
            status = "qp_command_bounds_failed"
        elif not np.any(trial != x):
            status = "qp_zero_direction"
        else:
            status = "qp_direction_qualified"
            direction = dict(
                commands=trial.reshape(horizon, 6),
                displacement=(trial - x).reshape(horizon, 6),
                epigraph=float(point[width]),
            )
    require(phase["status"] == status, "QP status follows the independently checked certificate and unprojected point")
    compare_contract(direction, phase["qp_direction"], compare=compare, name="qp_direction")
    if direction is not None:
        require(
            command_key(direction["commands"], horizon) == command_key(phase["qp_direction"]["commands"], horizon),
            "QP direction retains the exact unprojected trial bytes",
        )
        _exact_array(direction["displacement"], phase["qp_direction"]["displacement"], "Exact actual QP displacement")
    return B, pending, g, nonlinear, qp_check, curvature_check


def audit_accepted_point_sqp_ledger(answer, classifications, *, compare=numerical_compare):
    """Check exact cache/FD routing, saved-value algebra, acceptance, and budgets.

    Numerical replay must first validate every trace's full safety vector and TCP
    error, plus each ray/anchor full audit. Returned indices identify those full
    records; this function does not execute or independently certify dynamics.
    """
    squared, keys, prefixes, best, safety = _saved_trace_state(answer, classifications, compare)
    trace, horizon = answer["evaluated_candidates"], answer["horizon"]
    budget, iteration_budget, stage_budget = (
        answer[key] for key in ("max_evaluations", "max_iterations", "max_stages")
    )
    integer(budget, 6 * horizon + 1, 4000, "Finite global response budget")
    integer(iteration_budget, 2, 60, "Finite global iteration budget")
    integer(stage_budget, 1, 60, "Finite global stage budget")
    require(
        answer["finite_difference_step"] == STEP and answer["optimization_safety_buffer"] == 1e-8,
        "Original derivative step and optimization buffer",
    )
    require(
        isinstance(answer["backtracking_alphas"], list)
        and np.array_equal(np.asarray(answer["backtracking_alphas"]), ALPHAS),
        "Exact fixed backtracking alpha list",
    )
    require(answer["response_evaluations"] == len(trace) <= budget, "All unique responses count against one budget")
    require(
        command_key(answer["initial"]["commands"], horizon) == command_key(answer["used_start"], horizon) == keys[0],
        "Initial response binds the exact warm start",
    )
    require(trace[0]["joint_feasible"] is True, "Original warm joint and command audit")
    if not trace[0]["feasible"]:
        require(
            trace[0]["safety_min"] >= 1e-8 and classifications[0]["safety_min"] >= 1e-8,
            "Incomplete warm satisfies the original buffer",
        )
    safe = [i for i, row in enumerate(trace) if row["joint_feasible"]]
    witnesses = [i for i in safe if trace[i]["feasible"]]
    selected = best(safe)
    require(
        command_key(answer["selected"]["commands"], horizon) == keys[selected],
        "Final selection uses all cached safe responses and original ranking",
    )
    require(
        answer["witness_found"] is bool(witnesses)
        and answer["status"] == ("witness" if witnesses else "finite_search_without_witness")
        and answer["initial_prefix_length"] == prefixes[0]
        and answer["selected_prefix_length"] == prefixes[selected],
        "Strict final witness and prefix labels",
    )
    require(
        answer["method"] == "accepted_point_lagrangian_sqp"
        and answer["prefix_optimization_margin"] == 1e-4
        and answer["qp_iteration_limit"] == 100
        and answer["qp_certificate_tolerance"] == 1e-6,
        "Registered explicit SQP, prefix reserve, and local certificate constants",
    )
    require(
        answer["pose_caps_horizon"] == horizon and answer["all_pose_caps_enforced"] is True,
        "Witness retains all original full-horizon pose caps",
    )
    previous = np.asarray(answer["previous_command"], dtype=float)
    require(previous.shape == (6,) and np.isfinite(previous).all(), "Explicit causal previous command")
    phases, requests = answer["phases"], answer["request_ledger"]
    integer(answer["total_stages"], len(phases), len(phases), "Exact total stage count")
    require(len(phases) <= stage_budget, "Shared stage budget")
    decoded, touched, callbacks, lo, hi = _audit_requests(answer, keys)
    require(not any(callbacks.values()), "Explicit SQP has no nonlinear SLSQP callbacks")
    warm = np.asarray(answer["supplied_warm"], dtype=float)
    require(warm.shape == (horizon, 6) and np.isfinite(warm).all(), "Finite supplied warm")
    clipped = np.clip(warm, lo, hi)
    require(command_key(clipped, horizon) == keys[0], "Warm follows the declared one-time roundoff clipping")
    _exact_array(clipped - warm, answer["start_roundoff_clip"], "Explicit warm roundoff difference")
    require(
        np.all(warm >= lo - 1e-12)
        and np.all(warm <= hi + 1e-12)
        and np.all(np.abs(np.diff(np.vstack((previous, warm)), axis=0)) <= 0.25 + 1e-12),
        "Supplied warm satisfies original box and rate admission",
    )
    B, pending = np.eye(6 * horizon), None
    qp_checks, curvature_checks = [], []
    qp_iterations = 0
    current, used_iterations, radius_index = 0, 0, 0
    request_boundary, response_boundary = 1, 1
    accepted_specs, accepted_indices = [(0, 0, None)], [0]
    phase_checks, ray_response_indices = [], []
    ray_count = ray_new = 0

    for stage_index, phase in enumerate(phases):
        integer(phase["stage_index"], stage_index, stage_index, "Contiguous phase index")
        require(phase["phase"] == "accepted_point_sqp", "Registered explicit SQP stage")
        require(
            not trace[current]["feasible"]
            and radius_index < len(RADII)
            and response_boundary < budget
            and used_iterations < iteration_budget,
            "Only unfinished searches with remaining budgets open a stage",
        )
        require(
            phase["anchor_before_index"] == current
            and phase["requests_before"] == request_boundary
            and phase["evaluations_before"] == response_boundary,
            "Stage begins at the preceding accepted anchor and contiguous work boundary",
        )
        end = integer(
            phase["requests_after"], request_boundary + 1, len(requests), "Finite nonempty stage request interval"
        )
        generation_end = integer(
            phase["generation_requests_after"], request_boundary + 1, end, "Finite generation boundary"
        )
        response_end = requests[end - 1]["response_evaluations_after"]
        generation_responses = requests[generation_end - 1]["response_evaluations_after"]
        require(
            phase["evaluations_after"] == response_end
            and phase["response_budget_remaining_before"] == budget - response_boundary
            and phase["response_budget_remaining_after"] == budget - response_end,
            "Exact global response accounting across generation and ray",
        )
        require(
            phase["radius_index"] == radius_index
            and phase["radius"] == RADII[radius_index]
            and type(phase["iterations"]) is int
            and phase["iterations"] == 1,
            "Fixed radius and one counted nonlinear attempt per stage",
        )
        anchor = np.asarray(trace[current]["commands"], dtype=float)
        require(
            np.array_equal(np.asarray(phase["stage_lower_command"]), np.maximum(lo, anchor - RADII[radius_index]))
            and np.array_equal(np.asarray(phase["stage_upper_command"]), np.minimum(hi, anchor + RADII[radius_index])),
            "Exact trust-box intersection at the accepted anchor",
        )
        cursor = prefixes[current]
        require(
            phase["prefix_before"] == phase["target_cursor"] == phase["cursor"] == cursor
            and command_key(phase["initial_commands"], horizon) == keys[current],
            "Stage targets the anchor's first failed cursor",
        )
        compare(
            "initial_epigraph", max(0.0, float(np.max(squared[current][cursor])) - 1) + 1e-6, phase["initial_epigraph"]
        )
        compare("initial_safety", trace[current]["safety_min"], phase["initial_safety_min"])
        first = requests[request_boundary]
        require(
            first["role"] == "stage_anchor"
            and first["command_key"] == keys[current]
            and first["outcome"] == "cache_hit"
            and sum(row["role"] == "stage_anchor" for row in requests[request_boundary:end]) == 1,
            "One explicit cached stage anchor request",
        )
        require(
            all(row["role"] != "backtracking" for row in requests[request_boundary:generation_end])
            and all(row["role"] == "backtracking" for row in requests[generation_end:end]),
            "Generation and ray are consecutive separate request intervals",
        )
        B, pending, g, nonlinear, qp_check, curvature_check = _audit_generation(
            answer, phase, current, squared, safety, keys, B, pending, compare
        )
        iterations = 1
        qp_iterations += phase["qp_iterations"]
        if qp_check is not None:
            qp_checks.append(dict(stage_index=stage_index, **qp_check))
        if curvature_check is not None:
            curvature_checks.append(dict(stage_index=stage_index, **curvature_check))
        direction = phase["qp_direction"]

        attempts = phase["ray_attempts"]
        require(len(attempts) == end - generation_end <= len(ALPHAS), "One ray attempt per actual ray request")
        prior_witnesses = [i for i in range(generation_responses) if trace[i]["feasible"]]
        if prior_witnesses or direction is None:
            require(not attempts, "No ray without a qualified QP direction or after an existing global witness")
        chosen_ray = None
        denied = False
        for alpha_index, attempt in enumerate(attempts):
            ray_count += 1
            request_index = generation_end + alpha_index
            request = requests[request_index]
            require(
                direction is not None and chosen_ray is None and not denied,
                "Ray stops at its first qualified point or budget denial",
            )
            trial = np.asarray(direction["commands"], dtype=float)
            expected_command = trial.copy() if alpha_index == 0 else anchor + ALPHAS[alpha_index] * (trial - anchor)
            require(
                request["alpha_index"] == attempt["alpha_index"] == alpha_index
                and request["alpha"] == attempt["alpha"] == ALPHAS[alpha_index],
                "Fixed descending alpha order without additions or skipped points",
            )
            require(
                command_key(expected_command, horizon)
                == request["command_key"]
                == command_key(attempt["commands"], horizon),
                "Exact unprojected ray interpolation; alpha1 copies trial bytes",
            )
            require(
                attempt["request_index"] == request_index
                and attempt["response_index"] == request["response_index"]
                and attempt["new_response_index"] == request["new_response_index"]
                and attempt["outcome"] == request["outcome"],
                "Ray records the actual global cache and charge outcome",
            )
            ray_new += int(request["outcome"] == "new_response")
            response = request["response_index"]
            if response is None:
                denied = True
                require(
                    attempt["entry"] is None
                    and attempt["eligible"] is False
                    and attempt["selected"] is False
                    and attempt["rejection_reasons"] == ["response_budget"],
                    "Incomplete ray attempt cannot fabricate an evaluated entry",
                )
                continue
            ray_response_indices.append(response)
            entry = attempt["entry"]
            require(
                isinstance(entry, dict)
                and "audit" in entry
                and "rollout" in entry
                and command_key(entry["commands"], horizon) == keys[response],
                "Every completed ray retains its exact full audit and rollout",
            )
            require(
                entry["audit"]["feasible"] is trace[response]["joint_feasible"]
                and entry["feasible"] is trace[response]["feasible"]
                and entry["prefix_length"] == prefixes[response],
                "Strict saved ray entry labels match its unique response",
            )
            compare("ray_errors", squared[response], entry["normalized_squared_errors"])
            compare("ray_tcp_errors", trace[response]["tcp_errors"], entry["tcp_errors"])
            compare("ray_safety_slack", safety[response], entry["safety_slack"])
            reasons = (
                []
                if trace[response]["feasible"]
                else _ordinary_rejections(response, current, cursor, trace, squared, prefixes)
            )
            eligible = not reasons
            if eligible and not trace[response]["feasible"]:
                require(classifications[response]["safety_min"] >= 1e-8, "Independent ordinary ray buffer")
            require(
                attempt["eligible"] is eligible and attempt["rejection_reasons"] == reasons,
                "Exact original audit/prefix/buffer/descent ray qualification",
            )
            if eligible:
                chosen_ray = response
        completed_witnesses = [i for i in range(response_end) if trace[i]["feasible"]]
        chosen = best(completed_witnesses) if completed_witnesses else chosen_ray
        accepted = chosen is not None
        next_anchor = chosen if accepted else current
        reason = (
            None
            if not accepted
            else "witness"
            if trace[chosen]["feasible"]
            else "prefix_growth"
            if prefixes[chosen] > cursor
            else "target_descent"
        )
        ray_termination = (
            "witness"
            if completed_witnesses
            else "accepted"
            if chosen_ray is not None
            else "response_budget"
            if denied or phase["status"] == "response_budget"
            else "no_direction"
            if direction is None
            else "alphas_exhausted"
        )
        require(phase["ray_termination"] == ray_termination, "Honest finite ray stopping reason")
        if ray_termination == "alphas_exhausted":
            require(
                len(attempts) == len(ALPHAS), "An unsuccessful qualified direction uses the complete fixed alpha list"
            )
        for attempt in attempts:
            require(
                attempt["selected"] is bool(accepted and attempt["response_index"] == chosen),
                "Ray selection reflects the final accepted response",
            )
        require(
            phase["touched_response_indices"] == sorted(touched[stage_index]),
            "Stage touched set is every completed request and no future response",
        )
        require(
            phase["accepted"] is accepted
            and phase["acceptance_reason"] == reason
            and phase["anchor_after_index"] == next_anchor
            and phase["prefix_after"] == prefixes[next_anchor]
            and command_key(phase["selected_commands"], horizon) == keys[next_anchor],
            "Ordinary anchors come only from the first qualified ray; complete witnesses may come from any cache entry",
        )
        next_radius = RADII[0] if accepted else RADII[radius_index + 1] if radius_index + 1 < len(RADII) else None
        require(phase["radius_after"] == next_radius, "Acceptance recenters; rejection shrinks the fixed radius")
        if accepted:
            if prefixes[chosen] != cursor:
                B, pending = np.eye(6 * horizon), None
                reset = "cursor_changed"
            else:
                require(
                    phase["qp"] is not None and phase["qp"]["qualified"],
                    "Ordinary accepted ray has certified QP origin",
                )
                multipliers = np.asarray(phase["qp"]["multipliers"], dtype=float)[: len(nonlinear)]
                pending = dict(
                    stage_index=stage_index,
                    cursor=cursor,
                    base_response_index=current,
                    accepted_response_index=chosen,
                    base_commands=anchor.ravel().copy(),
                    multipliers=multipliers.copy(),
                    lagrangian_gradient=g - nonlinear[:, : 6 * horizon].T @ multipliers,
                )
                reset = None
        else:
            reset = None
        require(phase["curvature_reset"] == reset, "Only cursor changes reset curvature and pending pair")
        _exact_array(B, phase["hessian_after_stage"], "Rejected rays preserve B; cursor growth resets to identity")
        require(
            phase["pending_pair_after_stage"] == (None if pending is None else pending["stage_index"]),
            "Actual same-cursor accepted displacement remains pending exactly once",
        )
        if accepted:
            accepted_indices.append(chosen)
            accepted_specs.append((chosen, current, stage_index))
        used_iterations += iterations
        phase_checks.append(
            dict(
                stage_index=stage_index,
                selected_trace_index=next_anchor,
                accepted=accepted,
                radius=phase["radius"],
                ray_termination=ray_termination,
                ray_attempt_count=len(attempts),
                touched_response_indices=sorted(touched[stage_index]),
            )
        )
        current, request_boundary, response_boundary = next_anchor, end, response_end
        radius_index = 0 if accepted else radius_index + 1

    require(
        request_boundary == len(requests) and response_boundary == len(trace),
        "All observed work is warm or one declared stage",
    )
    require(
        answer["total_iterations"] == used_iterations <= iteration_budget
        and answer["total_backtracking_attempts"] == ray_count
        and answer["total_backtracking_new_responses"] == ray_new,
        "Shared iterations and all cache/denied ray attempts are counted separately",
    )
    require(
        [row["response_index"] for row in answer["accepted_anchors"]] == accepted_indices,
        "Exact accepted-anchor history",
    )
    for record, (index, before, stage_index) in zip(answer["accepted_anchors"], accepted_specs, strict=True):
        cursor = (
            phases[stage_index]["target_cursor"]
            if stage_index is not None
            else None
            if trace[index]["feasible"]
            else prefixes[index]
        )
        before_score = float(np.max(squared[before][cursor])) if cursor is not None else None
        after_score = float(np.max(squared[index][cursor])) if cursor is not None else None
        expected = dict(
            is_initial=stage_index is None,
            stage_index=stage_index,
            response_index=index,
            previous_response_index=before if stage_index is not None else None,
            commands=trace[index]["commands"],
            prefix_length=prefixes[index],
            joint_feasible=trace[index]["joint_feasible"],
            safety_min=trace[index]["safety_min"],
            feasible=trace[index]["feasible"],
            tcp_errors=trace[index]["tcp_errors"],
            normalized_squared_errors=squared[index],
            stage_target_cursor=cursor,
            stage_target_max_before=before_score,
            stage_target_max_after=after_score,
            target_improvement=before_score - after_score if cursor is not None else None,
            radius=phases[stage_index]["radius"] if stage_index is not None else None,
            acceptance_reason=phases[stage_index]["acceptance_reason"]
            if stage_index is not None
            else "initial_witness"
            if trace[index]["feasible"]
            else "warm",
        )
        require(set(record) == set(expected) | {"audit"}, "Complete accepted-anchor metadata plus full numerical audit")
        compare_contract(expected, {key: record[key] for key in expected}, compare=compare, name="accepted_anchor")
        require(command_key(record["commands"], horizon) == keys[index], "Exact accepted-anchor command bytes")
    termination = (
        "witness"
        if witnesses
        else "response_budget"
        if len(trace) >= budget
        else "iteration_budget"
        if used_iterations >= iteration_budget
        else "stage_budget"
        if len(phases) >= stage_budget
        else "trust_region_stalled"
    )
    if termination == "trust_region_stalled":
        require(radius_index == len(RADII), "Finite termination accounts for all three rejected radii")
    require(
        answer["termination_reason"] == termination
        and answer["final_selected_is_last_anchor"] is (selected == current),
        "Global final rank remains separate from accepted anchor history",
    )
    _exact_array(B, answer["final_hessian"], "Final curvature retains the actual stage lifecycle")
    require(
        answer["pending_pair_source_stage"] == (None if pending is None else pending["stage_index"])
        and answer["total_qp_iterations"] == qp_iterations <= 100 * len(phases),
        "Final pending pair and distinct algebraic iteration accounting",
    )
    return dict(
        selected_trace_index=selected,
        joint_feasible_indices=safe,
        witness_indices=witnesses,
        initial_prefix_length=prefixes[0],
        selected_prefix_length=prefixes[selected],
        total_iterations=used_iterations,
        total_qp_iterations=qp_iterations,
        qp_certificate_checks=qp_checks,
        curvature_checks=curvature_checks,
        phase_checks=phase_checks,
        accepted_anchor_indices=accepted_indices,
        ray_response_indices=ray_response_indices,
        request_count=len(requests),
        total_backtracking_attempts=ray_count,
        total_backtracking_new_responses=ray_new,
        termination_reason=termination,
        full_ray_and_anchor_audits_checked_here=False,
    )
