"""Audit callback-ray bookkeeping after independent numerical classification.

This module never imports the producer solver or runs a model. Full ray and
anchor audit arrays must additionally be checked by the numerical checker.
"""

import numpy as np
from check_tracking_v5_prefix_trust_control import (
    command_key,
    compare_contract,
    numerical_compare,
    trace_ranks,
)

RADII = (0.01, 0.005, 0.0025)
ALPHAS = (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 0.00390625)
STEP = 1e-4


def require(condition, message):
    if not condition:
        raise ValueError(message)


def integer(value, lower, upper, message):
    require(type(value) is int and lower <= value <= upper, message)
    return value


def _audit_requests(answer, keys):
    """Replay exact response-cache identity and every recorded FD request."""
    horizon, phases, requests = answer["horizon"], answer["phases"], answer["request_ledger"]
    budget, width = answer["max_evaluations"], 6 * horizon
    lo, hi = (np.asarray(answer[name], dtype=float) for name in ("lower_command", "upper_command"))
    require(
        lo.shape == hi.shape == (horizon, 6)
        and np.isfinite(lo).all()
        and np.isfinite(hi).all()
        and np.all(hi - lo > STEP),
        "Finite nondegenerate original command bounds",
    )
    roles = {
        "warm",
        "stage_anchor",
        "objective",
        "constraint",
        "gradient_base",
        "gradient_probe",
        "jacobian_base",
        "jacobian_probe",
        "callback",
        "final",
        "backtracking",
    }
    seen, decoded = {}, []
    touched = {stage: set() for stage in range(len(phases))}
    callbacks = {stage: [] for stage in range(len(phases))}
    for index, request in enumerate(requests):
        integer(request["request_index"], index, index, "Contiguous request indices")
        integer(request["response_evaluations_before"], len(seen), len(seen), "Exact prior response count")
        stage, role, key = request["stage_index"], request["role"], request["command_key"]
        require(role in roles and isinstance(key, str), "Known request role and explicit command bytes")
        try:
            command = np.frombuffer(bytes.fromhex(key), dtype=np.float64).reshape(horizon, 6)
        except (TypeError, ValueError) as error:
            raise ValueError("Invalid full float64 command bytes") from error
        require(command_key(command, horizon) == key, "Canonical finite command key")
        decoded.append(command)
        if index == 0:
            require(stage is None and role == "warm" and key == keys[0], "One charged initial warm")
            lower, upper = lo, hi
        else:
            integer(stage, 0, len(phases) - 1, "A declared request stage")
            phase = phases[stage]
            require(
                role != "warm" and phase["requests_before"] <= index < phase["requests_after"],
                "Request belongs to stage interval",
            )
            lower, upper = np.asarray(phase["stage_lower_command"]), np.asarray(phase["stage_upper_command"])
            require(
                lower.shape == upper.shape == (horizon, 6) and np.isfinite(lower).all() and np.isfinite(upper).all(),
                "Finite stage bounds",
            )
        require(
            np.all(command >= lower - 1e-12) and np.all(command <= upper + 1e-12),
            "Every cached or new request respects current bounds",
        )
        outcome, response = request["outcome"], request["response_index"]
        if outcome == "new_response":
            integer(response, len(seen), len(seen), "Ordered unique response index")
            require(
                key not in seen and len(seen) < budget and response < len(keys) and key == keys[response],
                "Every new response is uniquely charged",
            )
            integer(request["new_response_index"], response, response, "Explicit charged response index")
            seen[key] = response
        elif outcome == "cache_hit":
            require(
                key in seen
                and type(response) is int
                and response == seen[key]
                and request["new_response_index"] is None,
                "Exact earlier response cache hit",
            )
        else:
            require(
                outcome == "budget_rejected"
                and index == len(requests) - 1
                and key not in seen
                and len(seen) == budget
                and response is request["new_response_index"] is None,
                "Only the final uncached request can be budget denied",
            )
        integer(request["response_evaluations_after"], len(seen), len(seen), "Exact subsequent response count")
        if stage is not None:
            if response is not None:
                touched[stage].add(response)
            if role == "callback":
                callbacks[stage].append(index)
        if not role.endswith("_probe"):
            require(request["axis"] is request["nominal_delta"] is None, "Only FD probes have axis and nominal delta")
        if role == "backtracking":
            alpha_index = integer(request["alpha_index"], 0, len(ALPHAS) - 1, "Registered backtracking index")
            require(request["alpha"] == ALPHAS[alpha_index], "Registered exact backtracking alpha")
        else:
            require(request["alpha_index"] is request["alpha"] is None, "Only backtracking requests carry alpha")
    require(
        len(seen) == len(keys) and requests and requests[0]["outcome"] == "new_response",
        "All and only completed unique responses are retained",
    )
    covered = set()
    for index, request in enumerate(requests):
        if request["role"] not in ("gradient_base", "jacobian_base"):
            continue
        if request["outcome"] == "budget_rejected":
            continue
        stage = request["stage_index"]
        base, upper = decoded[index].ravel(), np.asarray(phases[stage]["stage_upper_command"]).ravel()
        for axis in range(width):
            next_index = index + axis + 1
            require(next_index < len(requests), "A derivative ends with all axes or explicit budget denial")
            probe = requests[next_index]
            require(
                probe["stage_index"] == stage
                and probe["role"] == request["role"].replace("_base", "_probe")
                and type(probe["axis"]) is int
                and probe["axis"] == axis,
                "Contiguous ordered same-stage FD group",
            )
            delta = STEP if base[axis] + STEP <= upper[axis] else -STEP
            expected = base.copy()
            expected[axis] += delta
            require(
                probe["nominal_delta"] == delta and expected.tobytes().hex() == probe["command_key"],
                "Finite differences obey actual stage bounds and nominal step",
            )
            require(next_index not in covered, "A probe belongs to exactly one FD group")
            covered.add(next_index)
            if probe["outcome"] == "budget_rejected":
                break
    require(
        covered == {i for i, row in enumerate(requests) if row["role"].endswith("_probe")},
        "No orphan finite-difference request",
    )
    return decoded, touched, callbacks, lo, hi


def _ordinary_rejections(index, anchor, cursor, trace, squared, prefixes):
    row, reasons = trace[index], []
    if not row["joint_feasible"]:
        reasons.append("joint_or_command")
    if prefixes[index] < cursor:
        reasons.append("prefix_lost")
    if row["safety_min"] < 1e-8:
        reasons.append("safety_buffer")
    improvement = float(np.max(squared[anchor][cursor])) - float(np.max(squared[index][cursor]))
    if not (prefixes[index] > cursor or improvement > 1e-12):
        reasons.append("no_target_descent")
    return reasons


def audit_callback_backtracking_ledger(answer, classifications, *, compare=numerical_compare):
    """Validate saved control flow after all unique responses were reclassified.

    The returned ray/anchor indices identify full records whose rollout and nine
    audit groups still require the caller's independent numerical comparison.
    """
    squared, keys, prefixes, best = trace_ranks(answer, classifications, compare=compare)
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
    phases, requests = answer["phases"], answer["request_ledger"]
    integer(answer["total_stages"], len(phases), len(phases), "Exact total stage count")
    require(len(phases) <= stage_budget, "Shared stage budget")
    decoded, touched, callback_requests, lo, hi = _audit_requests(answer, keys)
    current, used_iterations, radius_index = 0, 0, 0
    request_boundary, response_boundary = 1, 1
    accepted_specs, accepted_indices = [(0, 0, None)], [0]
    phase_checks, ray_response_indices = [], []
    ray_count = ray_new = 0

    for stage_index, phase in enumerate(phases):
        integer(phase["stage_index"], stage_index, stage_index, "Contiguous phase index")
        require(phase["phase"] == "prefix_callback_backtracking", "Registered callback-ray stage")
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
            and phase["max_iterations"] == min(5, iteration_budget - used_iterations),
            "Fixed radius and remaining iteration schedule",
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
        callbacks = callback_requests[stage_index]
        integer(
            phase["callback_count"], len(callbacks), len(callbacks), "Every callback, including a budget denial, counts"
        )
        iterations = integer(
            phase["iterations"], len(callbacks), phase["max_iterations"], "Finite actual stage iterations"
        )
        require(
            type(phase["optimizer_returned"]) is bool and type(phase["success"]) is bool,
            "Explicit optimizer return status",
        )
        zero_callbacks, captured_request = 0, None
        callback_witness = False
        for callback_index in callbacks:
            request = requests[callback_index]
            require(callback_index < generation_end, "No callback after direction generation ended")
            if request["outcome"] == "budget_rejected":
                require(
                    callback_index == generation_end - 1 and phase["status"] == "response_budget",
                    "Denied callback ends generation",
                )
                continue
            is_zero = bool(np.all(decoded[callback_index] - anchor == 0))
            zero_callbacks += int(is_zero)
            available_witness = any(row["feasible"] for row in trace[: request["response_evaluations_after"]])
            if available_witness:
                require(
                    callback_index == generation_end - 1 and phase["status"] == "witness_found",
                    "First callback observing a complete global witness stops generation",
                )
                callback_witness = True
            elif not is_zero:
                require(
                    captured_request is None
                    and callback_index == generation_end - 1
                    and phase["status"] == "direction_captured",
                    "The first nonzero callback immediately stops generation",
                )
                captured_request = callback_index
        integer(
            phase["zero_callback_count"],
            zero_callbacks,
            zero_callbacks,
            "Actual completed zero callbacks are counted separately",
        )
        captured = phase["captured_callback"]
        if captured_request is None:
            require(captured is None, "No fabricated captured direction")
        else:
            request = requests[captured_request]
            expected = dict(
                request_index=captured_request,
                response_index=request["response_index"],
                commands=decoded[captured_request],
                epigraph=phase["last_callback_epigraph"],
                displacement=decoded[captured_request] - anchor,
                zero_step=False,
            )
            compare_contract(expected, captured, compare=compare, name="captured_callback")
            require(
                command_key(captured["commands"], horizon) == request["command_key"],
                "Captured trial preserves the exact callback command bytes",
            )
            require(
                np.array_equal(np.asarray(captured["displacement"]), decoded[captured_request] - anchor)
                and captured["epigraph"] == phase["last_callback_epigraph"],
                "Exact captured displacement and epigraph snapshot",
            )
        if callbacks:
            require(
                command_key(phase["last_callback_commands"], horizon) == requests[callbacks[-1]]["command_key"]
                and np.isfinite(phase["last_callback_epigraph"])
                and phase["last_callback_epigraph"] >= 0,
                "Final callback snapshot and epigraph",
            )
        else:
            require(
                "last_callback_commands" not in phase and "last_callback_epigraph" not in phase,
                "No invented callback snapshot",
            )
        if phase["optimizer_returned"]:
            nit = integer(phase["optimizer_iterations"], 0, phase["max_iterations"], "Actual bounded normal-return nit")
            require(
                iterations == max(len(callbacks), nit) and captured is None and not callback_witness,
                "Normal return does not fabricate an interrupted optimizer",
            )
            finals = [i for i in range(request_boundary, generation_end) if requests[i]["role"] == "final"]
            require(
                finals == [generation_end - 1]
                and command_key(phase["final_commands"], horizon) == requests[finals[0]]["command_key"]
                and np.isfinite(phase["final_epigraph"])
                and phase["final_epigraph"] >= 0,
                "Normal final request is retained even if budget denied",
            )
        else:
            require(
                iterations == len(callbacks)
                and "optimizer_iterations" not in phase
                and not any(row["role"] == "final" for row in requests[request_boundary:generation_end]),
                "Interrupted generation charges only actual callbacks and has no fake return",
            )
        if phase["status"] == "direction_captured":
            require(
                captured is not None and phase["optimizer_returned"] is False and phase["success"] is False,
                "Explicit first-direction interruption",
            )
        elif phase["status"] == "witness_found":
            require(
                callback_witness and phase["optimizer_returned"] is False and phase["success"] is False,
                "Complete-witness callback interruption",
            )
        elif phase["status"] == "response_budget":
            require(
                requests[generation_end - 1]["outcome"] == "budget_rejected" and phase["success"] is False,
                "Generation budget interruption records its failed request",
            )
        else:
            require(
                type(phase["status"]) is int
                and phase["optimizer_returned"] is True
                and phase["success"] is (phase["status"] == 0),
                "Normal optimizer status does not establish acceptance",
            )

        attempts = phase["ray_attempts"]
        require(len(attempts) == end - generation_end <= len(ALPHAS), "One ray attempt per actual ray request")
        prior_witnesses = [i for i in range(generation_responses) if trace[i]["feasible"]]
        if prior_witnesses or captured is None:
            require(not attempts, "No ray without a captured direction or after an existing global witness")
        chosen_ray = None
        denied = False
        for alpha_index, attempt in enumerate(attempts):
            ray_count += 1
            request_index = generation_end + alpha_index
            request = requests[request_index]
            require(
                captured is not None and chosen_ray is None and not denied,
                "Ray stops at its first qualified point or budget denial",
            )
            trial = np.asarray(captured["commands"], dtype=float)
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
            if captured is None
            else "alphas_exhausted"
        )
        require(phase["ray_termination"] == ray_termination, "Honest finite ray stopping reason")
        if ray_termination == "alphas_exhausted":
            require(
                len(attempts) == len(ALPHAS), "An unsuccessful captured direction uses the complete fixed alpha list"
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
    return dict(
        selected_trace_index=selected,
        joint_feasible_indices=safe,
        witness_indices=witnesses,
        initial_prefix_length=prefixes[0],
        selected_prefix_length=prefixes[selected],
        total_iterations=used_iterations,
        phase_checks=phase_checks,
        accepted_anchor_indices=accepted_indices,
        ray_response_indices=ray_response_indices,
        request_count=len(requests),
        total_backtracking_attempts=ray_count,
        total_backtracking_new_responses=ray_new,
        termination_reason=termination,
        full_ray_and_anchor_audits_checked_here=False,
    )
