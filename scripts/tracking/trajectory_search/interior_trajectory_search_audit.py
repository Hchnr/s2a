"""Independent saved-algebra audit for bounded interior trajectory search.

This module imports no search, solver, dynamics, or response evaluator. Its
result concerns the saved ledger and local models; full native replay remains
the caller's separate responsibility.
"""

import numpy as np
from accepted_point_sqp_math_audit import audit_qp_certificate
from directional_derivative_audit import _response, array, boolean, exact_array, integer, numerical_compare, require
from restored_anchor_sqp_ledger_audit import compare_initial_records
from terminal_secant_audit import _tree


def _periods():
    periods = np.arange(12)
    return (
        np.r_[np.repeat(periods, 175), np.repeat(periods, 175), np.repeat(periods, 7), np.repeat(periods, 7), 11],
        np.repeat(periods, 2),
    )


def _pose(entry):
    error = array(entry["tcp_errors"], "full H12 pose errors", (12, 6))
    squared = np.column_stack(
        (np.sum(error[:, :3] ** 2, axis=1) / 0.01**2, np.sum(error[:, 3:] ** 2, axis=1) / 0.05**2)
    )
    exact_array("saved pose algebra", squared, entry["normalized_squared_errors"])
    return squared.ravel()


def _metrics(safety, squared, config):
    weights = np.ones(4369)
    weights[-1] = config["terminal_time_scale"]
    reserve = float(np.min(weights * (safety - config["safety_buffer"]) / config["reserve_target"]))
    return dict(
        reserve=reserve,
        violation=float(max(0.0, 1.0 - reserve)),
        pose_excess=float(max(0.0, np.max(squared) - 1.0)),
        safety_min=float(np.min(safety)),
        pose_max=float(np.max(squared)),
    )


def _causal_update(J, step, observed, periods, config):
    """Rebuild each masked, damped row update and every denominator independently."""
    step = array(step, "actual accepted displacement", (72,))
    J = array(J, "current causal Jacobian", (len(periods), 72))
    observed = array(observed, "actual observed row change", (len(periods),))
    mask = np.repeat(np.arange(12), 6)[None, :] <= periods[:, None]
    require(np.all(J[~mask] == 0), "Every future Jacobian column is exactly zero")
    masked = mask * step[None, :]
    denominator = np.sum(masked**2, axis=1)
    residual = observed - J @ step
    row_norm = np.linalg.norm(J, axis=1)
    correction = np.zeros_like(J)
    updated = denominator >= config["secant_min_squared_step"]
    correction[updated] = residual[updated, None] * masked[updated] / denominator[updated, None]
    correction_norm = np.linalg.norm(correction, axis=1)
    theta = np.zeros(len(periods))
    theta[updated] = 1.0
    nonzero = updated & (correction_norm > 0)
    theta[nonzero] = np.minimum(
        1.0,
        config["secant_relative_limit"]
        * np.maximum(row_norm[nonzero], config["secant_row_norm_floor"])
        / correction_norm[nonzero],
    )
    result = J + theta[:, None] * correction
    require(np.all(result[~mask] == 0), "Accepted secant must preserve exact causal zeros")
    metadata = dict(
        denominator=denominator,
        residual=residual,
        theta=theta,
        correction_norm=correction_norm,
        row_norm=row_norm,
        updated=updated & (correction_norm > 0),
    )
    return result, metadata


def _problem(commands, safety, squared, Js, Jq, previous, lower, upper, pose_ceiling, radius, mode, config):
    metrics = _metrics(safety, squared, config)
    weights = np.ones(4369)
    weights[-1] = config["terminal_time_scale"]
    reserve_floor = min(0.5 * metrics["reserve"], 0.5) if mode == "pose" else 0.0
    floor = config["safety_buffer"] + reserve_floor * config["reserve_target"] / weights
    identity = np.eye(72)
    difference, center = identity - np.eye(72, k=-6), np.r_[previous, np.zeros(66)]
    command_lower, command_upper = np.maximum(lower, commands - radius), np.minimum(upper, commands + radius)
    blocks = [
        ("safety", Js, np.zeros(4369), safety - floor),
        ("pose_ceiling", -Jq, np.zeros(24), pose_ceiling - squared),
    ]
    if mode == "restore":
        blocks.append(
            (
                "mode_epigraph",
                weights[:, None] * Js / config["reserve_target"],
                np.ones(4369),
                weights * (safety - config["safety_buffer"]) / config["reserve_target"] - 1.0,
            )
        )
    else:
        blocks.append(("mode_epigraph", -Jq, np.ones(24), 1.0 - squared))
    blocks.extend(
        [
            ("rate_lower", difference, np.zeros(72), 0.25 + difference @ commands.ravel() - center),
            ("rate_upper", -difference, np.zeros(72), 0.25 - difference @ commands.ravel() + center),
            ("command_lower", identity, np.zeros(72), (commands - command_lower).ravel()),
            ("command_upper", -identity, np.zeros(72), (command_upper - commands).ravel()),
            ("epigraph", np.zeros((1, 72)), np.ones(1), np.zeros(1)),
        ]
    )
    A = np.vstack([np.column_stack((matrix, t)) for _, matrix, t, _ in blocks])
    b = np.concatenate([offset for _, _, _, offset in blocks])
    return identity, np.zeros(72), A, b


def _admission(commands, anchor, previous, input_min, input_max, lower, upper, radius, seen):
    gates = dict(
        box=bool(np.all(commands >= input_min - 1e-12) and np.all(commands <= input_max + 1e-12)),
        rate=bool(np.all(np.abs(np.diff(np.vstack((previous, commands)), axis=0)) <= 0.25 + 1e-12)),
        trust=bool(np.max(np.abs(commands - anchor)) <= radius + 1e-12),
        nonzero=bool(np.any(commands != anchor)),
        unique=commands.tobytes() not in seen,
    )
    gates["global"] = bool(np.all(commands >= lower - 1e-12) and np.all(commands <= upper + 1e-12))
    return gates


def _update_binding(name, expected, saved, compare):
    require(isinstance(saved, dict) and saved.keys() == expected.keys(), name + ": complete rowwise diagnostics")
    for key, value in expected.items():
        if key == "updated":
            stored = np.asarray(saved[key])
            require(stored.dtype.kind == "b" and np.array_equal(stored, value), name + ": exact boolean row flags")
        elif key in ("denominator", "theta"):
            exact_array(name + "." + key, value, saved[key])
        else:
            compare(name + "." + key, value, saved[key])


def _request(saved, index, commands, *, role, stage=None, alpha_index=None, alpha=None):
    for name, value in (
        ("request_index", index),
        ("response_index", index),
        ("calls_before", index),
        ("calls_after", index + 1),
    ):
        integer(saved[name], value, "charged request " + name)
    for name, value in (("role", role), ("stage_index", stage), ("alpha_index", alpha_index), ("alpha", alpha)):
        require(type(saved[name]) is type(value) and saved[name] == value, "Exact request " + name)
    boolean(saved["completed"], True, "Every archived call completed")
    boolean(saved["charged"], True, "Repeated requests are billed")
    exact_array("charged command bytes", commands, saved["commands"])


def audit_search(
    answer,
    initial_Js,
    initial_Jq,
    historical_initial,
    previous,
    input_min,
    input_max,
    config,
    compare=numerical_compare,
):
    """Audit the entire saved state machine; scientific search failure may pass."""
    _tree("exact registered config", config, answer["config"], compare)
    require(
        config["seed"] == 42 and config["max_responses"] >= 2 and config["max_stages"] >= 1,
        "Explicit seed and finite response/stage budgets",
    )
    previous, input_min, input_max = (
        array(value, name, (6,))
        for value, name in ((previous, "previous"), (input_min, "input_min"), (input_max, "input_max"))
    )
    require(np.all(input_min < input_max), "Strict fixed command box")
    Js, Jq = (
        array(initial_Js, "original safety Jacobian", (4369, 72)).copy(),
        array(initial_Jq, "original pose Jacobian", (24, 72)).copy(),
    )
    safety_periods, pose_periods = _periods()
    for matrix, periods in ((Js, safety_periods), (Jq, pose_periods)):
        future = np.repeat(np.arange(12), 6)[None, :] > periods[:, None]
        require(np.all(matrix[future] == 0), "Original stencil has exact causal future zeros")
    entries, requests, stages = answer["evaluated_candidates"], answer["request_ledger"], answer["stages"]
    count = len(entries)
    require(
        2 <= count <= config["max_responses"] and len(requests) == count, "All complete calls plus mandatory repeat"
    )
    require(len(stages) <= config["max_stages"], "Finite stage/QP budget")
    initial = array(historical_initial["commands"], "fixed historical initial command", (12, 6))
    require(
        historical_initial["audit"]["feasible"] is True
        and np.all(
            array(historical_initial["safety_slack"], "historical original safety", (4369,)) >= config["safety_buffer"]
        ),
        "Historical start must pass original safety and the unchanged buffer",
    )
    lower = np.maximum(np.tile(input_min, (12, 1)), initial - config["global_displacement_limit"])
    upper = np.minimum(np.tile(input_max, (12, 1)), initial + config["global_displacement_limit"])
    lower[0], upper[0] = np.maximum(lower[0], previous - 0.25), np.minimum(upper[0], previous + 0.25)
    exact_array("fixed global lower", lower, answer["global_lower"])
    exact_array("fixed global upper", upper, answer["global_upper"])
    pose_ceiling = np.maximum(_pose(historical_initial), 1.0) * (1.0 + config["pose_ceiling_fraction"])
    exact_array("nonaccumulating original pose ceiling", pose_ceiling, answer["pose_ceiling"])
    safety, squared, all_metrics = [], [], []
    for index, entry in enumerate(entries):
        command = array(entry["commands"], "complete recorded command", (12, 6))
        s, q = _response(entry, command, previous, input_min, input_max, compare, index)
        require(s.shape == (4369,), "Original full H12 safety layout")
        _pose(entry)
        safety.append(s)
        squared.append(q.ravel())
        all_metrics.append(_metrics(s, q.ravel(), config))
    _request(requests[0], 0, initial, role="initial")
    exact_array("initial full response command", initial, entries[0]["commands"])
    _request(requests[-1], count - 1, initial, role="initial_repeat")
    exact_array("forced repeat response command", initial, entries[-1]["commands"])
    fields = (
        "commands",
        "rollout",
        "audit",
        "tcp_errors",
        "normalized_squared_errors",
        "safety_slack",
        "prefix_length",
        "maximum_normalized_squared_error",
        "maximum_pose_cap_violation",
        "pose_loss",
        "feasible",
    )
    require(set(fields) <= set(historical_initial), "Full eleven-field historical comparison")
    historical = compare_initial_records({key: historical_initial[key] for key in fields}, entries[0])
    repeat = compare_initial_records(entries[0], entries[-1])
    _tree("complete strict historical comparison", historical, answer["initial_comparison"], exact_array)
    _tree("complete strict mandatory repeat comparison", repeat, answer["repeatability"], exact_array)

    def full_witness(index):
        return bool(entries[index]["audit"]["feasible"] and np.all(squared[index] <= 1))

    def original_safe(index):
        return bool(entries[index]["audit"]["feasible"] and np.all(safety[index] >= config["safety_buffer"]))

    initial_qualified = original_safe(0) and full_witness(0)
    if not historical["exact"] or initial_qualified:
        require(not stages and count == 2, "Historical failure/initial witness cannot launch a search")
    anchor, calls = 0, 1
    seen = {initial.tobytes()}
    radius = config["radius_initial"]
    total_iterations, stage_checks, accepted_indices = 0, [], []
    terminating_stage = None
    for si, stage in enumerate(stages):
        require(
            historical["exact"] and not initial_qualified and terminating_stage is None,
            "No stage after an already terminal condition",
        )
        require(
            calls < config["max_responses"] - 1 and radius >= config["radius_min"],
            "Search entry retains repeat budget and radius",
        )
        integer(stage["stage_index"], si, "Sequential stage index")
        integer(stage["anchor_before_index"], anchor, "Actual accepted anchor before stage")
        compare("stage radius before", radius, stage["radius_before"])
        before = all_metrics[anchor]
        _tree("actual anchor metrics before", before, stage["anchor_metrics_before"], compare)
        mode = "restore" if before["reserve"] < config["reserve_entry_fraction"] else "pose"
        require(stage["mode"] == mode, "Actual reserve determines stage mode")
        x = array(entries[anchor]["commands"], "current actual anchor", (12, 6))
        B, g, A, b = _problem(
            x, safety[anchor], squared[anchor], Js, Jq, previous, lower, upper, pose_ceiling, radius, mode, config
        )
        certificate = audit_qp_certificate(stage["qp"], B, g, A, b)
        total_iterations += stage["qp"]["solver_nit"]
        attempts = stage["attempts"]
        require(len(attempts) <= len(config["alphas"]), "Fixed finite backtracking ray")
        accepted, accepted_ratio, accepted_step = None, None, None
        mode_key = "violation" if mode == "restore" else "pose_excess"
        predicted_floor, actual_floor = config["minimum_predicted_reduction"], config["minimum_actual_reduction"]
        if mode == "pose":
            predicted_floor = min(predicted_floor, config["pose_reduction_fraction"] * before["pose_excess"])
            actual_floor = min(actual_floor, config["pose_reduction_fraction"] * before["pose_excess"])
        expected_termination = "qp_unqualified" if not certificate["qualified"] else "all_attempts_rejected"
        if not certificate["qualified"]:
            require(not attempts, "Unqualified local QP cannot issue a ray")
        else:
            direction = array(stage["qp"]["point"], "qualified local point", (73,))[:72].reshape(12, 6)
            require(attempts, "A qualified local QP must inspect its fixed ray")
            for ai, attempt in enumerate(attempts):
                require(accepted is None, "No more attempts after actual acceptance")
                alpha = config["alphas"][ai]
                integer(attempt["alpha_index"], ai, "Fixed alpha order")
                require(type(attempt["alpha"]) is type(alpha) and attempt["alpha"] == alpha, "Registered alpha")
                command = x + alpha * direction
                actual_step = (command - x).ravel()
                exact_array("unclipped fixed ray command", command, attempt["commands"])
                exact_array(
                    "actual represented ray step",
                    actual_step,
                    array(attempt["actual_step"], "saved actual step").ravel(),
                )
                admission = _admission(command, x, previous, input_min, input_max, lower, upper, radius, seen)
                _tree("actual ray admission", admission, attempt["admission"], compare)
                predicted = _metrics(safety[anchor] + Js @ actual_step, squared[anchor] + Jq @ actual_step, config)
                predicted_reduction = before[mode_key] - predicted[mode_key]
                _tree("actual-step linear prediction", predicted, attempt["predicted_metrics"], compare)
                compare("predicted mode reduction", predicted_reduction, attempt["predicted_reduction"])
                reason = None
                if not all(admission.values()):
                    reason = "command_preflight_failed"
                elif predicted_reduction < predicted_floor:
                    reason = "predicted_reduction_insufficient"
                elif calls >= config["max_responses"] - 1:
                    reason = "response_budget_exhausted"
                if reason is not None:
                    require(
                        attempt["response_index"] is None
                        and attempt["actual_metrics"] is None
                        and attempt["actual_reduction"] is None
                        and attempt["ratio"] is None,
                        "Denied candidate has no invented numerical response",
                    )
                    boolean(attempt["accepted"], False, "Denied candidate cannot become an anchor")
                else:
                    ri = calls
                    require(ri < count - 1, "Every actual trial is present before forced repeat")
                    integer(attempt["response_index"], ri, "Complete actual response routing")
                    _request(requests[ri], ri, command, role="trial", stage=si, alpha_index=ai, alpha=alpha)
                    exact_array("trial actual entry command", command, entries[ri]["commands"])
                    seen.add(command.tobytes())
                    calls += 1
                    actual = all_metrics[ri]
                    actual_reduction = before[mode_key] - actual[mode_key]
                    ratio = actual_reduction / predicted_reduction
                    _tree("actual response metrics", actual, attempt["actual_metrics"], compare)
                    compare("actual mode reduction", actual_reduction, attempt["actual_reduction"])
                    compare("actual over predicted reduction", ratio, attempt["ratio"])
                    if full_witness(ri) and original_safe(ri):
                        reason = "original_witness"
                    elif not original_safe(ri) or np.any(squared[ri] > pose_ceiling):
                        reason = "actual_admission_failed"
                    elif actual_reduction < actual_floor:
                        reason = "actual_reduction_insufficient"
                    elif ratio < config["accept_ratio"]:
                        reason = "ratio_insufficient"
                    else:
                        reason = "accepted_reduction"
                    accepted_here = reason in ("original_witness", "accepted_reduction")
                    boolean(attempt["accepted"], accepted_here, "Strict actual acceptance decision")
                    if accepted_here:
                        accepted, accepted_ratio, accepted_step = ri, ratio, actual_step
                        expected_termination = "qualified_witness" if reason == "original_witness" else "accepted"
                require(attempt["acceptance_reason"] == reason, "Ordered acceptance/rejection reason")
                if reason == "response_budget_exhausted":
                    expected_termination = reason
                    require(ai == len(attempts) - 1, "Budget exhaustion ends this ray")
            if accepted is None and expected_termination != "response_budget_exhausted":
                require(
                    len(attempts) == len(config["alphas"]), "All fixed alphas retained when no candidate is accepted"
                )
        _tree("actual accepted response index", accepted, stage["accepted_response_index"], compare)
        require(stage["termination"] == expected_termination, "Exact stage terminal state")
        new_radius = radius
        if accepted is None:
            new_radius *= 0.5
            require(stage["secant_update"] is None, "No secant update without an actual accepted displacement")
        else:
            update = stage["secant_update"]
            require(
                isinstance(update, dict) and set(update) == {"step", "safety", "pose"},
                "Full two-model causal secant record",
            )
            exact_array("actual accepted secant step", accepted_step, array(update["step"], "secant step").ravel())
            new_Js, safety_update = _causal_update(
                Js, accepted_step, safety[accepted] - safety[anchor], safety_periods, config
            )
            new_Jq, pose_update = _causal_update(
                Jq, accepted_step, squared[accepted] - squared[anchor], pose_periods, config
            )
            _update_binding("safety causal secant", safety_update, update["safety"], compare)
            _update_binding("pose causal secant", pose_update, update["pose"], compare)
            Js, Jq = new_Js, new_Jq
            if accepted_ratio < config["shrink_ratio"]:
                new_radius *= 0.5
            elif accepted_ratio >= config["expand_ratio"] and np.max(np.abs(accepted_step)) >= 0.8 * radius:
                new_radius = min(2.0 * radius, config["radius_max"])
            anchor = accepted
            accepted_indices.append(accepted)
        compare("actual radius update", new_radius, stage["radius_after"])
        radius = new_radius
        integer(stage["anchor_after_index"], anchor, "Actual anchor after stage")
        _tree("actual anchor metrics after", all_metrics[anchor], stage["anchor_metrics_after"], compare)
        if expected_termination == "qualified_witness":
            terminating_stage = "qualified_witness"
        elif radius < config["radius_min"]:
            terminating_stage = "radius_below_minimum"
        elif expected_termination == "response_budget_exhausted" or (
            calls >= config["max_responses"] - 1 and si + 1 < config["max_stages"]
        ):
            terminating_stage = "response_budget_exhausted"
        stage_checks.append(
            dict(stage_index=si, mode=mode, qp=certificate, accepted_response_index=accepted, radius_after=radius)
        )
    require(calls == count - 1, "No orphan, omitted, cached, or duplicated billed response")
    if not historical["exact"]:
        termination = "historical_initial_mismatch"
    elif initial_qualified:
        termination = "initial_witness"
    elif terminating_stage is not None:
        termination = terminating_stage
    elif calls >= config["max_responses"] - 1 and len(stages) < config["max_stages"]:
        termination = "response_budget_exhausted"
    else:
        require(len(stages) == config["max_stages"], "No unaccounted early search exit")
        termination = "stage_budget_exhausted"
    if not repeat["exact"]:
        termination = "repeat_mismatch"
    require(answer["termination_reason"] == termination, "Exact bounded search termination")
    integer(answer["final_anchor_index"], anchor, "Final numerical anchor remains explicit")
    integer(answer["response_evaluations"], count, "Every complete response billed")
    integer(answer["prediction_calls"], count, "Every actual callback including repeat charged")
    integer(answer["unique_command_count"], len(seen), "Distinct commands include every rejected actual trial")
    integer(answer["total_stages"], len(stages), "Every stage/QP counted")
    integer(answer["total_qp_iterations"], total_iterations, "Actual saved QP iteration total")
    boolean(answer["control_qualified"], False, "Local search grants no physical control qualification")
    witnesses = [i for i in range(count) if full_witness(i)]
    qualified = [i for i in witnesses if original_safe(i)] if historical["exact"] and repeat["exact"] else []
    _tree("all original full witnesses", witnesses, answer["witness_indices"], compare)
    _tree("buffer/history/repeat qualified full witnesses", qualified, answer["qualified_witness_indices"], compare)
    selected = qualified[0] if qualified else None
    _tree("only a full qualified witness may be selected", selected, answer["selected_response_index"], compare)
    if selected is None:
        require(answer["selected"] is None, "A numerical anchor cannot masquerade as an executable witness")
    else:
        _tree("full selected witness", entries[selected], answer["selected"], compare)
        exact_array("selected command bytes", entries[selected]["commands"], answer["selected"]["commands"])
    return dict(
        passed=True,
        response_evaluations=count,
        total_stages=len(stages),
        total_qp_iterations=total_iterations,
        accepted_response_indices=accepted_indices,
        stage_checks=stage_checks,
        initial_exact=historical["exact"],
        repeat_exact=repeat["exact"],
        termination_reason=termination,
        final_anchor_index=anchor,
        witness_indices=witnesses,
        qualified_witness_indices=qualified,
        selected_response_index=selected,
        full_responses_independently_replayed_here=False,
        control_qualified=False,
    )
