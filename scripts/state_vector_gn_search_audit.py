"""Independent full-ledger audit of normalized TCP-vector Gauss-Newton search.

The objective, full constraints, causal models, and decisions are reconstructed
from archived responses. No search core, solver, or dynamics is imported.
"""

import numpy as np
from accepted_point_sqp_math_audit import audit_qp_certificate
from directional_derivative_audit import _response, array, boolean, exact_array, integer, numerical_compare, require
from interior_trajectory_search_audit import (
    _admission,
    _causal_update,
    _periods,
    _pose,
    _request,
    _update_binding,
)
from interior_trajectory_search_audit import (
    _metrics as _original_metrics,
)
from restored_anchor_sqp_ledger_audit import compare_initial_records
from stage_budget_search_audit import validate_config
from terminal_secant_audit import _tree

FEEDBACK_POLICY = "safety_qualified_pose_ceiling_rejection_minimum_alpha"
POSE_CEILING_POLICY = "fixed_initial_global_maximum"
POSE_OBJECTIVE_POLICY = "normalized_state_vector_gauss_newton"
POSE_DAMPING = 1.0


def _error(entry):
    """Raw normalized TCP residual components, preserving their signs."""
    error = array(entry["tcp_errors"], "full signed H12 TCP residual", (12, 6))
    return (error / np.array([0.01, 0.01, 0.01, 0.05, 0.05, 0.05])).ravel()


def reconstruct_error_vector_jacobian(bundle):
    """Independently read the original signed compact stencil, never a fitted Je."""
    base, probes = bundle["base"], bundle["probes"]
    initial = array(base["commands"], "original stencil anchor command", (12, 6))
    _pose(base)
    error = _error(base)
    nominal = array(bundle["nominal_deltas"], "original signed nominal stencil", (72,))
    require(len(probes) == 72 and np.all(np.abs(nominal) == 1e-4), "Original complete signed1e-4 stencil")
    columns = []
    for axis, probe in enumerate(probes):
        command = array(probe["commands"], "compact original probe command", (12, 6))
        change = command.ravel() - initial.ravel()
        tolerance = 4 * np.finfo(float).eps * max(1.0, abs(initial.ravel()[axis]), abs(command.ravel()[axis]))
        require(
            np.all(change[np.arange(72) != axis] == 0)
            and change[axis] != 0
            and abs(change[axis] - nominal[axis]) <= tolerance,
            "One original registered signed probe axis",
        )
        if "normalized_squared_errors" in probe:
            _pose(probe)
        columns.append((_error(probe) - error) / nominal[axis])
    matrix = array(np.asarray(columns).T, "independently reconstructed original Je", (72, 72))
    future = np.repeat(np.arange(12), 6)[None, :] > np.repeat(np.arange(12), 6)[:, None]
    require(np.all(matrix[future] == 0), "Original error stencil has exact future-command zeros")
    return matrix


def _metrics(safety, squared, config):
    result = _original_metrics(safety, squared, config)
    result["state_loss"] = float(0.5 * np.sum(squared))
    return result


def _problem(commands, safety, squared, error, Js, Jq, Je, previous, lower, upper, pose_ceiling, radius, mode, config):
    """Rebuild the GN objective and every unchanged hard row independently."""
    metrics = _metrics(safety, squared, config)
    weights = np.ones(4369)
    weights[-1] = config["terminal_time_scale"]
    reserve_floor = min(0.5 * metrics["reserve"], 0.5) if mode == "pose" else 0.0
    floor = config["safety_buffer"] + reserve_floor * config["reserve_target"] / weights
    identity = np.eye(72)
    difference, center = identity - np.eye(72, k=-6), np.r_[previous, np.zeros(66)]
    command_lower, command_upper = np.maximum(lower, commands - radius), np.minimum(upper, commands + radius)
    blocks = [
        (Js, np.zeros(4369), safety - floor),
        (-Jq, np.zeros(24), pose_ceiling - squared),
    ]
    if mode == "restore":
        blocks.append(
            (
                weights[:, None] * Js / config["reserve_target"],
                np.ones(4369),
                weights * (safety - config["safety_buffer"]) / config["reserve_target"] - 1.0,
            )
        )
        B, g = identity, np.zeros(72)
    else:
        require(mode == "pose", "Registered restore or normalized-vector pose objective")
        B, g = POSE_DAMPING * identity + Je.T @ Je, Je.T @ error
    blocks.extend(
        [
            (difference, np.zeros(72), 0.25 + difference @ commands.ravel() - center),
            (-difference, np.zeros(72), 0.25 - difference @ commands.ravel() + center),
            (identity, np.zeros(72), (commands - command_lower).ravel()),
            (-identity, np.zeros(72), (command_upper - commands).ravel()),
            (np.zeros((1, 72)), np.ones(1), np.zeros(1)),
        ]
    )
    A = np.vstack([np.column_stack((matrix, t)) for matrix, t, _ in blocks])
    b = np.concatenate([offset for _, _, offset in blocks])
    return B, g, A, b


def audit_search(
    answer,
    initial_Js,
    initial_Jq,
    initial_Je,
    historical_initial,
    previous,
    input_min,
    input_max,
    config,
    compare=numerical_compare,
):
    """Audit the entire saved state machine; scientific search failure may pass."""
    validate_config(config)
    _tree("exact registered config", config, answer["config"], compare)
    require(answer["pose_objective_policy"] == POSE_OBJECTIVE_POLICY, "Registered normalized-vector GN objective")
    require(type(answer["pose_damping"]) is float and answer["pose_damping"] == POSE_DAMPING, "Fixed GN damping1")
    require(answer["feedback_policy"] == FEEDBACK_POLICY, "Fixed rejected-response feedback policy")
    require(answer["pose_ceiling_policy"] == POSE_CEILING_POLICY, "Fixed initial global-maximum ceiling policy")
    require(
        config["seed"] == 42 and config["max_responses"] >= 2 and config["max_stages"] >= 1,
        "Explicit seed and finite response/stage budgets",
    )
    previous, input_min, input_max = (
        array(value, name, (6,))
        for value, name in ((previous, "previous"), (input_min, "input_min"), (input_max, "input_max"))
    )
    require(np.all(input_min < input_max), "Strict fixed command box")
    Js, Jq, Je = (
        array(initial_Js, "original safety Jacobian", (4369, 72)).copy(),
        array(initial_Jq, "original pose Jacobian", (24, 72)).copy(),
        array(initial_Je, "original normalized-error Jacobian", (72, 72)).copy(),
    )
    safety_periods, pose_periods = _periods()
    error_periods = np.repeat(np.arange(12), 6)
    for matrix, periods in ((Js, safety_periods), (Jq, pose_periods), (Je, error_periods)):
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
    initial_maximum = max(1.0, float(np.max(_pose(historical_initial))))
    pose_ceiling = np.full(24, initial_maximum * (1.0 + config["pose_ceiling_fraction"]))
    exact_array("fixed common initial-maximum pose envelope", pose_ceiling, answer["pose_ceiling"])
    safety, squared, error_vectors, all_metrics = [], [], [], []
    for index, entry in enumerate(entries):
        command = array(entry["commands"], "complete recorded command", (12, 6))
        s, q = _response(entry, command, previous, input_min, input_max, compare, index)
        require(s.shape == (4369,), "Original full H12 safety layout")
        _pose(entry)
        safety.append(s)
        squared.append(q.ravel())
        error_vectors.append(_error(entry))
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
    rejected_sources = []
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
            x,
            safety[anchor],
            squared[anchor],
            error_vectors[anchor],
            Js,
            Jq,
            Je,
            previous,
            lower,
            upper,
            pose_ceiling,
            radius,
            mode,
            config,
        )
        certificate = audit_qp_certificate(stage["qp"], B, g, A, b)
        total_iterations += stage["qp"]["solver_nit"]
        attempts = stage["attempts"]
        require(len(attempts) <= len(config["alphas"]), "Fixed finite backtracking ray")
        accepted, accepted_ratio, accepted_step = None, None, None
        mode_key = "violation" if mode == "restore" else "state_loss"
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
                predicted_error = error_vectors[anchor] + Je @ actual_step
                predicted["state_loss"] = float(0.5 * (predicted_error @ predicted_error))
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
                isinstance(update, dict) and set(update) == {"step", "safety", "pose", "error_vector"},
                "Full three-model causal secant record",
            )
            exact_array("actual accepted secant step", accepted_step, array(update["step"], "secant step").ravel())
            new_Js, safety_update = _causal_update(
                Js, accepted_step, safety[accepted] - safety[anchor], safety_periods, config
            )
            new_Jq, pose_update = _causal_update(
                Jq, accepted_step, squared[accepted] - squared[anchor], pose_periods, config
            )
            new_Je, error_update = _causal_update(
                Je, accepted_step, error_vectors[accepted] - error_vectors[anchor], error_periods, config
            )
            _update_binding("safety causal secant", safety_update, update["safety"], compare)
            _update_binding("pose causal secant", pose_update, update["pose"], compare)
            _update_binding("error-vector causal secant", error_update, update["error_vector"], compare)
            Js, Jq, Je = new_Js, new_Jq, new_Je
            if accepted_ratio < config["shrink_ratio"]:
                new_radius *= 0.5
            elif accepted_ratio >= config["expand_ratio"] and np.max(np.abs(accepted_step)) >= 0.8 * radius:
                new_radius = min(2.0 * radius, config["radius_max"])
            anchor = accepted
            accepted_indices.append(accepted)
        eligible = []
        if accepted is None:
            for attempt in attempts:
                ri = attempt["response_index"]
                if ri is None or attempt["accepted"] or attempt["acceptance_reason"] != "actual_admission_failed":
                    continue
                step = array(attempt["actual_step"], "actual rejected-response step", (12, 6)).ravel()
                if (
                    original_safe(ri)
                    and np.any(squared[ri] > pose_ceiling)
                    and np.any(step != 0)
                    and np.sum(step**2) >= config["secant_min_squared_step"]
                ):
                    eligible.append((attempt["alpha"], ri, step))
        chosen = min(eligible, key=lambda row: (row[0], row[1])) if eligible else None
        rejected_update = stage["rejected_update"]
        if chosen is None:
            require(rejected_update is None, "No rejected-response update outside the fixed eligible set")
        else:
            alpha, ri, step = chosen
            require(
                isinstance(rejected_update, dict)
                and set(rejected_update)
                == {"source_response_index", "alpha", "step", "safety", "pose", "error_vector"},
                "Complete rejected-response update",
            )
            integer(rejected_update["source_response_index"], ri, "Independently selected minimum-alpha safe rejection")
            require(
                type(rejected_update["alpha"]) is type(alpha) and rejected_update["alpha"] == alpha,
                "Fixed feedback source alpha",
            )
            exact_array(
                "actual rejected-response secant step",
                step,
                array(rejected_update["step"], "feedback step", (12, 6)).ravel(),
            )
            new_Js, safety_update = _causal_update(Js, step, safety[ri] - safety[anchor], safety_periods, config)
            new_Jq, pose_update = _causal_update(Jq, step, squared[ri] - squared[anchor], pose_periods, config)
            new_Je, error_update = _causal_update(
                Je, step, error_vectors[ri] - error_vectors[anchor], error_periods, config
            )
            _update_binding("rejected safety causal secant", safety_update, rejected_update["safety"], compare)
            _update_binding("rejected pose causal secant", pose_update, rejected_update["pose"], compare)
            _update_binding(
                "rejected error-vector causal secant", error_update, rejected_update["error_vector"], compare
            )
            Js, Jq, Je = new_Js, new_Jq, new_Je
            rejected_sources.append(ri)
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
    integer(
        answer["rejected_update_count"], len(rejected_sources), "All feedback updates counted without extra responses"
    )
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
        feedback_policy=FEEDBACK_POLICY,
        pose_ceiling_policy=POSE_CEILING_POLICY,
        pose_objective_policy=POSE_OBJECTIVE_POLICY,
        pose_damping=POSE_DAMPING,
        rejected_update_count=len(rejected_sources),
        rejected_response_indices=rejected_sources,
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
