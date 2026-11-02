"""Independent bookkeeping and arithmetic audit of a fixed direction grid.

The caller must independently replay every billed response, including the final
anchor repeat. This module checks saved values and their routing, not dynamics.
It imports no producer, optimizer, simulator, or response evaluator.
"""

import numpy as np

STEPS = (1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 5e-6, 1e-6, 5e-7, 1e-7, 5e-8, 1e-8)
STAGES = (24, 25, 26)
SAFETY_GROUPS = ("inner_lower", "inner_upper", "braking_lower", "braking_upper", "terminal")
COMMAND_GROUPS = ("box_lower", "box_upper", "rate_lower", "rate_upper")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def array(value, name, shape=None):
    require(not np.iscomplexobj(value), f"{name}: real array required")
    try:
        result = np.asarray(value, dtype=np.float64)
    except (ValueError, TypeError, OverflowError) as error:
        raise ValueError(f"{name}: numeric array required") from error
    require(shape is None or result.shape == shape, f"{name}: shape mismatch")
    require(np.isfinite(result).all(), f"{name}: finite array required")
    return result


def numerical_compare(name, expected, saved):
    expected = array(expected, f"expected {name}")
    saved = array(saved, name, expected.shape)
    require(np.all(np.abs(saved - expected) <= 1e-12 + 1e-10 * np.abs(expected)), f"{name}: numeric mismatch")


def exact_array(name, expected, saved):
    expected = array(expected, f"expected {name}")
    saved = array(saved, name, expected.shape)
    require(expected.tobytes() == saved.tobytes(), f"{name}: exact array mismatch")


def integer(value, expected, name):
    require(type(value) is int and value == expected, f"{name}: exact integer required")


def boolean(value, expected, name):
    require(type(value) is bool and value is bool(expected), f"{name}: exact boolean required")


def minimum(value):
    return float(np.min(value)) if value.size else None


def optional_number(compare, name, expected, saved):
    if expected is None:
        require(saved is None, f"{name}: expected absent value")
    else:
        compare(name, expected, saved)


def admission(commands, previous, lower, upper):
    increments = np.diff(np.vstack((previous, commands)), axis=0)
    values = dict(
        box_lower=commands - lower,
        box_upper=upper - commands,
        rate_lower=0.25 + increments,
        rate_upper=0.25 - increments,
    )
    reasons = [name for name, value in values.items() if np.any(value < -1e-12)]
    return dict(admitted=not reasons, tolerance=1e-12, **values, reasons=reasons)


def _response(entry, command, previous, lower, upper, compare, index):
    name = f"response[{index}]"
    exact_array(name + ".commands", command, entry["commands"])
    h = len(command)
    errors = array(entry["tcp_errors"], name + ".tcp_errors", (h, 6))
    squared = np.column_stack(
        (np.sum(errors[:, :3] ** 2, axis=1) / 0.01**2, np.sum(errors[:, 3:] ** 2, axis=1) / 0.05**2)
    )
    compare(name + ".normalized_squared_errors", squared, entry["normalized_squared_errors"])
    failed = np.flatnonzero(np.any(squared > 1.0, axis=1))
    prefix = int(failed[0]) if failed.size else h
    integer(entry["prefix_length"], prefix, name + ".prefix_length")
    compare(
        name + ".maximum_normalized_squared_error", float(np.max(squared)), entry["maximum_normalized_squared_error"]
    )
    compare(name + ".pose_loss", float(np.sum(squared)), entry["pose_loss"])
    compare(
        name + ".maximum_pose_cap_violation",
        float(max(0.0, np.max(np.sqrt(squared)) - 1.0)),
        entry["maximum_pose_cap_violation"],
    )
    audit = entry["audit"]
    integer(audit["horizon"], h, name + ".audit.horizon")
    n = audit["joint_count"]
    require(type(n) is int and n > 0, name + ": positive joint count")
    require(set(audit["groups"]) == set(COMMAND_GROUPS + SAFETY_GROUPS), name + ": complete audit groups")
    require(audit["activity_tolerance"] == 1e-8, name + ": fixed audit activity threshold")
    raw_commands = admission(command, previous, lower, upper)
    slacks, feasible = {}, True
    for group_name in COMMAND_GROUPS + SAFETY_GROUPS:
        group = audit["groups"][group_name]
        if group_name in COMMAND_GROUPS:
            shape, tolerance, first = (h, 6), 1e-12, 1
        elif group_name.startswith("inner"):
            shape, tolerance, first = (25 * h, n), 1e-10, 25
        elif group_name.startswith("braking"):
            shape, tolerance, first = (h, n), 1e-10, 1
        else:
            shape, tolerance, first = None, 1e-10, None
        slack = array(group["slack"], name + "." + group_name, shape)
        require(slack.ndim == (1 if first is None else 2), name + ": audit slack rank")
        if group_name in COMMAND_GROUPS:
            compare(name + ".audit." + group_name, raw_commands[group_name], slack)
        require(group["tolerance"] == tolerance, name + ": fixed original audit tolerance")
        active, violated = np.argwhere(slack <= 1e-8), np.argwhere(slack < -tolerance)
        for field, expected in (("active_indices", active), ("violated_indices", violated)):
            # JSON loses the trailing dimension of an empty index array.
            saved = np.asarray(group[field])
            require(
                (expected.size == saved.size == 0) or np.array_equal(saved, expected),
                name + "." + group_name + "." + field,
            )
        optional_number(compare, name + ".minimum", minimum(slack), group["minimum"])
        feasible = feasible and not len(violated)
        if first is not None:
            optional_number(
                compare, name + ".first_minimum", minimum(slack[:first]), audit["first_period_minima"][group_name]
            )
            optional_number(
                compare, name + ".future_minimum", minimum(slack[first:]), audit["future_period_minima"][group_name]
            )
        slacks[group_name] = slack
    boolean(audit["feasible"], feasible, name + ".audit.feasible")
    safety = np.concatenate([slacks[group].ravel() for group in SAFETY_GROUPS])
    compare(name + ".safety_slack", safety, entry["safety_slack"])
    boolean(entry["feasible"], feasible and prefix == h, name + ".feasible")
    # Physical values and TCP geometry are supplied to the separate replay audit.
    rollout = entry["rollout"]
    for key, shape in (
        ("inner_joint_position", (25 * h, n)),
        ("inner_joint_velocity", (25 * h, n)),
        ("joint_position", (h, n)),
        ("joint_velocity", (h, n)),
        ("pose", (h, 4, 4)),
        ("twist", (h, 6)),
    ):
        array(rollout[key], name + ".rollout." + key, shape)
    return safety, squared


def _difference(compare, name, saved, a, b, denominator, reason):
    boolean(saved["valid"], reason is None, name + ".valid")
    require(saved["reason"] == reason, name + ": derivative availability reason")
    for index, field in enumerate(("safety_slack", "normalized_squared_errors")):
        if reason is not None:
            require(saved[field] is None, name + ": unavailable derivative cannot invent values")
        else:
            compare(name + "." + field, (a[index] - b[index]) / denominator, saved[field])


def _json_values(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _json_values(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_values(item) for item in value]
    return value


def _numeric_list(value):
    if not isinstance(value, list):
        return None
    try:
        result = np.asarray(value)
    except (TypeError, ValueError):
        return None
    return result if result.dtype.kind in "biuf" else None


def _audit_repeat(answer, responses, compare):
    del compare  # Repeat equality and the saved difference report are exact.
    stack = [("", _json_values(responses[0]), _json_values(responses[-1]))]
    arrays, structure, scalars = {}, True, True
    while stack:
        path, left, right = stack.pop()
        a, b = _numeric_list(left), _numeric_list(right)
        if a is not None or b is not None:
            compatible = a is not None and b is not None and a.shape == b.shape
            structure = structure and compatible
            difference = None
            if compatible and np.isfinite(a).all() and np.isfinite(b).all():
                difference = float(np.max(np.abs(np.subtract(a.astype(float), b.astype(float))))) if a.size else 0.0
            arrays[path] = dict(
                shape=None if a is None else list(a.shape),
                repeated_shape=None if b is None else list(b.shape),
                exact=bool(compatible and np.array_equal(a, b)),
                max_abs_difference=difference,
            )
        elif isinstance(left, dict) and isinstance(right, dict):
            structure = structure and left.keys() == right.keys()
            stack.extend(
                (f"{path}.{key}" if path else key, left[key], right[key]) for key in left.keys() & right.keys()
            )
        elif isinstance(left, list) and isinstance(right, list):
            structure = structure and len(left) == len(right)
            stack.extend((f"{path}[{index}]", a, b) for index, (a, b) in enumerate(zip(left, right)))
        else:
            structure = structure and not isinstance(left, (dict, list)) and not isinstance(right, (dict, list))
            scalars = scalars and type(left) is type(right) and left == right
    exact = bool(structure and scalars and all(item["exact"] for item in arrays.values()))
    expected = dict(
        initial_response_index=0,
        repeated_response_index=len(responses) - 1,
        comparison_scope="Canonical JSON numeric arrays and scalars; no dtype or hidden empty-array dimension claim",
        exact=exact,
        structure_equal=bool(structure),
        scalars_equal=bool(scalars),
        arrays=arrays,
        max_abs_difference=max(
            (item["max_abs_difference"] for item in arrays.values() if item["max_abs_difference"] is not None),
            default=0.0,
        ),
    )
    require(answer["repeatability"] == expected, "Complete exact per-array anchor repeat comparison")
    boolean(answer["derivative_reliability_limited"], not exact, "Repeat limits derivative reliability")
    return expected


def audit_directional_derivative_ledger(answer, *, compare=numerical_compare):
    """Check the fixed 80-request protocol using already replayed saved values."""
    require(answer["method"] == "fixed_directional_derivative_probe", "Fixed diagnostic method")
    h = answer["horizon"]
    require(type(h) is int and 1 <= h <= 16, "Explicit bounded horizon")
    anchor = array(answer["anchor_commands"], "anchor_commands", (h, 6))
    previous = array(answer["previous_command"], "previous_command", (6,))
    lower, upper = array(answer["input_min"], "input_min", (6,)), array(answer["input_max"], "input_max", (6,))
    require(np.all(lower < upper), "Strict command bounds")
    require(answer["source_stages"] == list(STAGES), "Fixed saved stages")
    exact_array("step_sizes", STEPS, answer["step_sizes"])
    directions = array(answer["directions"], "directions", (3, h, 6))
    scales = np.max(np.abs(directions), axis=(1, 2))
    require(np.all(scales > 0), "Nonzero saved directions")
    units = directions / scales[:, None, None]
    exact_array("direction_scales", scales, answer["direction_scales"])
    exact_array("unit_directions", units, answer["unit_directions"])
    linf = np.max(np.abs(units[:, None] - units[None, :]), axis=(2, 3))
    flat = units.reshape(3, -1)
    norms = np.linalg.norm(flat, axis=1)
    cosine = (flat @ flat.T) / (norms[:, None] * norms[None, :])
    compare("pairwise_direction_linf", linf, answer["pairwise_direction_linf"])
    compare("pairwise_direction_cosine", cosine, answer["pairwise_direction_cosine"])
    integer(answer["planned_request_count"], 80, "planned_request_count")
    integer(answer["max_responses"], 80, "max_responses")
    integer(answer["total_iterations"], 0, "No optimization")
    boolean(answer["control_qualified"], False, "No control qualification")
    requests, responses = answer["request_ledger"], answer["evaluated_candidates"]
    require(len(requests) == 80, "All planned requests, including final repeat")
    plan = [("anchor", None, None, None, None, anchor.copy(), np.zeros_like(anchor))]
    for direction_index, stage in enumerate(STAGES):
        for step_index, step in enumerate(STEPS):
            for sign in (1, -1):
                nominal = sign * step * units[direction_index]
                plan.append(("grid", direction_index, step_index, stage, sign, anchor + nominal, nominal))
    plan.append(("anchor_repeat", None, None, None, None, anchor.copy(), np.zeros_like(anchor)))
    cache, calls, values = {}, 0, []
    for index, (saved, specification) in enumerate(zip(requests, plan, strict=True)):
        role, di, si, stage, sign, command, nominal = specification
        integer(saved["request_index"], index, "request_index")
        require(saved["role"] == role, "Fixed request role/order")
        for field, expected in (("direction_index", di), ("step_index", si), ("source_stage", stage), ("sign", sign)):
            require(saved[field] == expected and type(saved[field]) is type(expected), "Exact request " + field)
        step = None if si is None else STEPS[si]
        require(saved["step"] == step, "Exact nominal step")
        exact_array("request.commands", command, saved["commands"])
        key = command.tobytes().hex()
        require(saved["command_key"] == key, "Exact request command bytes")
        actual = command - anchor
        error = float(np.max(np.abs(actual - nominal)))
        exact_array("actual_displacement", actual, saved["actual_displacement"])
        exact_array("nominal_displacement", nominal, saved["nominal_displacement"])
        compare("displacement_error_inf", error, saved["displacement_error_inf"])
        optional_number(
            compare,
            "relative_displacement_error_inf",
            None if step is None else error / step,
            saved["relative_displacement_error_inf"],
        )
        boolean(saved["zero_displacement"], not np.any(actual != 0), "zero_displacement")
        admitted = admission(command, previous, lower, upper)
        stored = saved["admission"]
        boolean(stored["admitted"], admitted["admitted"], "command admission")
        require(
            stored["tolerance"] == 1e-12 and stored["reasons"] == admitted["reasons"],
            "Original command admission reasons",
        )
        for group in COMMAND_GROUPS:
            compare("admission." + group, admitted[group], stored[group])
        require(role != "anchor" or admitted["admitted"], "Initial anchor must be admissible")
        integer(saved["calls_before"], calls, "calls_before")
        if not admitted["admitted"]:
            outcome, charged, response_index = "command_rejected", False, None
        elif role != "anchor_repeat" and key in cache:
            outcome, charged, response_index = "cached", False, cache[key]
        else:
            outcome, charged, response_index = ("repeated" if role == "anchor_repeat" else "computed"), True, calls
            require(calls < len(responses), "Every billed response is retained")
            values.append(_response(responses[calls], command, previous, lower, upper, compare, calls))
            require(values[-1][0].shape == values[0][0].shape, "Unchanged full safety-vector row count")
            require(
                responses[calls]["audit"]["joint_count"] == responses[0]["audit"]["joint_count"],
                "Unchanged audit joint count",
            )
            calls += 1
            if role != "anchor_repeat":
                cache[key] = response_index
        require(saved["outcome"] == outcome, "Request cache/billing outcome")
        boolean(saved["charged"], charged, "charged")
        require(
            saved["response_index"] == response_index and type(saved["response_index"]) is type(response_index),
            "Exact response routing",
        )
        repeat_of = 0 if role == "anchor_repeat" else None
        require(
            saved["repeat_of"] == repeat_of and type(saved["repeat_of"]) is type(repeat_of),
            "Only final anchor is a forced repeat",
        )
        integer(saved["calls_after"], calls, "calls_after")
    require(calls == len(responses) <= 80 and len(cache) <= 79, "Complete bounded billed responses")
    integer(answer["response_evaluations"], calls, "response_evaluations")
    integer(answer["unique_command_count"], len(cache), "unique_command_count")
    differences = answer["differences"]
    require(len(differences) == 39, "All directional derivative pairs")
    for pair_index, saved in enumerate(differences):
        di, si = divmod(pair_index, 13)
        step = STEPS[si]
        plus_index, minus_index = 1 + 2 * pair_index, 2 + 2 * pair_index
        for field, expected in (
            ("direction_index", di),
            ("source_stage", STAGES[di]),
            ("step_index", si),
            ("plus_request_index", plus_index),
            ("minus_request_index", minus_index),
        ):
            integer(saved[field], expected, "difference." + field)
        require(saved["step"] == step, "Difference nominal denominator")
        plus, minus = requests[plus_index], requests[minus_index]

        def reason(request):
            return (
                "command_rejected"
                if not request["admission"]["admitted"]
                else "zero_displacement"
                if request["zero_displacement"]
                else None
            )

        rp, rm = reason(plus), reason(minus)
        vp = values[plus["response_index"]] if plus["response_index"] is not None else None
        vm = values[minus["response_index"]] if minus["response_index"] is not None else None
        central_reason = (
            "command_rejected"
            if "command_rejected" in (rp, rm)
            else "zero_displacement"
            if "zero_displacement" in (rp, rm)
            else None
        )
        _difference(compare, "forward", saved["forward"], vp, values[0], step, rp)
        _difference(compare, "backward", saved["backward"], values[0], vm, step, rm)
        _difference(compare, "central", saved["central"], vp, vm, 2 * step, central_reason)
    # Repeat comparison is defined below without importing the producer.
    repeat_check = _audit_repeat(answer, responses, compare)
    return dict(
        passed=True,
        request_count=80,
        response_evaluations=calls,
        unique_command_count=len(cache),
        directional_pairs_checked=39,
        total_iterations=0,
        repeatability=repeat_check,
        full_responses_independently_replayed_here=False,
        source_archive_binding_checked_here=False,
        control_qualified=False,
    )


def _trace_values(record, name):
    errors = array(record["tcp_errors"], name + ".tcp_errors", (12, 6))
    squared = np.column_stack(
        (np.sum(errors[:, :3] ** 2, axis=1) / 0.01**2, np.sum(errors[:, 3:] ** 2, axis=1) / 0.05**2)
    )
    safety = array(record["safety_slack"], name + ".safety_slack")
    require(safety.ndim == 1 and safety.size > 0, "Full old/new safety vector")
    return safety, squared


def audit_saved_stencil_comparison(sqp_answer, answer, saved, *, compare=numerical_compare):
    """Bind old FD columns/directions and check their predictions on new calls.

    Source-file identity and both full response replays belong to the caller.
    The new 80-request ledger must also pass ``audit_directional_derivative_ledger``.
    This function never reconstructs an unsaved old complete rollout.
    """
    integer(sqp_answer["horizon"], 12, "Original H12")
    integer(answer["horizon"], 12, "New H12")
    require(sqp_answer["finite_difference_step"] == 1e-4, "Original fixed FD step")
    integer(saved["anchor_response_index"], 1819, "Fixed old anchor")
    require(saved["source_stages"] == list(STAGES), "Fixed old source stages")
    require(answer["source_stages"] == list(STAGES), "New direction stage binding")
    trace, requests = sqp_answer["evaluated_candidates"], sqp_answer["request_ledger"]
    require(len(trace) > 1819 and len(sqp_answer["phases"]) > 26, "Required old trace/stages exist")
    anchor = array(trace[1819]["commands"], "old anchor", (12, 6))
    exact_array("new anchor binding", anchor, answer["anchor_commands"])
    base_safety, base_squared = _trace_values(trace[1819], "old anchor")
    mapping = jacobians = None
    directions = []
    for stage_index in STAGES:
        phase = sqp_answer["phases"][stage_index]
        integer(phase["stage_index"], stage_index, "Old stage slot identity")
        integer(phase["anchor_before_index"], 1819, "Old stage anchor")
        exact_array("old stage initial commands", anchor, phase["initial_commands"])
        direction = array(phase["qp_direction"]["commands"], "old QP commands", (12, 6)) - anchor
        exact_array("old actual QP displacement", direction, phase["qp_direction"]["displacement"])
        directions.append(direction)
        start, end = phase["requests_before"], phase["generation_requests_after"]
        require(type(start) is int and type(end) is int and 0 <= start < end <= len(requests), "Old FD request slice")
        rows = requests[start:end]
        require(len(rows) == 74, "Two cached anchor requests and 72 ordered FD probes")
        lower = array(phase["stage_lower_command"], "old stage lower", (12, 6)).ravel()
        upper = array(phase["stage_upper_command"], "old stage upper", (12, 6)).ravel()
        require(np.all(lower <= anchor.ravel()) and np.all(anchor.ravel() <= upper), "Anchor inside old stage bounds")
        indices, deltas, safety_columns, pose_columns = [], [], [], []
        for offset, row in enumerate(rows):
            integer(row["request_index"], start + offset, "Original request slot")
            integer(row["stage_index"], stage_index, "Original request stage")
            if offset < 2:
                require(row["role"] == ("stage_anchor" if offset == 0 else "gradient_base"), "Cached anchor roles")
                require(row["outcome"] == "cache_hit", "Original base does not incur another response")
                integer(row["response_index"], 1819, "Cached original anchor identity")
                require(row["command_key"] == anchor.tobytes().hex(), "Cached original anchor bytes")
                continue
            axis = offset - 2
            integer(row["axis"], axis, "Ordered original FD axis")
            require(row["role"] == "gradient_probe", "Original FD role")
            delta = 1e-4 if anchor.ravel()[axis] + 1e-4 <= upper[axis] else -1e-4
            require(row["nominal_delta"] == delta, "Original bound-aware FD nominal sign")
            command = anchor.ravel().copy()
            command[axis] += delta
            require(np.all(command >= lower - 1e-12) and np.all(command <= upper + 1e-12), "Original legal FD command")
            require(row["command_key"] == command.tobytes().hex(), "Original FD command bytes")
            response_index = row["response_index"]
            require(type(response_index) is int and 0 <= response_index < len(trace), "Original FD response index")
            require(row["outcome"] in {"new_response", "cache_hit"}, "Completed original FD response")
            probe = trace[response_index]
            exact_array("Original FD response command binding", command.reshape(12, 6), probe["commands"])
            safety, squared = _trace_values(probe, "old FD probe")
            require(safety.shape == base_safety.shape, "Original safety row identity")
            indices.append(response_index)
            deltas.append(delta)
            safety_columns.append((safety - base_safety) / delta)
            pose_columns.append((squared - base_squared).ravel() / delta)
        linearization = phase["linearization"]
        boolean(linearization["complete"], True, "Complete old stencil")
        integer(linearization["base_response_index"], 1819, "Old linearization anchor")
        require(linearization["probe_response_indices"] == indices, "Original linearization probe mapping")
        exact_array("Original linearization signed steps", deltas, linearization["nominal_deltas"])
        current = (indices, deltas)
        Js, Jq = np.asarray(safety_columns).T, np.asarray(pose_columns).T
        if mapping is None:
            mapping, jacobians = current, (Js, Jq)
        else:
            require(current == mapping, "All three stages use the identical original stencil")
            exact_array("Shared original safety Jacobian", jacobians[0], Js)
            exact_array("Shared original pose Jacobian", jacobians[1], Jq)
    directions = np.asarray(directions)
    scales = np.max(np.abs(directions), axis=(1, 2))
    require(np.all(scales > 0), "Nonzero original QP directions")
    units = directions / scales[:, None, None]
    exact_array("New raw directions bound to old QP commands", directions, answer["directions"])
    exact_array("New original-direction norms", scales, answer["direction_scales"])
    exact_array("New normalized original directions", units, answer["unit_directions"])
    Js, Jq = jacobians
    compare("Saved original safety Jacobian", Js, saved["jacobian_safety_slack"])
    compare("Saved original pose Jacobian", Jq, saved["jacobian_normalized_squared_errors"])
    predictions = saved["unit_predictions"]
    require(len(predictions) == 3, "All original unit-direction predictions")
    for stage, unit, prediction in zip(STAGES, units, predictions, strict=True):
        integer(prediction["source_stage"], stage, "Unit prediction stage")
        compare("Unit safety prediction", Js @ unit.ravel(), prediction["safety_slack"])
        compare("Unit pose prediction", (Jq @ unit.ravel()).reshape(12, 2), prediction["normalized_squared_errors"])
    new_responses, new_requests = answer["evaluated_candidates"], answer["request_ledger"]
    base_new = _trace_values(new_responses[0], "new initial response")
    require(base_new[0].shape == base_safety.shape, "Old/new full safety row count")
    predictions = saved["request_predictions"]
    require(len(predictions) == len(new_requests) == 80, "All planned requests have old-stencil predictions")
    for index, (request, prediction) in enumerate(zip(new_requests, predictions, strict=True)):
        integer(request["request_index"], index, "New request slot")
        integer(prediction["request_index"], index, "Prediction request slot")
        command = array(request["commands"], "new requested commands", (12, 6))
        displacement = command - anchor
        exact_array("Prediction uses actual floating-point displacement", displacement, request["actual_displacement"])
        predicted = (Js @ displacement.ravel(), (Jq @ displacement.ravel()).reshape(12, 2))
        response_index = request["response_index"]
        if response_index is None:
            require(
                prediction["observed_change"] is None and prediction["residual"] is None,
                "No fabricated observation for a skipped request",
            )
            observed = None
        else:
            require(type(response_index) is int and 0 <= response_index < len(new_responses), "New response routing")
            exact_array("New observed response command binding", command, new_responses[response_index]["commands"])
            response = _trace_values(new_responses[response_index], "new observed response")
            observed = tuple(value - base for value, base in zip(response, base_new, strict=True))
        for item, field in enumerate(("safety_slack", "normalized_squared_errors")):
            compare("Request predicted " + field, predicted[item], prediction["predicted_change"][field])
            if observed is not None:
                compare("Request observed " + field, observed[item], prediction["observed_change"][field])
                compare("Request residual " + field, observed[item] - predicted[item], prediction["residual"][field])
    return dict(
        passed=True,
        anchor_response_index=1819,
        source_stages=list(STAGES),
        original_fd_axes_checked=72,
        shared_stencils_checked=3,
        unit_predictions_checked=3,
        request_predictions_checked=80,
        full_responses_independently_replayed_here=False,
        source_file_identity_checked_here=False,
        old_complete_rollout_reconstructed=False,
        control_qualified=False,
    )
