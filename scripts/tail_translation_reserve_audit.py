"""Independent saved-value audit of the fixed final-period translation grid.

Only frozen numerical/ledger primitives are reused. The caller must bind source
archives and independently replay every billed trajectory; this module does not
run a response model or certify physical control.
"""

import numpy as np
from directional_derivative_audit import (
    _audit_repeat,
    _json_values,
    _numeric_list,
    _response,
    admission,
    array,
    boolean,
    exact_array,
    integer,
    numerical_compare,
    require,
)
from terminal_secant_audit import _clip_flags, _terminal, _tree

STEPS = (0.001, -0.001, 0.0005, -0.0005)
AXES = ("x", "y", "z")


def _admission(name, command, previous, lower, upper, saved, compare):
    expected = admission(command, previous, lower, upper)
    _tree(name, expected, saved, compare)
    require(expected["admitted"], name + ": complete unclipped grid admission")


def _displacement(name, command, anchor, nominal, saved, step, compare):
    actual = command - anchor
    for field, value in (("commands", command), ("actual_displacement", actual), ("nominal_displacement", nominal)):
        exact_array(name + "." + field, value, saved[field])
    require(saved["command_key"] == command.tobytes().hex(), name + ": exact float64 command key")
    error = float(np.max(np.abs(actual - nominal)))
    compare(name + ".displacement_error_inf", error, saved["displacement_error_inf"])
    _tree(
        name + ".relative_displacement_error_inf",
        None if step is None else error / abs(step),
        saved["relative_displacement_error_inf"],
        compare,
    )
    compare(name + ".actual_displacement_linf", float(np.max(np.abs(actual))), saved["actual_displacement_linf"])
    boolean(saved["zero_displacement"], not np.any(actual != 0), name + ".zero_displacement")


def _grid_specs(anchor):
    for axis in range(3):
        for si, step in enumerate(STEPS):
            command = anchor.copy()
            command[11, axis] += step
            nominal = np.zeros_like(anchor)
            nominal[11, axis] = step
            yield axis, si, command, nominal


def audit_tail_translation_grid(grid, *, compare=numerical_compare):
    """Rebuild all12 commands before any model is required by the caller."""
    integer(grid["horizon"], 12, "H12")
    integer(grid["tail_period_index"], 11, "last period")
    require(grid["axis_names"] == list(AXES), "Fixed translation axis order")
    exact_array("signed steps", STEPS, grid["signed_steps"])
    anchor = array(grid["anchor_commands"], "anchor", (12, 6))
    previous = array(grid["previous_command"], "previous", (6,))
    lower, upper = (array(grid[k], k, (6,)) for k in ("input_min", "input_max"))
    require(np.all(lower < upper), "Original nondegenerate box")
    _admission("anchor", anchor, previous, lower, upper, grid["anchor_admission"], compare)
    require(len(grid["grid_commands"]) == 12, "Complete fixed twelve-point grid")
    keys = {anchor.tobytes()}
    for index, (saved, spec) in enumerate(zip(grid["grid_commands"], _grid_specs(anchor), strict=True)):
        axis, si, command, nominal = spec
        step = STEPS[si]
        expected = dict(grid_index=index, axis_index=axis, axis_name=AXES[axis], step_index=si, signed_step=step)
        for key, value in expected.items():
            require(type(saved[key]) is type(value) and saved[key] == value, "grid fixed " + key)
        _displacement("grid", command, anchor, nominal, saved, step, compare)
        require(np.any(command != anchor) and command.tobytes() not in keys, "Thirteen unique commands")
        keys.add(command.tobytes())
        boolean(saved["unique_command"], True, "grid unique command")
        _admission("grid.admission", command, previous, lower, upper, saved["admission"], compare)
    boolean(grid["admitted"], True, "grid admitted")
    integer(grid["planned_request_count"], 14, "fourteen billed calls")
    integer(grid["expected_unique_command_count"], 13, "thirteen unique commands")
    return dict(
        passed=True,
        horizon=12,
        grid_commands_checked=12,
        planned_request_count=14,
        unique_command_count=13,
        source_archive_binding_checked_here=False,
    )


def _prior_residuals(prior, anchor, historical, compare):
    """Recompute E17 from all old holdout responses, never new grid labels."""
    require(prior["method"] == "fixed_terminal_secant_probe", "Closed prior secant protocol")
    exact_array("prior anchor binding", anchor, prior["anchor_commands"])
    exact_array("prior fit anchor binding", anchor, prior["fit"]["anchor_commands"])
    row = array(prior["fit"]["original_terminal_row"], "prior original terminal row", (72,))
    directions = array(prior["directions"], "prior directions", (3, 12, 6))
    steps = (0.001, 0.00075, 0.0005, 0.00025, 0.0001)
    exact_array("prior fixed holdout steps", steps, prior["step_sizes"])
    entries, requests, predictions = prior["evaluated_candidates"], prior["request_ledger"], prior["predictions"]
    require(len(entries) == len(requests) == 17 and len(predictions) == 15, "All prior17 calls and15 residuals")
    exact_array("historical anchor command", anchor, historical["commands"])
    require(_terminal(historical)[1] == _terminal(entries[0])[1], "Same prior terminal baseline")
    require(
        prior["fit"]["anchor_terminal_slack"] == _terminal(entries[0])[1],
        "Prior fit shares the full anchor baseline",
    )
    residuals = []
    for index in range(1, 16):
        di, si = divmod(index - 1, 5)
        command = anchor + steps[si] * directions[di]
        request, entry, prediction = requests[index], entries[index], predictions[index - 1]
        for name, value in (
            ("request_index", index),
            ("response_index", index),
            ("direction_index", di),
            ("step_index", si),
        ):
            integer(request[name], value, "prior request " + name)
        require(request["role"] == "grid" and request["step"] == steps[si], "Fixed prior grid request")
        exact_array("prior request commands", command, request["commands"])
        exact_array("prior response commands", command, entry["commands"])
        require(request["command_key"] == command.tobytes().hex(), "Prior request byte key")
        displacement = command - anchor
        original = float(row @ displacement.ravel())
        observed = _terminal(entry)[1] - _terminal(entries[0])[1]
        residual = observed - original
        integer(prediction["request_index"], index, "prior prediction association")
        compare("prior original prediction", original, prediction["original_predicted_change"])
        compare("prior observed change", observed, prediction["observed_change"])
        compare("prior original residual", residual, prediction["original_residual"])
        residuals.append(residual)
    return dict(
        passed=True,
        prior_grid_responses_checked=15,
        request_indices=list(range(1, 16)),
        response_indices=list(range(1, 16)),
        original_residuals=residuals,
        prior_max_abs_residual=max(abs(value) for value in residuals),
        source_sha_binding_checked_here=False,
    )


def audit_tail_translation_prior_residuals(prior_answer, *, compare=numerical_compare):
    """Metadata-only E17 reconstruction from the source-bound prior archive."""
    return _prior_residuals(
        prior_answer,
        array(prior_answer["anchor_commands"], "prior anchor", (12, 6)),
        prior_answer["evaluated_candidates"][0],
        compare,
    )


def _comparison(saved, left, right, index, compare):
    integer(saved["initial_response_index"], 0, "comparison initial index")
    integer(saved["repeated_response_index"], index, "comparison response index")
    local = dict(saved, repeated_response_index=1)
    checked = _audit_repeat(
        dict(repeatability=local, derivative_reliability_limited=not local["exact"]), [left, right], compare
    )
    checked["repeated_response_index"] = index
    return checked


def _causal_payload(entry):
    rollout = entry["rollout"]
    require(len(rollout["periods"]) == 12, "All twelve complete periods")
    for period in rollout["periods"]:
        for key in ("joint_position", "joint_velocity", "requested_torque", "applied_torque"):
            array(period[key], "period." + key, (25, 7))
        for key in ("mass_change", "jacobian_change"):
            array(period[key], "period." + key, (25,))
        array(period["endpoint_pose"], "period endpoint pose", (4, 4))
        array(period["endpoint_twist"], "period endpoint twist", (6,))
        boolean(period["update_matrices"], True, "State-updated period")
        integer(period["dynamics_queries"], 26, "All period queries")
    prefix = dict(periods=rollout["periods"][:11])
    for key in ("inner_joint_position", "inner_joint_velocity"):
        prefix[key] = array(rollout[key], key, (300, 7))[:275]
    for key, shape in (
        ("joint_position", (12, 7)),
        ("joint_velocity", (12, 7)),
        ("pose", (12, 4, 4)),
        ("twist", (12, 6)),
        ("goals", (12, 4, 4)),
    ):
        prefix[key] = array(rollout[key], key, shape)[:11]
    rotations = dict(
        goals=array(rollout["goals"], "goals", (12, 4, 4))[:, :3, :3],
        period_goals=np.asarray([array(p["goal"], "period goal", (4, 4))[:3, :3] for p in rollout["periods"]]),
    )
    return prefix, rotations


def _comparison_report(left, right, index):
    pending = [("", _json_values(left), _json_values(right))]
    arrays, structure, scalars = {}, True, True
    while pending:
        path, first, second = pending.pop()
        a, b = _numeric_list(first), _numeric_list(second)
        if a is not None or b is not None:
            compatible = a is not None and b is not None and a.shape == b.shape
            structure = structure and compatible
            difference = None
            if compatible and np.isfinite(a).all() and np.isfinite(b).all():
                difference = float(np.max(np.abs(a.astype(float) - b.astype(float)))) if a.size else 0.0
            arrays[path] = dict(
                shape=None if a is None else list(a.shape),
                repeated_shape=None if b is None else list(b.shape),
                exact=bool(compatible and np.array_equal(a, b)),
                max_abs_difference=difference,
            )
        elif isinstance(first, dict) and isinstance(second, dict):
            structure = structure and first.keys() == second.keys()
            pending.extend(
                (f"{path}.{key}" if path else key, first[key], second[key]) for key in first.keys() & second.keys()
            )
        elif isinstance(first, list) and isinstance(second, list):
            structure = structure and len(first) == len(second)
            pending.extend((f"{path}[{i}]", a, b) for i, (a, b) in enumerate(zip(first, second)))
        else:
            structure = structure and not isinstance(first, (dict, list)) and not isinstance(second, (dict, list))
            scalars = scalars and type(first) is type(second) and first == second
    return dict(
        initial_response_index=0,
        repeated_response_index=index,
        comparison_scope="Canonical JSON numeric arrays and scalars; no dtype or hidden empty-array dimension claim",
        exact=bool(structure and scalars and all(row["exact"] for row in arrays.values())),
        structure_equal=bool(structure),
        scalars_equal=bool(scalars),
        arrays=arrays,
        max_abs_difference=max(
            (row["max_abs_difference"] for row in arrays.values() if row["max_abs_difference"] is not None),
            default=0.0,
        ),
    )


def compare_tail_causal_rollouts(anchor_rollout, rollout, response_index):
    """Compare saved or independently replayed rollouts within their own route."""
    require(type(response_index) is int and 0 <= response_index < 14, "Fixed response index")
    first, first_rotation = _causal_payload(dict(rollout=anchor_rollout))
    second, second_rotation = _causal_payload(dict(rollout=rollout))
    prefix = _comparison_report(first, second, response_index)
    rotation = _comparison_report(first_rotation, second_rotation, response_index)
    return dict(
        response_index=response_index,
        prefix_period_count=11,
        prefix_inner_count=275,
        prefix_endpoint_count=11,
        prefix_comparison=prefix,
        rotation_goal_comparison=rotation,
        prefix_exact=prefix["exact"],
        rotation_goals_exact=rotation["exact"],
        passed=prefix["exact"] and rotation["exact"],
    )


def audit_tail_translation_ledger(answer, *, historical_anchor, prior_answer, compare=numerical_compare):
    """Audit14 fully replayed saved records; scientific failure remains valid."""
    require(answer["method"] == "fixed_tail_translation_reserve", "Fixed restoration protocol")
    grid = answer["grid"]
    grid_check = audit_tail_translation_grid(grid, compare=compare)
    for key in ("horizon", "tail_period_index", "axis_names"):
        _tree("answer/grid " + key, grid[key], answer[key], compare)
    for key in ("anchor_commands", "previous_command", "input_min", "input_max", "signed_steps"):
        exact_array("answer/grid " + key, grid[key], answer[key])
    anchor = array(answer["anchor_commands"], "anchor", (12, 6))
    previous = array(answer["previous_command"], "previous", (6,))
    lower, upper = (array(answer[k], k, (6,)) for k in ("input_min", "input_max"))
    for key in ("previous_command", "input_min", "input_max"):
        exact_array("prior/current " + key, prior_answer[key], answer[key])
    exact_array("prior/new anchor", prior_answer["anchor_commands"], anchor)
    _tree("prior historical anchor", prior_answer["evaluated_candidates"][0], historical_anchor, compare)
    prior_check = audit_tail_translation_prior_residuals(prior_answer, compare=compare)
    e17 = prior_check["prior_max_abs_residual"]
    for key, value in (
        ("max_responses", 14),
        ("planned_request_count", 14),
        ("response_evaluations", 14),
        ("unique_command_count", 13),
        ("total_iterations", 0),
        ("required_prefix", 9),
    ):
        integer(answer[key], value, key)
    require(answer["safety_buffer"] == 1e-8 and answer["reserve_factor"] == 10, "Original fixed safety gates")
    for key, value in (
        ("prior_max_abs_residual", e17),
        ("reserve_target", 10 * e17),
        ("terminal_threshold", 1e-8 + 10 * e17),
    ):
        compare(key, value, answer[key])
    boolean(answer["control_qualified"], False, "No control qualification")
    requests, responses = answer["request_ledger"], answer["evaluated_candidates"]
    require(len(requests) == len(responses) == 14, "All fourteen billed complete calls")
    specs = [(None, None, anchor.copy(), np.zeros_like(anchor)), *_grid_specs(anchor)]
    specs.append((None, None, anchor.copy(), np.zeros_like(anchor)))
    terminal, safety, squared, keys, offset = [], [], [], [], None
    for index, (request, entry, spec) in enumerate(zip(requests, responses, specs, strict=True)):
        axis, si, command, nominal = spec
        step = None if si is None else STEPS[si]
        expected = dict(
            request_index=index,
            role="anchor" if index == 0 else ("anchor_repeat" if index == 13 else "grid"),
            axis_index=axis,
            axis_name=None if axis is None else AXES[axis],
            step_index=si,
            signed_step=step,
            response_index=index,
            calls_before=index,
            calls_after=index + 1,
            charged=True,
            outcome="repeated" if index == 13 else "computed",
            repeat_of=0 if index == 13 else None,
        )
        for key, value in expected.items():
            require(type(request[key]) is type(value) and request[key] == value, "Fixed request " + key)
        _displacement("request", command, anchor, nominal, request, step, compare)
        _admission("request.admission", command, previous, lower, upper, request["admission"], compare)
        s, q = _response(entry, command, previous, lower, upper, compare, index)
        current_offset, value = _terminal(entry)
        require(offset is None or offset == current_offset, "Fixed terminal row layout")
        offset = current_offset
        safety.append(s)
        squared.append(q)
        terminal.append(value)
        keys.append(command.tobytes())
    require(len(set(keys)) == 13 and keys[0] == keys[-1], "Mandatory repeat only, always billed")
    integer(answer["terminal_row_index"], offset, "terminal row index")
    compare("terminal_values", terminal, answer["terminal_values"])
    repeat = _comparison(answer["repeatability"], responses[0], responses[-1], 13, compare)
    historical = answer["historical_anchor_comparison"]
    integer(historical["historical_response_index"], 0, "historical response index")
    integer(historical["new_response_index"], 0, "new response index")
    history = _comparison(
        dict(initial_response_index=0, repeated_response_index=1, **historical["comparison"]),
        historical_anchor,
        responses[0],
        1,
        compare,
    )
    causal, clipping = [], []
    require(len(answer["causal_checks"]) == 14, "Every causal comparison recorded")
    base_flags = _clip_flags(responses[0], 12)
    for index, entry in enumerate(responses):
        saved = answer["causal_checks"][index]
        expected = compare_tail_causal_rollouts(responses[0]["rollout"], entry["rollout"], index)
        require(saved == expected, "Complete exact causal difference report")
        causal.append(expected)
        changed = int(np.count_nonzero(_clip_flags(entry, 12) != base_flags))
        clipping.append(dict(response_index=index, exact=changed == 0, changed_count=changed, period_count=12))
    _tree("clipping checks", clipping, answer["clipping_checks"], compare)
    candidate_checks = []
    for index in range(1, 13):
        entry, request, current = responses[index], requests[index], causal[index]
        joint = entry["audit"]["feasible"]
        buffer = bool(np.all(safety[index] >= 1e-8))
        prefix = bool(np.all(squared[index][:9] <= 1.0))
        conditions = (
            (joint, "joint_audit"),
            (buffer, "safety_buffer"),
            (prefix, "prefix9"),
            (current["prefix_exact"], "causal_prefix"),
            (current["rotation_goals_exact"], "rotation_goals"),
            (history["exact"], "historical_anchor"),
            (repeat["exact"], "anchor_repeat"),
        )
        reasons = [name for passed, name in conditions if not passed]
        candidate_checks.append(
            dict(
                request_index=index,
                response_index=index,
                axis_index=request["axis_index"],
                axis_name=request["axis_name"],
                step_index=request["step_index"],
                signed_step=request["signed_step"],
                actual_displacement_linf=float(np.max(np.abs(array(entry["commands"], "command") - anchor))),
                terminal=terminal[index],
                terminal_reserve=terminal[index] - 1e-8,
                terminal_improvement=terminal[index] - terminal[0],
                other_safety_min=float(np.min(safety[index][:offset])),
                safety_min=float(np.min(safety[index])),
                joint_feasible=joint,
                safety_buffer_satisfied=buffer,
                prefix9_preserved=prefix,
                causal_exact=current["passed"],
                historical_exact=history["exact"],
                repeat_exact=repeat["exact"],
                restoration_eligible=not reasons,
                rejection_reasons=reasons,
            )
        )
    _tree("candidate_checks", candidate_checks, answer["candidate_checks"], compare)
    ranked = sorted(
        (row for row in candidate_checks if row["restoration_eligible"]),
        key=lambda row: (-row["terminal"], row["actual_displacement_linf"], row["request_index"]),
    )
    ranking = [row["response_index"] for row in ranked]
    require(
        isinstance(answer["eligible_ranking"], list)
        and all(type(value) is int for value in answer["eligible_ranking"])
        and answer["eligible_ranking"] == ranking,
        "Strict eligible ranking",
    )
    best = ranked[0] if ranked and ranked[0]["terminal"] > terminal[0] else None
    selected_index = None if best is None else best["response_index"]
    _tree("selected response index", selected_index, answer["selected_response_index"], compare)
    sufficient = bool(best is not None and best["terminal"] - 1e-8 >= 10 * e17)
    summary = None
    if best is None:
        require(answer["selected"] is None, "No selected trajectory without strict improvement")
    else:
        chosen = responses[selected_index]
        # The selection copies a saved response; numeric tolerance must not hide
        # even a sub-tolerance alteration to its reported strict classifications.
        _tree("selected full response", chosen, answer["selected"], compare)
        exact_array("selected command bytes", chosen["commands"], answer["selected"]["commands"])
        _response(
            answer["selected"],
            array(chosen["commands"], "selected command"),
            previous,
            lower,
            upper,
            compare,
            selected_index,
        )
        summary = {
            key: best[key]
            for key in (
                "response_index",
                "request_index",
                "terminal",
                "terminal_reserve",
                "terminal_improvement",
                "other_safety_min",
                "safety_min",
                "actual_displacement_linf",
            )
        }
        summary.update(
            prefix_length=chosen["prefix_length"],
            tcp_errors=chosen["tcp_errors"],
            normalized_squared_errors=chosen["normalized_squared_errors"],
            feasible=chosen["feasible"],
            restoration_eligible=True,
            sufficient_reserve_restored=sufficient,
        )
    _tree("selected summary", summary, answer["selected_summary"], compare)
    boolean(answer["improvement_found"], best is not None, "Strict terminal improvement")
    boolean(answer["sufficient_reserve_restored"], sufficient, "Fixed empirical reserve gate")
    all_causal = all(row["passed"] for row in causal)
    boolean(answer["causal_checks_all_passed"], all_causal, "All causal checks")
    return dict(
        passed=True,
        response_evaluations=14,
        unique_command_count=13,
        grid_check=grid_check,
        prior_residual_check=prior_check,
        repeatability=repeat,
        historical_anchor_exact=history["exact"],
        causal_checks=causal,
        causal_checks_all_passed=all_causal,
        eligible_ranking=ranking,
        selected_response_index=selected_index,
        selected_summary=summary,
        improvement_found=best is not None,
        sufficient_reserve_restored=sufficient,
        witness_indices=[i for i, entry in enumerate(responses) if entry["feasible"]],
        full_responses_independently_replayed_here=False,
        control_qualified=False,
    )
