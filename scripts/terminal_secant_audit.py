"""Independent saved-array audit of the fixed terminal-row secant experiment.

No production fitting, geometry, predictor, or simulator is imported. Callers
must bind the historical archives and independently replay all billed responses;
this module checks their algebra, routing, original labels, and qualifications.
"""

import numpy as np
from directional_derivative_audit import (
    COMMAND_GROUPS,
    SAFETY_GROUPS,
    _audit_repeat,
    _response,
    admission,
    array,
    boolean,
    exact_array,
    integer,
    numerical_compare,
    require,
)

STEPS = (1e-3, 7.5e-4, 5e-4, 2.5e-4, 1e-4)
FIT_STEPS = (1e-3, 5e-4)
NAMES = ("orthogonal", "source_plus_orthogonal", "source_minus_orthogonal")


def _tree(name, expected, saved, compare):
    if isinstance(expected, dict):
        require(isinstance(saved, dict) and expected.keys() == saved.keys(), name + ": exact fields")
        for key, value in expected.items():
            _tree(name + "." + key, value, saved[key], compare)
    elif isinstance(expected, list) and (not expected or isinstance(expected[0], (dict, str))):
        require(isinstance(saved, list) and len(saved) == len(expected), name + ": complete sequence")
        for index, (left, right) in enumerate(zip(expected, saved, strict=True)):
            _tree(f"{name}[{index}]", left, right, compare)
    elif expected is None or isinstance(expected, (str, bool, int)):
        require(type(saved) is type(expected) and saved == expected, name + ": exact value")
    else:
        compare(name, expected, saved)


def _admission(name, commands, previous, lower, upper, saved, compare):
    expected = admission(commands, previous, lower, upper)
    _tree(name, expected, saved, compare)
    return expected


def audit_terminal_secant_geometry(geometry, *, compare=numerical_compare):
    """Rebuild the fixed signed guard rows, one RNG draw, and all 15 commands."""
    h = geometry["horizon"]
    require(type(h) is int and 1 <= h <= 16, "Bounded horizon")
    anchor = array(geometry["anchor_commands"], "anchor", (h, 6))
    previous = array(geometry["previous_command"], "previous", (6,))
    lower, upper = (array(geometry[key], key, (6,)) for key in ("input_min", "input_max"))
    require(np.all(lower < upper), "Original strict input bounds")
    sources = array(geometry["source_unit_directions"], "source directions", (3, h, 6))
    require(np.all(np.max(np.abs(sources), axis=(1, 2)) == 1.0), "Original infinity-normalized source rows")
    for key, expected in (
        ("seed", 42),
        ("svd_rcond", 1e-12),
        ("box_guard", 1e-3),
        ("rate_guard", 2e-3),
        ("source_direction_index", 0),
        ("random_generator", "numpy.default_rng/PCG64"),
        ("sign_resolution_threshold", 1e-12),
    ):
        require(geometry[key] == expected, "Fixed geometry " + key)
    exact_array("step_sizes", STEPS, geometry["step_sizes"])
    require(geometry["direction_names"] == list(NAMES), "Fixed three holdout directions")
    base_admission = _admission(
        "anchor_admission", anchor, previous, lower, upper, geometry["anchor_admission"], compare
    )
    require(base_admission["admitted"], "Unclipped admissible anchor")
    width = anchor.size
    identity = np.eye(width)
    difference = identity - np.eye(width, k=-6)
    gradients = dict(box_lower=identity, box_upper=-identity, rate_lower=difference, rate_upper=-difference)
    rows = list(sources.reshape(3, width).copy())
    metadata = [dict(kind="source_direction", source_direction_index=i) for i in range(3)]
    for group in COMMAND_GROUPS:
        guard = 1e-3 if group.startswith("box") else 2e-3
        slack = base_admission[group].ravel()
        for axis in np.flatnonzero(slack <= guard):
            rows.append(gradients[group][axis])
            metadata.append(
                dict(kind="command_guard", group=group, axis=int(axis), slack=float(slack[axis]), guard=guard)
            )
    rows = np.asarray(rows)
    exact_array("signed unnormalized projection_rows", rows, geometry["projection_rows"])
    _tree("projection_row_metadata", metadata, geometry["projection_row_metadata"], compare)
    _, singular, right = np.linalg.svd(rows, full_matrices=False)
    cutoff = float(singular[0] * 1e-12)
    rank = int(np.count_nonzero(singular > cutoff))
    require(rank < width, "Nonempty orthogonal complement")
    gaussian = np.random.default_rng(42).standard_normal(width)
    basis = right[:rank]
    projected = gaussian - basis.T @ (basis @ gaussian)
    norm = float(np.linalg.norm(projected))
    require(norm > 1e-10, "Fixed absolute Gaussian projection floor")
    v = projected / norm
    sign_index = int(np.flatnonzero(np.abs(v) > 1e-12)[0])
    sign = 1.0 if v[sign_index] > 0 else -1.0
    v *= sign
    e = sources[0].ravel() / np.linalg.norm(sources[0])
    residual = rows @ v
    require(np.max(np.abs(residual)) <= 1e-10, "Fixed orthogonal projection residual gate")
    raw = np.asarray((v, e + v, e - v))
    scales = np.max(np.abs(raw), axis=1)
    directions = (raw / scales[:, None]).reshape(3, h, 6)
    expected = dict(
        singular_values=singular,
        singular_cutoff=cutoff,
        projection_rank=rank,
        nullspace_dimension=width - rank,
        gaussian=gaussian,
        projected_gaussian=projected,
        projected_norm=norm,
        projected_to_gaussian_norm_ratio=norm / float(np.linalg.norm(gaussian)),
        sign_index=sign_index,
        sign_multiplier=sign,
        e=e.reshape(h, 6),
        v=v.reshape(h, 6),
        projection_residual=residual,
        projection_residual_inf=float(np.max(np.abs(residual))),
        raw_directions=raw.reshape(3, h, 6),
        direction_scales=scales,
        directions=directions,
    )
    for key, value in expected.items():
        _tree(key, value, geometry[key], compare)
    exact_array("fixed holdout direction bytes", directions, geometry["directions"])
    flat = directions.reshape(3, width)
    norms = np.linalg.norm(flat, axis=1)
    source_flat = sources.reshape(3, width)
    compare(
        "pairwise_direction_cosine", (flat @ flat.T) / np.outer(norms, norms), geometry["pairwise_direction_cosine"]
    )
    compare(
        "source_direction_cosine",
        (flat @ source_flat.T) / np.outer(norms, np.linalg.norm(source_flat, axis=1)),
        geometry["source_direction_cosine"],
    )
    require(len(geometry["grid_commands"]) == 15, "All prescribed 3 by 5 commands")
    seen = {anchor.tobytes()}
    for index, saved in enumerate(geometry["grid_commands"]):
        di, si = divmod(index, 5)
        step = STEPS[si]
        nominal = step * directions[di]
        command = anchor + nominal
        delta = command - anchor
        key = command.tobytes()
        require(key not in seen and np.any(delta != 0), "15 unique nonzero commands, distinct from anchor")
        seen.add(key)
        for field, value in (("commands", command), ("actual_displacement", delta), ("nominal_displacement", nominal)):
            exact_array("grid." + field, value, saved[field])
        expected = dict(
            grid_index=index,
            direction_index=di,
            direction_name=NAMES[di],
            step_index=si,
            step=step,
            command_key=key.hex(),
            unique_command=True,
            zero_displacement=False,
        )
        for field, value in expected.items():
            _tree("grid." + field, value, saved[field], compare)
        error = float(np.max(np.abs(delta - nominal)))
        compare("grid displacement_error_inf", error, saved["displacement_error_inf"])
        compare("grid relative_displacement_error_inf", error / step, saved["relative_displacement_error_inf"])
        admitted = _admission("grid.admission", command, previous, lower, upper, saved["admission"], compare)
        require(admitted["admitted"], "Entire fixed grid must pass command admission before responses")
    boolean(geometry["admitted"], True, "geometry admitted")
    integer(geometry["planned_request_count"], 17, "17 requests")
    integer(geometry["expected_unique_command_count"], 16, "16 unique commands")
    return dict(
        passed=True,
        projection_row_count=len(rows),
        projection_rank=rank,
        nullspace_dimension=width - rank,
        projection_residual_inf=float(np.max(np.abs(residual))),
        grid_commands_checked=15,
        source_archive_binding_checked_here=False,
    )


def _terminal(entry):
    groups = entry["audit"]["groups"]
    values = [array(groups[key]["slack"], "saved safety " + key) for key in SAFETY_GROUPS]
    require(values[-1].shape == (1,), "One fixed e7 terminal row")
    joined = np.concatenate([value.ravel() for value in values])
    exact_array("complete terminal-layout safety", joined, entry["safety_slack"])
    offset = joined.size - 1
    velocity = array(entry["rollout"]["joint_velocity"], "joint_velocity")
    require(velocity.ndim == 2 and velocity.shape[1] == 7, "Seven joint terminal velocity")
    require(joined[offset] == -velocity[-1, 6], "terminal[0] is exactly negative final joint-seven velocity")
    return offset, float(joined[offset])


def audit_terminal_secant_fit(fit, *, source_answer=None, compare=numerical_compare):
    """Recompute the two-sample zero-intercept fit using actual displacements."""
    anchor = array(fit["anchor_commands"], "fit anchor")
    require(anchor.ndim == 2 and anchor.shape[1] == 6 and 1 <= len(anchor) <= 16, "Complete fit anchor")
    unit = array(fit["source_unit_direction"], "fit unit", anchor.shape)
    require(np.max(np.abs(unit)) == 1.0, "Source unit infinity norm")
    row = array(fit["original_terminal_row"], "original terminal row", (anchor.size,))
    initial = float(array(fit["anchor_terminal_slack"], "historical terminal", ()))
    commands = array(fit["fit_commands"], "two fit commands", (2, *anchor.shape))
    terminal = array(fit["fit_terminal_slacks"], "two fit slacks", (2,))
    exact_array("fixed positive training steps", FIT_STEPS, fit["fit_step_sizes"])
    for index, step in enumerate(FIT_STEPS):
        exact_array("fixed training command", anchor + step * unit, commands[index])
    delta = commands - anchor
    flat, p = unit.ravel(), delta.reshape(2, -1)
    squared = float(flat @ flat)
    coordinate = (p @ flat) / squared
    old = p @ row
    observed = terminal - initial
    residual = observed - old
    denominator = float(coordinate @ coordinate)
    require(denominator > 0 and np.all(coordinate > 0), "Positive finite source coordinates")
    coefficient = float((coordinate @ residual) / denominator)
    correction = coefficient * flat / squared
    corrected = row + correction
    predicted = p @ corrected
    expected = dict(
        actual_displacements=delta,
        source_coordinates=coordinate,
        original_predicted_changes=old,
        observed_changes=observed,
        original_residuals=residual,
        coefficient=coefficient,
        correction_row=correction,
        corrected_terminal_row=corrected,
        corrected_predicted_changes=predicted,
        corrected_residuals=observed - predicted,
        least_squares_denominator=denominator,
    )
    for key, value in expected.items():
        compare("fit." + key, value, fit[key])
    bound = []
    if source_answer is not None:
        exact_array("historical anchor association", anchor, source_answer["anchor_commands"])
        require(source_answer["source_stages"][0] == 24, "Original stage24 source direction")
        exact_array("historical unit association", unit, source_answer["unit_directions"][0])
        responses = source_answer["evaluated_candidates"]
        require(_terminal(responses[0])[1] == initial, "Original historical terminal baseline")
        for index, step in enumerate(FIT_STEPS):
            matching = [
                item
                for item in source_answer["request_ledger"]
                if item["role"] == "grid"
                and item["source_stage"] == 24
                and item["direction_index"] == 0
                and item["sign"] == 1
                and item["step"] == step
            ]
            require(len(matching) == 1, "Unique original positive stage24 training request")
            request = matching[0]
            require(
                request["admission"]["admitted"] is True and request["outcome"] in ("computed", "cached"),
                "Completed admitted training response",
            )
            ri = request["response_index"]
            require(type(ri) is int and 0 <= ri < len(responses), "Training response routing")
            exact_array("training request association", commands[index], request["commands"])
            exact_array("training response association", commands[index], responses[ri]["commands"])
            require(_terminal(responses[ri])[1] == terminal[index], "Training terminal label association")
            bound.append(dict(request_index=request["request_index"], response_index=ri, step=step))
    return dict(
        passed=True,
        training_samples=2,
        coefficient=coefficient,
        source_archive_binding_checked_here=source_answer is not None,
        source_training_requests=bound,
        original_jacobian_source_binding_checked_here=False,
    )


def _clip_flags(entry, horizon):
    periods = entry["rollout"]["periods"]
    require(len(periods) == horizon, "All saved clipping periods")
    result = []
    for period in periods:
        flags = np.asarray(period["clipped"])
        require(flags.shape == (25, 7) and flags.dtype == np.dtype(bool), "Exact boolean clipping matrix")
        result.append(flags)
    return np.asarray(result)


def audit_terminal_secant_ledger(answer, *, historical_anchor, source_answer=None, compare=numerical_compare):
    """Audit all17 prevalidated responses, without calling any response model."""
    require(answer["method"] == "fixed_terminal_secant_probe", "Fixed terminal secant protocol")
    geometry, fit = answer["geometry"], answer["fit"]
    geometry_check = audit_terminal_secant_geometry(geometry, compare=compare)
    fit_check = audit_terminal_secant_fit(fit, source_answer=source_answer, compare=compare)
    h = geometry["horizon"]
    integer(answer["horizon"], h, "horizon")
    for key in (
        "previous_command",
        "input_min",
        "input_max",
        "anchor_commands",
        "source_unit_directions",
        "directions",
        "step_sizes",
    ):
        exact_array("answer/geometry " + key, geometry[key], answer[key])
    require(answer["direction_names"] == list(NAMES), "Fixed direction names")
    anchor = array(answer["anchor_commands"], "anchor", (h, 6))
    previous = array(answer["previous_command"], "previous", (6,))
    lower, upper = (array(answer[key], key, (6,)) for key in ("input_min", "input_max"))
    exact_array("fit/answer anchor", anchor, fit["anchor_commands"])
    exact_array("fit/geometry source unit", geometry["source_unit_directions"][0], fit["source_unit_direction"])
    exact_array("historical full anchor commands", anchor, historical_anchor["commands"])
    require(
        _terminal(historical_anchor)[1] == fit["anchor_terminal_slack"],
        "Fit baseline belongs to historical full anchor",
    )
    for key, value in (
        ("max_responses", 17),
        ("planned_request_count", 17),
        ("response_evaluations", 17),
        ("unique_command_count", 16),
        ("total_iterations", 0),
    ):
        integer(answer[key], value, key)
    boolean(answer["control_qualified"], False, "No control qualification")
    ledger, responses = answer["request_ledger"], answer["evaluated_candidates"]
    require(len(ledger) == len(responses) == 17, "Exactly17 charged requests and complete responses")
    directions = array(answer["directions"], "directions", (3, h, 6))
    plan = [("anchor", None, None, anchor.copy(), np.zeros_like(anchor))]
    for di in range(3):
        for si, step in enumerate(STEPS):
            nominal = step * directions[di]
            plan.append(("grid", di, si, anchor + nominal, nominal))
    plan.append(("anchor_repeat", None, None, anchor.copy(), np.zeros_like(anchor)))
    terminal, offset, keys = [], None, []
    for index, (request, spec, entry) in enumerate(zip(ledger, plan, responses, strict=True)):
        role, di, si, command, nominal = spec
        step = None if si is None else STEPS[si]
        expected = dict(
            request_index=index,
            role=role,
            direction_index=di,
            direction_name=None if di is None else NAMES[di],
            step_index=si,
            step=step,
            command_key=command.tobytes().hex(),
            response_index=index,
            calls_before=index,
            calls_after=index + 1,
            charged=True,
            outcome="repeated" if index == 16 else "computed",
            repeat_of=0 if index == 16 else None,
        )
        for key, value in expected.items():
            _tree("request." + key, value, request[key], compare)
        actual = command - anchor
        error = float(np.max(np.abs(actual - nominal)))
        for key, value in (("commands", command), ("actual_displacement", actual), ("nominal_displacement", nominal)):
            exact_array("request." + key, value, request[key])
        compare("request.displacement_error_inf", error, request["displacement_error_inf"])
        _tree(
            "request.relative_displacement_error_inf",
            None if step is None else error / step,
            request["relative_displacement_error_inf"],
            compare,
        )
        boolean(request["zero_displacement"], index in (0, 16), "zero displacement anchors only")
        require(
            _admission("request.admission", command, previous, lower, upper, request["admission"], compare)["admitted"],
            "No inadmissible or clipped requests",
        )
        _response(entry, command, previous, lower, upper, compare, index)
        this_offset, value = _terminal(entry)
        if offset is None:
            offset = this_offset
        require(offset == this_offset, "Invariant terminal row layout")
        terminal.append(value)
        keys.append(command.tobytes())
    require(len(set(keys)) == 16 and keys[0] == keys[-1], "Only the mandatory repeat duplicates a command")
    integer(answer["terminal_row_index"], offset, "terminal row index")
    repeat = _audit_repeat(
        dict(
            repeatability=answer["repeatability"], derivative_reliability_limited=not answer["repeatability"]["exact"]
        ),
        responses,
        compare,
    )
    if "derivative_reliability_limited" in answer:
        boolean(answer["derivative_reliability_limited"], not repeat["exact"], "Repeat reliability report")
    historical = answer["historical_anchor_comparison"]
    integer(historical["historical_response_index"], 0, "historical response index")
    integer(historical["new_response_index"], 0, "new historical comparison index")
    historical_record = dict(initial_response_index=0, repeated_response_index=1, **historical["comparison"])
    history = _audit_repeat(
        dict(repeatability=historical_record, derivative_reliability_limited=not historical_record["exact"]),
        [historical_anchor, responses[0]],
        compare,
    )
    baseline_flags = _clip_flags(responses[0], h)
    clipping = []
    for index, entry in enumerate(responses):
        changed = int(np.count_nonzero(_clip_flags(entry, h) != baseline_flags))
        clipping.append(dict(response_index=index, exact=changed == 0, changed_count=changed, period_count=h))
    _tree("clipping_checks", clipping, answer["clipping_checks"], compare)
    old = array(fit["original_terminal_row"], "old terminal row", (anchor.size,))
    # Qualification uses independently reconstructed algebra, never a perturbed
    # stored row that merely falls within the numeric comparison tolerance.
    fit_unit = array(fit["source_unit_direction"], "fit source unit", anchor.shape).ravel()
    corrected = old + fit_check["coefficient"] * fit_unit / float(fit_unit @ fit_unit)
    require(len(answer["predictions"]) == 15, "All15 holdout predictions")
    predictions = []
    for index, saved in enumerate(answer["predictions"], start=1):
        di, si = divmod(index - 1, 5)
        step = STEPS[si]
        displacement = array(responses[index]["commands"], "response command", anchor.shape) - anchor
        original = float(old @ displacement.ravel())
        updated = float(corrected @ displacement.ravel())
        observed = terminal[index] - terminal[0]
        residual, new_residual = observed - original, observed - updated
        row = dict(
            request_index=index,
            direction_index=di,
            direction_name=NAMES[di],
            step_index=si,
            step=step,
            original_predicted_change=original,
            corrected_predicted_change=updated,
            observed_change=observed,
            original_residual=residual,
            corrected_residual=new_residual,
            original_residual_over_step=residual / step,
            corrected_residual_over_step=new_residual / step,
            prediction_correction=updated - original,
        )
        _tree("holdout prediction", row, saved, compare)
        predictions.append(row)
    mixed = []
    for di in (1, 2):
        group = predictions[di * 5 : (di + 1) * 5]
        oldmax = max(abs(row["original_residual"]) for row in group)
        newmax = max(abs(row["corrected_residual"]) for row in group)
        oldscaled = max(abs(row["original_residual_over_step"]) for row in group)
        newscaled = max(abs(row["corrected_residual_over_step"]) for row in group)
        absolute, normalized = newmax <= 0.5 * oldmax, newscaled <= 0.5 * oldscaled
        mixed.append(
            dict(
                direction_index=di,
                direction_name=NAMES[di],
                original_max_abs_residual=oldmax,
                corrected_max_abs_residual=newmax,
                original_max_abs_normalized_residual=oldscaled,
                corrected_max_abs_normalized_residual=newscaled,
                absolute_passed=absolute,
                normalized_passed=normalized,
                passed=absolute and normalized,
            )
        )
    puremax = max(abs(row["prediction_correction"]) for row in predictions[:5])
    pure = dict(max_abs_prediction_change=puremax, tolerance=1e-12, passed=puremax <= 1e-12)
    clips_equal = all(row["exact"] for row in clipping)
    passed = (
        all(row["passed"] for row in mixed) and pure["passed"] and clips_equal and repeat["exact"] and history["exact"]
    )
    margin = float(fit["anchor_terminal_slack"]) - 1e-8
    max_residual = max(abs(row["corrected_residual"]) for row in predictions)
    accuracy = max_residual <= margin
    qualification = dict(
        mixed_direction_checks=mixed,
        pure_direction_check=pure,
        all_clipping_equal=clips_equal,
        repeat_exact=repeat["exact"],
        historical_exact=history["exact"],
        improvement_factor=0.5,
        safety_buffer=1e-8,
        safety_available_margin=margin,
        max_abs_corrected_residual=max_residual,
        extrapolation_passed=passed,
        safety_accuracy_passed=accuracy,
    )
    _tree("qualification", qualification, answer["qualification"], compare)
    boolean(answer["extrapolation_passed"], passed, "Strict extrapolation classification")
    boolean(answer["safety_accuracy_passed"], accuracy, "Separate absolute prediction accuracy classification")
    return dict(
        passed=True,
        response_evaluations=17,
        unique_command_count=16,
        grid_predictions_checked=15,
        geometry_check=geometry_check,
        fit_check=fit_check,
        repeatability=repeat,
        historical_anchor_exact=history["exact"],
        qualification=qualification,
        extrapolation_passed=passed,
        safety_accuracy_passed=accuracy,
        witness_indices=[i for i, entry in enumerate(responses) if entry["feasible"]],
        full_responses_independently_replayed_here=False,
        control_qualified=False,
    )
