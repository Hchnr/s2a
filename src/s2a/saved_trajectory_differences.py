"""Describe finite differences already present in a saved multi-period trace.

This module performs arithmetic on saved commands and errors only. It never
predicts a response, constructs an optimizer, or changes the control contract.
"""

import numpy as np

STEP = 1e-4
COMMAND_TOLERANCE = 1e-12
SAFETY_TOLERANCE = 1e-10
SAFETY_BUFFER = 1e-8
STEP_BINS = (
    ("<=1e-4", 1e-4),
    ("(1e-4,1e-3]", 1e-3),
    ("(1e-3,1e-2]", 1e-2),
    ("(1e-2,1e-1]", 1e-1),
    (">1e-1", None),
)
LIMITATIONS = (
    "These are complete saved finite-difference groups and adjacent new evaluations, not recovered SLSQP Jacobian calls or accepted steps.",
    "The trace omits cache-only function and gradient calls; matching saved bytes cannot recover their ordering or the optimizer merit function.",
    "Selected points and callbacks are exact-byte references only, not evidence of feasibility, acceptance, convergence, or an available derivative group.",
    "Minimum safety scalar prediction is not a complete safety Jacobian. Active-constraint switching can make this scalar nonlinear even when every individual constraint is affine.",
    "Label checks use saved TCP errors, saved safety minima and exact command arithmetic; missing individual dynamic slacks are not reconstructed or independently validated.",
    "Only registered adjacent saved evaluations are compared. No new command, dynamics response, feasible-direction certificate, KKT proof, physical control or task qualification is produced.",
    "Bases with no identified probes are structurally unidentified; they do not establish a failed gradient call or a cache-only call.",
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _array(value, shape, name):
    result = np.asarray(value, dtype=np.float64)
    _require(result.shape == shape and np.all(np.isfinite(result)), f"Finite {name} of shape {shape} required")
    return result


def _integer(value, name, lower, upper):
    _require(
        isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)) and lower <= value <= upper,
        f"Integer {name} in [{lower},{upper}] required",
    )
    return int(value)


def _boolean(value, name):
    _require(isinstance(value, (bool, np.bool_)), f"Boolean {name} required")
    return bool(value)


def _command_audit(commands, previous, input_min, input_max):
    change = np.diff(np.vstack((previous, commands)), axis=0)
    lower, upper = commands - input_min, input_max - commands
    rate_lower, rate_upper = 0.25 + change, 0.25 - change
    box_min = float(min(np.min(lower), np.min(upper)))
    rate_min = float(min(np.min(rate_lower), np.min(rate_upper)))
    box_ok, rate_ok = box_min >= -COMMAND_TOLERANCE, rate_min >= -COMMAND_TOLERANCE
    return dict(
        box_lower=lower.tolist(),
        box_upper=upper.tolist(),
        rate_lower=rate_lower.tolist(),
        rate_upper=rate_upper.tolist(),
        box_minimum=box_min,
        rate_minimum=rate_min,
        box_feasible=bool(box_ok),
        rate_feasible=bool(rate_ok),
        feasible=bool(box_ok and rate_ok),
        tolerance=COMMAND_TOLERANCE,
    )


def _statistics(values):
    if not values:
        return dict(mean=None, mean_absolute=None, maximum_absolute=None, minimum=None, maximum=None)
    values = np.asarray(values, dtype=np.float64)
    return dict(
        mean=float(np.mean(values)),
        mean_absolute=float(np.mean(np.abs(values))),
        maximum_absolute=float(np.max(np.abs(values))),
        minimum=float(np.min(values)),
        maximum=float(np.max(values)),
    )


def _summary(pairs):
    residuals = [pair["target"]["prediction_residual"] for pair in pairs]
    maxima = [pair["max_abs_q_prediction_residual"] for pair in pairs]
    return dict(
        count=len(pairs),
        predicted_prefix_pass_actual_fail_count=sum(
            pair["prefix"]["predicted_passed"] and not pair["prefix"]["actual_passed"] for pair in pairs
        ),
        target_prediction_residual=_statistics(residuals),
        max_abs_q_prediction_residual=dict(
            mean=float(np.mean(maxima)) if maxima else None,
            maximum=float(np.max(maxima)) if maxima else None,
        ),
    )


def _summary_with_bins(pairs):
    result = _summary(pairs)
    bins = []
    lower = None
    for label, upper in STEP_BINS:
        members = [
            pair
            for pair in pairs
            if (lower is None or pair["step"]["linf"] > lower) and (upper is None or pair["step"]["linf"] <= upper)
        ]
        bins.append(dict(label=label, **_summary(members)))
        lower = upper
    result["step_bins"] = bins
    return result


def diagnose_saved_differences(answer, previous_command, *, target_index):
    """Identify saved stencils and compare only their immediate non-fit neighbor.

    ``answer`` is a frozen prefix solver answer with ``evaluated_candidates``,
    command bounds, phases and selected commands. H>=2 is required: the later
    command-bound rows recover the original interface box, while the first row
    is its intersection with the preceding command's rate interval. This is
    needed to reproduce the original audit's floating-point box/rate formulas.

    Every legal found probe whose base occurs earlier is marked as fit data,
    including probes belonging to incomplete or rejected groups. A complete
    group needs later probes in consecutive axis order within one phase. Its
    only possible comparison point is the immediately following saved row.
    No missing responses are reconstructed and no unfavorable row is skipped.

    Returned values are strict JSON-compatible Python objects. Input arrays
    and dictionaries are not mutated. A passed arithmetic diagnostic would not
    qualify dynamics, optimization convergence, or actual control.
    """
    horizon = _integer(answer["horizon"], "horizon", 2, 16)
    width = 6 * horizon
    target_index = _integer(target_index, "target_index", 0, horizon - 1)
    _require(answer["finite_difference_step"] == STEP, "Frozen nominal difference step must equal1e-4")
    previous = _array(previous_command, (6,), "previous command")
    lower = _array(answer["lower_command"], (horizon, 6), "lower command bounds")
    upper = _array(answer["upper_command"], (horizon, 6), "upper command bounds")
    _require(np.all(upper - lower > STEP), "Nondegenerate frozen difference intervals required")
    input_min, input_max = lower[1].copy(), upper[1].copy()
    _require(
        np.array_equal(lower[1:], np.tile(input_min, (horizon - 1, 1)))
        and np.array_equal(upper[1:], np.tile(input_max, (horizon - 1, 1)))
        and np.array_equal(lower[0], np.maximum(input_min, previous - 0.25))
        and np.array_equal(upper[0], np.minimum(input_max, previous + 0.25)),
        "Original box must be uniform after row0; row0 must be its exact previous-command rate intersection",
    )
    trace = answer["evaluated_candidates"]
    _require(isinstance(trace, list) and len(trace) > 0, "Nonempty saved unique-response trace required")
    count = len(trace)
    if "response_evaluations" in answer:
        _require(answer["response_evaluations"] == count, "Recorded response count must match the saved trace")
    commands = np.asarray([_array(row["commands"], (horizon, 6), "trace commands") for row in trace])
    _require(
        np.all(commands >= lower - COMMAND_TOLERANCE) and np.all(commands <= upper + COMMAND_TOLERANCE),
        "Every saved command must satisfy frozen solver bounds within1e-12",
    )
    errors = np.asarray([_array(row["tcp_errors"], (horizon, 6), "TCP errors") for row in trace])
    flat = commands.reshape(count, width)
    lookup = {row.tobytes(): index for index, row in enumerate(flat)}
    _require(len(lookup) == count, "Every trace command requires unique exact float64 bytes")
    with np.errstate(over="ignore", invalid="ignore"):
        scores = np.stack(
            (np.sum(errors[:, :, :3] ** 2, axis=2) / 0.01**2, np.sum(errors[:, :, 3:] ** 2, axis=2) / 0.05**2),
            axis=2,
        )
    _require(np.all(np.isfinite(scores)), "Finite normalized squared TCP errors required")
    safety = np.asarray([_array(row["safety_min"], (), "minimum safety scalar").item() for row in trace])
    audits, prefixes, joint_feasible = [], [], []
    for index, row in enumerate(trace):
        audit = _command_audit(commands[index], previous, input_min, input_max)
        failed = np.flatnonzero(~np.all(scores[index] <= 1.0, axis=1))
        prefix = int(failed[0]) if len(failed) else horizon
        joint = bool(audit["feasible"] and safety[index] >= -SAFETY_TOLERANCE)
        _require(
            _integer(row["prefix_length"], "saved prefix", 0, horizon) == prefix, "Strict saved pose prefix differs"
        )
        _require(_boolean(row["joint_feasible"], "joint_feasible") == joint, "Saved joint/command label differs")
        _require(
            _boolean(row["feasible"], "feasible") == (joint and prefix == horizon), "Strict saved witness label differs"
        )
        audits.append(audit)
        prefixes.append(prefix)
        joint_feasible.append(joint)
    phases = answer["phases"]
    _require(isinstance(phases, list), "Saved phases must be a list")
    phase_for = np.full(count, -1, dtype=int)
    phase_records = []
    previous_after = 0
    for index, phase in enumerate(phases):
        before = _integer(phase["evaluations_before"], "evaluations_before", 0, count)
        after = _integer(phase["evaluations_after"], "evaluations_after", before, count)
        _require(before >= previous_after, "Saved phase response intervals must be ordered and nonoverlapping")
        previous_after = after
        prefix = _integer(phase["prefix_before"], "phase prefix_before", 0, horizon - 1)
        cursor = _integer(phase["target_cursor"], "phase target_cursor", 0, horizon - 1)
        _require(cursor == prefix, "Frozen phase target_cursor must equal prefix_before")
        phase_for[before:after] = index
        phase_records.append(
            dict(
                phase_index=index,
                evaluations_before=before,
                evaluations_after=after,
                prefix_before=prefix,
                target_cursor=cursor,
            )
        )

    group_names = (
        "no_found_probe",
        "partial_group",
        "missing_probe",
        "illegal_probe",
        "cached_or_self_probe",
        "repeated_probe",
        "out_of_order_probe",
        "noncontiguous_probe",
        "unassigned_phase",
        "cross_phase",
        "nonunique_base",
    )
    group_counts = dict.fromkeys(group_names, 0)
    fit_probes = {}
    groups, excluded_groups = [], []
    flat_lower, flat_upper = lower.ravel(), upper.ravel()
    for base_index, base in enumerate(flat):
        probe_indices, deltas = [], []
        illegal_axes = []
        for axis in range(width):
            delta = STEP if base[axis] + STEP <= flat_upper[axis] else -STEP
            probe = base.copy()
            probe[axis] += delta
            legal = bool(
                np.all(probe >= flat_lower - COMMAND_TOLERANCE) and np.all(probe <= flat_upper + COMMAND_TOLERANCE)
            )
            found = lookup.get(probe.tobytes()) if legal else None
            if not legal:
                illegal_axes.append(axis)
            probe_indices.append(found)
            deltas.append(delta)
            if found is not None and found > base_index:
                fit_probes.setdefault(found, []).append(dict(base_index=base_index, axis=axis, nominal_delta=delta))
        found_indices = [index for index in probe_indices if index is not None]
        missing_axes = [axis for axis, index in enumerate(probe_indices) if index is None]
        reasons = []
        if not found_indices:
            reasons.append("no_found_probe")
        if 0 < len(found_indices) < width:
            reasons.append("partial_group")
        if missing_axes:
            reasons.append("missing_probe")
        if illegal_axes:
            reasons.append("illegal_probe")
        if any(index <= base_index for index in found_indices):
            reasons.append("cached_or_self_probe")
        if len(set(found_indices)) != len(found_indices):
            reasons.append("repeated_probe")
        if any(right <= left for left, right in zip(found_indices, found_indices[1:])):
            reasons.append("out_of_order_probe")
        if len(found_indices) == width and any(
            right != left + 1 for left, right in zip(found_indices, found_indices[1:])
        ):
            reasons.append("noncontiguous_probe")
        found_phases = {int(phase_for[index]) for index in found_indices}
        if -1 in found_phases:
            reasons.append("unassigned_phase")
        if len(found_phases) > 1:
            reasons.append("cross_phase")
        for reason in reasons:
            group_counts[reason] += 1
        if reasons:
            if found_indices:
                excluded_groups.append(
                    dict(
                        base_index=base_index,
                        probe_indices=probe_indices,
                        missing_axes=missing_axes,
                        illegal_axes=illegal_axes,
                        reasons=reasons,
                    )
                )
            continue
        phase_index = next(iter(found_phases))
        phase = phase_records[phase_index]
        groups.append(
            dict(
                base_index=base_index,
                probe_indices=probe_indices,
                nominal_deltas=deltas,
                phase_index=phase_index,
                base_cached_before_phase=base_index < phase["evaluations_before"],
                probe_start=probe_indices[0],
                probe_end=probe_indices[-1],
            )
        )
    signature_counts = {}
    for group in groups:
        signature = tuple(group["probe_indices"])
        signature_counts[signature] = signature_counts.get(signature, 0) + 1
    unique_groups = []
    for group in groups:
        if signature_counts[tuple(group["probe_indices"])] != 1:
            group_counts["nonunique_base"] += 1
            excluded_groups.append(
                dict(
                    base_index=group["base_index"],
                    probe_indices=group["probe_indices"],
                    missing_axes=[],
                    illegal_axes=[],
                    reasons=["nonunique_base"],
                )
            )
        else:
            unique_groups.append(group)
    groups = sorted(unique_groups, key=lambda group: group["probe_start"])
    adjacent_counts = dict.fromkeys(("no_next_row", "different_phase", "earlier_base_fit_probe"), 0)
    excluded_pairs, pairs = [], []
    for group_index, group in enumerate(groups):
        group["group_index"] = group_index
        base_index = group["base_index"]
        trial_index = group["probe_end"] + 1
        reasons = []
        if trial_index == count:
            reasons.append("no_next_row")
        else:
            if int(phase_for[trial_index]) != group["phase_index"]:
                reasons.append("different_phase")
            if trial_index in fit_probes:
                reasons.append("earlier_base_fit_probe")
        if reasons:
            for reason in reasons:
                adjacent_counts[reason] += 1
            excluded_pairs.append(
                dict(group_index=group_index, base_index=base_index, trial_index=trial_index, reasons=reasons)
            )
            continue
        ids = group["probe_indices"]
        deltas = np.asarray(group["nominal_deltas"])
        jacobian = ((scores[ids] - scores[base_index]) / deltas[:, None, None]).reshape(width, -1).T
        displacement = flat[trial_index] - flat[base_index]
        predicted = scores[base_index] + (jacobian @ displacement).reshape(horizon, 2)
        scalar_gradient = (safety[ids] - safety[base_index]) / deltas
        predicted_safety = float(safety[base_index] + scalar_gradient @ displacement)
        _require(
            np.all(np.isfinite(predicted)) and np.isfinite(predicted_safety), "Finite saved affine predictions required"
        )
        phase = phase_records[group["phase_index"]]
        prefix = phase["prefix_before"]
        pred_prefix, actual_prefix = predicted[:prefix], scores[trial_index, :prefix]
        predicted_passed, actual_passed = bool(np.all(pred_prefix <= 1.0)), bool(np.all(actual_prefix <= 1.0))
        base_target = float(np.max(scores[base_index, target_index]))
        predicted_target = float(np.max(predicted[target_index]))
        actual_target = float(np.max(scores[trial_index, target_index]))
        predicted_decrease, actual_decrease = base_target - predicted_target, base_target - actual_target
        linf = float(np.max(np.abs(displacement)))
        rms = float(np.sqrt(np.mean(displacement**2)))
        main = bool(
            phase["target_cursor"] == target_index
            and joint_feasible[base_index]
            and safety[base_index] >= SAFETY_BUFFER
            and prefixes[base_index] >= prefix
            and audits[base_index]["feasible"]
        )
        pair = dict(
            **group,
            trial_index=trial_index,
            main_subset=main,
            phase_target_cursor=phase["target_cursor"],
            prefix_before=prefix,
            step=dict(linf=linf, rms=rms, ratio_to_h=linf / STEP),
            qbase=scores[base_index].tolist(),
            qpred=predicted.tolist(),
            qactual=scores[trial_index].tolist(),
            max_abs_q_prediction_residual=float(np.max(np.abs(scores[trial_index] - predicted))),
            target=dict(
                index=target_index,
                base=scores[base_index, target_index].tolist(),
                predicted=predicted[target_index].tolist(),
                actual=scores[trial_index, target_index].tolist(),
                base_max=base_target,
                predicted_max=predicted_target,
                actual_max=actual_target,
                predicted_change=-predicted_decrease,
                actual_change=-actual_decrease,
                predicted_decrease=predicted_decrease,
                actual_decrease=actual_decrease,
                prediction_residual=actual_target - predicted_target,
                descent_ratio=actual_decrease / predicted_decrease if predicted_decrease > 1e-12 else None,
            ),
            prefix=dict(
                predicted_max_cap_excess=float(max(0.0, np.max(pred_prefix) - 1.0)) if prefix else 0.0,
                actual_max_cap_excess=float(max(0.0, np.max(actual_prefix) - 1.0)) if prefix else 0.0,
                predicted_passed=predicted_passed,
                actual_passed=actual_passed,
            ),
            scalar_safety=dict(
                name="minimum safety scalar prediction",
                base=float(safety[base_index]),
                predicted=predicted_safety,
                actual=float(safety[trial_index]),
                prediction_residual=float(safety[trial_index] - predicted_safety),
                optimization_buffer=SAFETY_BUFFER,
                predicted_buffer_satisfied=predicted_safety >= SAFETY_BUFFER,
                actual_buffer_satisfied=bool(safety[trial_index] >= SAFETY_BUFFER),
                base_joint_feasible=joint_feasible[base_index],
                trial_joint_feasible=joint_feasible[trial_index],
                full_safety_jacobian_available=False,
            ),
            base_command_audit=audits[base_index],
            trial_command_audit=audits[trial_index],
        )
        pairs.append(pair)

    def link(value):
        if value is None:
            return dict(
                present=False,
                exact_match=False,
                trace_index=None,
                fit_probe_of=[],
                complete_group_indices=[],
                paired_trial_group_indices=[],
            )
        key = _array(value, (horizon, 6), "linked commands").ravel().tobytes()
        index = lookup.get(key)
        return dict(
            present=True,
            exact_match=index is not None,
            trace_index=index,
            fit_probe_of=fit_probes.get(index, []),
            complete_group_indices=[group["group_index"] for group in groups if group["base_index"] == index],
            paired_trial_group_indices=[pair["group_index"] for pair in pairs if pair["trial_index"] == index],
        )

    callbacks = [
        dict(phase_index=index, **link(phase.get("last_callback_commands"))) for index, phase in enumerate(phases)
    ]
    return dict(
        horizon=horizon,
        target_index=target_index,
        trace_count=count,
        nominal_difference_step=STEP,
        original_input_min=input_min.tolist(),
        original_input_max=input_max.tolist(),
        complete_group_count=len(groups),
        complete_groups=groups,
        excluded_groups=excluded_groups,
        fit_probe_indices=sorted(fit_probes),
        fit_probe_count=len(fit_probes),
        unidentified_base_count=group_counts["no_found_probe"],
        exclusions=dict(
            counts_are_overlapping=True,
            group_counts=group_counts,
            group_partition=dict(
                total_bases=count,
                complete_groups=len(groups),
                rejected_with_found_probe=len(excluded_groups),
                unidentified_without_found_probe=group_counts["no_found_probe"],
            ),
            adjacent_trial_counts=adjacent_counts,
        ),
        excluded_pairs=excluded_pairs,
        phases=phase_records,
        pairs=pairs,
        links=dict(selected=link(answer["selected"]["commands"]), callbacks=callbacks),
        summaries=dict(
            all_pairs=_summary_with_bins(pairs),
            main_subset=_summary_with_bins([pair for pair in pairs if pair["main_subset"]]),
        ),
        limitations=list(LIMITATIONS),
        new_forward_responses=0,
        new_queries=0,
        physics_steps=0,
        world_model_calls=0,
        fresh_cases=0,
        control_qualified=False,
    )
