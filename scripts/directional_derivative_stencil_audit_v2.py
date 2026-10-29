"""Audit the archived stencil with its original command-bound roundoff tolerance.

The frozen v1 audit rejected a source anchor outside a derived stage bound by
roundoff below 1e-12. This isolated version changes only that admission check;
anchor bytes, bound-aware FD signs, probe routing and predictions stay exact.
No producer, controller, response evaluator or simulator is imported.
"""

import numpy as np
from directional_derivative_audit import (
    STAGES,
    array,
    boolean,
    exact_array,
    integer,
    numerical_compare,
    require,
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
        require(
            np.all(lower - 1e-12 <= anchor.ravel()) and np.all(anchor.ravel() <= upper + 1e-12),
            "Anchor inside old stage bounds at original 1e-12 tolerance",
        )
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
