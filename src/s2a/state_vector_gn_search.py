"""Bounded causal trajectory search with a normalized-state Gauss-Newton pose QP.

The full original execution gates and finite response state machine remain in
force. State-vector loss is a numerical objective, not task qualification.
"""

from copy import deepcopy

import numpy as np

from s2a.accepted_point_sqp_core import solve_epigraph_qp
from s2a.geometry import finite
from s2a.global_envelope_search import POSE_CEILING_POLICY, initial_global_envelope
from s2a.interior_trajectory_search import (
    SearchEvaluationError,
    _entry,
    _jacobian,
    _outputs,
    _require,
    causal_periods,
    causal_secant_update,
)
from s2a.interior_trajectory_search import _metrics as original_metrics
from s2a.interior_trajectory_search import build_stage_problem as original_stage_problem
from s2a.rejected_response_search import FEEDBACK_POLICY, choose_rejected_source
from s2a.restored_anchor_sqp_plan import canonical_initial_comparison
from s2a.stage_budget_search import validate_config

POSE_OBJECTIVE_POLICY = "normalized_state_vector_gauss_newton"
POSE_DAMPING = 1.0
ERROR_SCALE = np.array([0.01, 0.01, 0.01, 0.05, 0.05, 0.05])


def normalized_error_vector(entry):
    """Read signed state errors; compact FD entries need no redundant q/loss."""
    error = finite(entry["tcp_errors"], (12, 6))
    if "normalized_squared_errors" in entry:
        q = np.column_stack((np.sum(error[:, :3] ** 2, axis=1) / 0.01**2, np.sum(error[:, 3:] ** 2, axis=1) / 0.05**2))
        _require(
            np.array_equal(q, finite(entry["normalized_squared_errors"], (12, 2))),
            "Saved normalized pose errors must match the original algebra",
        )
    return finite(error / ERROR_SCALE).ravel().copy()


def initial_error_vector_jacobian(bundle):
    """Build the causal signed-error derivative from the original compact72 FD."""
    base, probes = bundle["base"], bundle["probes"]
    x = finite(base["commands"], (12, 6))
    _outputs(base)  # A full base requires its original normalized q field.
    E = normalized_error_vector(base)
    nominal = finite(bundle["nominal_deltas"], (72,))
    _require(len(probes) == 72 and np.all(np.abs(nominal) == 1e-4), "Complete original signed nominal stencil")
    columns = []
    for axis, (probe, delta) in enumerate(zip(probes, nominal, strict=True)):
        command = finite(probe["commands"], (12, 6))
        actual = command.ravel() - x.ravel()
        tolerance = 4 * np.finfo(float).eps * max(1.0, abs(x.ravel()[axis]), abs(command.ravel()[axis]))
        _require(
            np.all(actual[np.arange(72) != axis] == 0) and actual[axis] != 0 and abs(actual[axis] - delta) <= tolerance,
            "One original registered nominal FD axis",
        )
        columns.append((normalized_error_vector(probe) - E) / delta)
    return _jacobian(np.asarray(columns).T, np.repeat(np.arange(12), 6))


def _metrics(safety, q, config, error_vector=None):
    result = original_metrics(safety, q, config)
    result["state_loss"] = (
        float(0.5 * np.sum(q)) if error_vector is None else float(0.5 * (error_vector @ error_vector))
    )
    return result


def metrics(entry, config):
    """Actual loss is exactly half the sum of the original24 squared errors."""
    return _metrics(*_outputs(entry), config)


def build_stage_problem(anchor, Js, Jq, Je, previous, global_lower, global_upper, pose_ceiling, radius, mode, config):
    """Use the unchanged restoration QP or a hard-constrained state-vector QP."""
    Je = _jacobian(Je, np.repeat(np.arange(12), 6))
    problem = original_stage_problem(
        anchor, Js, Jq, previous, global_lower, global_upper, pose_ceiling, radius, mode, config
    )
    if mode == "restore":
        return problem
    start, stop = problem["row_blocks"]["mode_epigraph"]
    keep = np.r_[np.arange(start), np.arange(stop, len(problem["b"]))]
    problem["A"], problem["b"] = problem["A"][keep], problem["b"][keep]
    problem["row_blocks"] = {
        name: [a if a < start else a - (stop - start), b if b <= start else b - (stop - start)]
        for name, (a, b) in problem["row_blocks"].items()
        if name != "mode_epigraph"
    }
    E = normalized_error_vector(anchor)
    problem.update(B=finite(POSE_DAMPING * np.eye(72) + Je.T @ Je), g=finite(Je.T @ E), initial_t=0.0)
    return problem


def search(evaluate, historical_initial, Js, Jq, Je, previous, input_min, input_max, config):
    """Search finite full-H12 responses, always reserving the final initial repeat."""
    config = validate_config(config)
    initial_command = finite(historical_initial["commands"], (12, 6)).copy()
    historical_initial = _entry(historical_initial, initial_command)
    sp, qp_periods = causal_periods()
    Js, Jq = _jacobian(Js, sp), _jacobian(Jq, qp_periods)
    error_periods = np.repeat(np.arange(12), 6)
    Je = _jacobian(Je, error_periods)
    previous, input_min, input_max = (finite(value, (6,)).copy() for value in (previous, input_min, input_max))
    _require(np.all(input_min < input_max), "Strict original input bounds")
    global_lower = np.maximum(np.tile(input_min, (12, 1)), initial_command - config["global_displacement_limit"])
    global_upper = np.minimum(np.tile(input_max, (12, 1)), initial_command + config["global_displacement_limit"])
    global_lower[0] = np.maximum(global_lower[0], previous - 0.25)
    global_upper[0] = np.minimum(global_upper[0], previous + 0.25)
    _require(
        np.all(initial_command >= global_lower - 1e-12)
        and np.all(initial_command <= global_upper + 1e-12)
        and np.all(np.abs(np.diff(np.vstack((previous, initial_command)), axis=0)) <= 0.25 + 1e-12),
        "Original initial command admission",
    )
    initial_safety, initial_q = _outputs(historical_initial)
    _require(
        historical_initial["audit"]["feasible"] and np.all(initial_safety >= config["safety_buffer"]),
        "Original safety-buffer-qualified initial anchor",
    )
    ceiling = initial_global_envelope(initial_q, config)
    answer = dict(
        pose_ceiling_policy=POSE_CEILING_POLICY,
        pose_objective_policy=POSE_OBJECTIVE_POLICY,
        pose_damping=POSE_DAMPING,
        feedback_policy=FEEDBACK_POLICY,
        rejected_update_count=0,
        config=config,
        global_lower=global_lower,
        global_upper=global_upper,
        pose_ceiling=ceiling,
        evaluated_candidates=[],
        request_ledger=[],
        stages=[],
        initial_comparison=None,
        repeatability=None,
        final_anchor_index=0,
        witness_indices=[],
        qualified_witness_indices=[],
        selected_response_index=None,
        selected=None,
        termination_reason=None,
        prediction_calls=0,
        response_evaluations=0,
        unique_command_count=0,
        total_stages=0,
        total_qp_iterations=0,
        control_qualified=False,
    )
    seen = set()

    def request(commands, role, stage_index=None, alpha_index=None, alpha=None):
        commands = finite(commands, (12, 6)).copy()
        index = len(answer["request_ledger"])
        _require(index < config["max_responses"], "No response outside the registered budget")
        row = dict(
            request_index=index,
            response_index=None,
            role=role,
            stage_index=stage_index,
            alpha_index=alpha_index,
            alpha=alpha,
            commands=commands.copy(),
            calls_before=index,
            calls_after=index + 1,
            completed=False,
            charged=True,
        )
        answer["request_ledger"].append(row)
        answer["prediction_calls"] += 1
        seen.add(commands.tobytes())
        answer["unique_command_count"] = len(seen)
        try:
            entry = _entry(evaluate(commands.copy()), commands)
        except Exception as error:
            row["error"] = dict(type=type(error).__name__, message=str(error))
            raise SearchEvaluationError("Trajectory response failed; charged prefix retained", answer) from error
        response_index = len(answer["evaluated_candidates"])
        answer["evaluated_candidates"].append(entry)
        row.update(response_index=response_index, completed=True)
        answer["response_evaluations"] += 1
        return response_index

    def original_witness(entry):
        safety, q = _outputs(entry)
        return bool(entry["audit"]["feasible"] and np.all(q <= 1)), bool(np.all(safety >= config["safety_buffer"]))

    anchor_index = request(initial_command, "initial")
    answer["initial_comparison"] = canonical_initial_comparison(historical_initial, answer["evaluated_candidates"][0])
    if not answer["initial_comparison"]["exact"]:
        answer["termination_reason"] = "historical_initial_mismatch"
    elif all(original_witness(answer["evaluated_candidates"][0])):
        answer["termination_reason"] = "initial_witness"
    else:
        radius = config["radius_initial"]
        for stage_index in range(config["max_stages"]):
            if answer["prediction_calls"] >= config["max_responses"] - 1:
                answer["termination_reason"] = "response_budget_exhausted"
                break
            anchor = answer["evaluated_candidates"][anchor_index]
            before = metrics(anchor, config)
            mode = "restore" if before["reserve"] < config["reserve_entry_fraction"] else "pose"
            stage = dict(
                stage_index=stage_index,
                mode=mode,
                anchor_before_index=anchor_index,
                anchor_after_index=anchor_index,
                radius_before=radius,
                radius_after=None,
                anchor_metrics_before=before,
                anchor_metrics_after=before.copy(),
                qp=None,
                attempts=[],
                accepted_response_index=None,
                secant_update=None,
                rejected_update=None,
                termination=None,
            )
            answer["stages"].append(stage)
            answer["total_stages"] = len(answer["stages"])
            problem = build_stage_problem(
                anchor, Js, Jq, Je, previous, global_lower, global_upper, ceiling, radius, mode, config
            )
            qp = solve_epigraph_qp(
                problem["B"],
                problem["g"],
                problem["A"],
                problem["b"],
                problem["initial_t"],
                max_iterations=100,
                certificate_tolerance=1e-6,
            )
            stage["qp"] = qp
            answer["total_qp_iterations"] += qp["solver_nit"]
            accepted, witness = False, False
            if qp["qualified"]:
                point = finite(qp["point"], (73,))
                x = finite(anchor["commands"], (12, 6))
                safety, q = _outputs(anchor)
                E = normalized_error_vector(anchor)
                metric_key = "violation" if mode == "restore" else "state_loss"
                minimum_predicted = config["minimum_predicted_reduction"]
                minimum_actual = config["minimum_actual_reduction"]
                if mode == "pose":
                    minimum_predicted = min(
                        minimum_predicted, config["pose_reduction_fraction"] * before["pose_excess"]
                    )
                    minimum_actual = min(minimum_actual, config["pose_reduction_fraction"] * before["pose_excess"])
                for alpha_index, alpha in enumerate(config["alphas"]):
                    command = finite(x + alpha * point[:72].reshape(12, 6))
                    step = command - x
                    change = np.diff(np.vstack((previous, command)), axis=0)
                    gates = dict(
                        box=bool(np.all(command >= input_min - 1e-12) and np.all(command <= input_max + 1e-12)),
                        rate=bool(np.all(np.abs(change) <= 0.25 + 1e-12)),
                        global_=bool(
                            np.all(command >= global_lower - 1e-12) and np.all(command <= global_upper + 1e-12)
                        ),
                        trust=bool(np.max(np.abs(step)) <= radius + 1e-12),
                        nonzero=bool(np.any(step != 0)),
                        unique=command.tobytes() not in seen,
                    )
                    gates["global"] = gates.pop("global_")
                    predicted = _metrics(
                        safety + Js @ step.ravel(), q + Jq @ step.ravel(), config, E + Je @ step.ravel()
                    )
                    predicted_reduction = before[metric_key] - predicted[metric_key]
                    attempt = dict(
                        alpha_index=alpha_index,
                        alpha=alpha,
                        commands=command.copy(),
                        actual_step=step.copy(),
                        admission=gates,
                        predicted_metrics=predicted,
                        predicted_reduction=predicted_reduction,
                        response_index=None,
                        actual_metrics=None,
                        actual_reduction=None,
                        ratio=None,
                        accepted=False,
                        acceptance_reason=None,
                    )
                    stage["attempts"].append(attempt)
                    if not all(gates.values()):
                        attempt["acceptance_reason"] = "command_preflight_failed"
                        continue
                    if predicted_reduction < minimum_predicted:
                        attempt["acceptance_reason"] = "predicted_reduction_insufficient"
                        continue
                    if answer["prediction_calls"] >= config["max_responses"] - 1:
                        attempt["acceptance_reason"] = "response_budget_exhausted"
                        stage["termination"] = "response_budget_exhausted"
                        break
                    index = request(command, "trial", stage_index, alpha_index, alpha)
                    actual_entry = answer["evaluated_candidates"][index]
                    actual = metrics(actual_entry, config)
                    actual_reduction = before[metric_key] - actual[metric_key]
                    ratio = actual_reduction / predicted_reduction
                    actual_safety, actual_q = _outputs(actual_entry)
                    witness = all(original_witness(actual_entry))
                    admission = bool(
                        actual_entry["audit"]["feasible"]
                        and np.all(actual_safety >= config["safety_buffer"])
                        and np.all(actual_q <= ceiling)
                    )
                    if witness:
                        reason, accepted = "original_witness", True
                    elif not admission:
                        reason = "actual_admission_failed"
                    elif actual_reduction < minimum_actual:
                        reason = "actual_reduction_insufficient"
                    elif ratio < config["accept_ratio"]:
                        reason = "ratio_insufficient"
                    else:
                        reason, accepted = "accepted_reduction", True
                    attempt.update(
                        response_index=index,
                        actual_metrics=actual,
                        actual_reduction=actual_reduction,
                        ratio=ratio,
                        accepted=accepted,
                        acceptance_reason=reason,
                    )
                    if accepted:
                        Js, safety_update = causal_secant_update(Js, step.ravel(), actual_safety - safety, sp, config)
                        Jq, pose_update = causal_secant_update(Jq, step.ravel(), actual_q - q, qp_periods, config)
                        Je, error_update = causal_secant_update(
                            Je, step.ravel(), normalized_error_vector(actual_entry) - E, error_periods, config
                        )
                        stage["secant_update"] = dict(
                            step=step.copy(), safety=safety_update, pose=pose_update, error_vector=error_update
                        )
                        stage.update(
                            anchor_after_index=index,
                            anchor_metrics_after=actual,
                            accepted_response_index=index,
                            termination="qualified_witness" if witness else "accepted",
                        )
                        anchor_index = index
                        if ratio < config["shrink_ratio"]:
                            radius *= 0.5
                        elif ratio >= config["expand_ratio"] and np.max(np.abs(step)) >= 0.8 * radius:
                            radius = min(config["radius_max"], 2 * radius)
                        break
            if not accepted:
                source = choose_rejected_source(stage, answer["evaluated_candidates"], ceiling, config)
                if source is not None:
                    source_index = source["response_index"]
                    observed_safety, observed_q = _outputs(answer["evaluated_candidates"][source_index])
                    anchor_safety, anchor_q = _outputs(anchor)
                    feedback_step = finite(source["actual_step"], (12, 6))
                    Js, safety_update = causal_secant_update(
                        Js, feedback_step.ravel(), observed_safety - anchor_safety, sp, config
                    )
                    Jq, pose_update = causal_secant_update(
                        Jq, feedback_step.ravel(), observed_q - anchor_q, qp_periods, config
                    )
                    Je, error_update = causal_secant_update(
                        Je,
                        feedback_step.ravel(),
                        normalized_error_vector(answer["evaluated_candidates"][source_index])
                        - normalized_error_vector(anchor),
                        error_periods,
                        config,
                    )
                    stage["rejected_update"] = dict(
                        source_response_index=source_index,
                        alpha=source["alpha"],
                        step=feedback_step.copy(),
                        safety=safety_update,
                        pose=pose_update,
                        error_vector=error_update,
                    )
                    answer["rejected_update_count"] += 1
                radius *= 0.5
                if stage["termination"] is None:
                    stage["termination"] = "all_attempts_rejected" if qp["qualified"] else "qp_unqualified"
            stage["radius_after"] = radius
            answer["final_anchor_index"] = anchor_index
            if witness and accepted:
                answer["termination_reason"] = "qualified_witness"
                break
            if radius < config["radius_min"]:
                answer["termination_reason"] = "radius_below_minimum"
                break
            if stage["termination"] == "response_budget_exhausted":
                answer["termination_reason"] = "response_budget_exhausted"
                break
        if answer["termination_reason"] is None:
            answer["termination_reason"] = "stage_budget_exhausted"
    repeat_index = request(initial_command, "initial_repeat")
    answer["repeatability"] = canonical_initial_comparison(
        answer["evaluated_candidates"][0], answer["evaluated_candidates"][repeat_index]
    )
    for index, entry in enumerate(answer["evaluated_candidates"]):
        original, buffered = original_witness(entry)
        if original:
            answer["witness_indices"].append(index)
            if buffered and answer["initial_comparison"]["exact"] and answer["repeatability"]["exact"]:
                answer["qualified_witness_indices"].append(index)
    if answer["qualified_witness_indices"]:
        index = answer["qualified_witness_indices"][0]
        answer.update(selected_response_index=index, selected=deepcopy(answer["evaluated_candidates"][index]))
    elif not answer["repeatability"]["exact"]:
        answer["termination_reason"] = "repeat_mismatch"
    return answer
