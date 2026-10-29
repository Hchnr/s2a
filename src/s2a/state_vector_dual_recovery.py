"""Recover pose duals on a fixed tight-slack subset under original KKT gates.

This changes only the one certificate fit after the unchanged scaled primal
solver. It never substitutes subset feasibility for the complete primal audit.
"""

from copy import deepcopy

import numpy as np
from scipy.optimize import minimize, nnls

from s2a.accepted_point_sqp_core import _array, _inf, _scalar, _spd
from s2a.geometry import finite
from s2a.restored_anchor_sqp_plan import canonical_initial_comparison
from s2a.state_vector_objective_scale import (
    FEEDBACK_POLICY,
    POSE_CEILING_POLICY,
    POSE_DAMPING,
    POSE_OBJECTIVE_POLICY,
    QP_OBJECTIVE_SCALING_POLICY,
    SearchEvaluationError,
    _entry,
    _jacobian,
    _metrics,
    _outputs,
    _require,
    build_stage_problem,
    causal_periods,
    causal_secant_update,
    choose_rejected_source,
    initial_global_envelope,
    metrics,
    normalized_error_vector,
    validate_config,
)
from s2a.state_vector_objective_scale import objective_divisor as objective_divisor

DUAL_RECOVERY_POLICY = "pose_slack_restricted_nnls_restore_full_active"


def solve_recovered_epigraph_qp(
    B,
    g,
    A,
    b,
    initial_t,
    *,
    objective_divisor=1.0,
    dual_recovery_mode="pose",
    max_iterations=100,
    certificate_tolerance=1e-6,
):
    """Keep the primal solve while fitting pose duals only on the fixed tight set.

    Complete active indices, the full NNLS iteration budget, and every original
    constraint/KKT gate retain their meanings. A pose subset fit and its mu/C
    algebraic dual are explicitly distinguished from the old full-active NNLS.
    Empty recovery sets make no fit call; nnls_fit_attempts records actual calls,
    including exceptions, independently of the legacy numerical success flag.
    """
    if dual_recovery_mode not in ("pose", "restore"):
        raise ValueError("Explicit pose or restore dual recovery mode required")
    B = _spd(B)
    width = len(B)
    g, A, b = _array(g, "g"), _array(A, "A"), _array(b, "b")
    initial_t = _scalar(initial_t, "initial_t")
    tolerance = _scalar(certificate_tolerance, "certificate_tolerance")
    divisor = _scalar(objective_divisor, "objective_divisor")
    if divisor <= 0.0:
        raise ValueError("Positive finite whole-objective divisor required")
    if (
        g.shape != (width,)
        or A.ndim != 2
        or A.shape[1] != width + 1
        or b.shape != (len(A),)
        or not all(np.all(np.isfinite(value)) for value in (g, A, b))
        or type(max_iterations) is not int
        or max_iterations < 1
        or tolerance <= 0.0
    ):
        raise ValueError("Finite consistent QP arrays and positive explicit budgets/tolerance required")
    row_scale = np.maximum(1.0, np.maximum(np.max(np.abs(A), axis=1), np.abs(b)))
    scaled_A, scaled_b = A / row_scale[:, None], b / row_scale
    initial = np.r_[np.zeros(width), initial_t]
    result = dict(
        dual_recovery_mode=dual_recovery_mode,
        full_active_threshold=10.0 * tolerance,
        recovery_threshold=(0.01 if dual_recovery_mode == "pose" else 10.0) * tolerance,
        recovery_indices=[],
        nnls_fit_attempts=0,
        solver_objective_divisor=divisor,
        solver_objective_value=None,
        objective_scaled_multipliers=None,
        objective_scaled_multipliers_provenance=(
            "unscaled_gradient_subset_nnls_certificate_divided_by_objective_divisor"
            if dual_recovery_mode == "pose"
            else "original_nnls_certificate_divided_by_objective_divisor"
        ),
        B=B.copy(),
        point=None,
        solver_success=False,
        solver_status=None,
        solver_message=None,
        solver_nit=0,
        solver_exception=None,
        qualified=False,
        objective=None,
        multipliers=None,
        scaled_multipliers=None,
        row_scale=row_scale.copy(),
        active_indices=[],
        primal_min=None,
        unscaled_primal_min=None,
        stationarity_relative=None,
        stationarity_residual_inf=None,
        complementarity_relative=None,
        complementarity_residual_inf=None,
        complementarity_sum_gap=None,
        certificate_tolerance=tolerance,
        nnls_success=False,
        nnls_error=None,
        nnls_residual_norm=None,
        nnls_max_iterations=None,
        qualification_failures=[],
    )

    def objective(point):
        with np.errstate(over="raise", invalid="raise"):
            d = point[:width]
            return float(point[width] + g @ d + 0.5 * (d @ (B @ d)))

    def gradient(point):
        with np.errstate(over="raise", invalid="raise"):
            return np.r_[g + B @ point[:width], 1.0]

    def constraint(point):
        with np.errstate(over="raise", invalid="raise"):
            return scaled_A @ point + scaled_b

    def solver_objective(point):
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            return float(np.divide(objective(point), divisor))

    def solver_gradient(point):
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            return gradient(point) / divisor

    try:
        solved = minimize(
            solver_objective,
            initial,
            jac=solver_gradient,
            method="SLSQP",
            constraints=[dict(type="ineq", fun=constraint, jac=lambda point: scaled_A.copy())],
            options=dict(maxiter=max_iterations, ftol=1e-14, disp=False),
        )
    except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as error:
        result.update(
            solver_message=str(error),
            solver_exception=f"{type(error).__name__}: {error}",
            qualification_failures=["solver_exception"],
        )
        return result
    result.update(
        solver_success=bool(solved.success),
        solver_status=int(solved.status),
        solver_message=str(solved.message),
        solver_nit=int(solved.nit),
    )
    if not result["solver_success"]:
        result["qualification_failures"].append("solver_unsuccessful")
    point = _array(solved.x, "solver point")
    if point.shape != (width + 1,) or not np.all(np.isfinite(point)):
        result["qualification_failures"].append("invalid_solver_point")
        return result
    result["point"] = point.copy()
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            value, grad, slack = objective(point), gradient(point), constraint(point)
            raw_slack = A @ point + b
            if not all(np.all(np.isfinite(item)) for item in (value, grad, slack, raw_slack)):
                raise FloatingPointError("Nonfinite QP certificate inputs")
            active = np.flatnonzero(slack <= result["full_active_threshold"])
            recovery = active[slack[active] <= result["recovery_threshold"]]
            result.update(
                objective=value,
                solver_objective_value=float(np.divide(value, divisor)),
                active_indices=active.tolist(),
                recovery_indices=recovery.tolist(),
                primal_min=float(np.min(slack)) if len(slack) else None,
                unscaled_primal_min=float(np.min(raw_slack)) if len(raw_slack) else None,
                nnls_max_iterations=3 * max(len(active), 1),
            )
            if len(slack) and result["primal_min"] < -tolerance:
                result["qualification_failures"].append("primal_infeasible")
            mu = np.zeros(len(A))
            try:
                if len(recovery):
                    result["nnls_fit_attempts"] += 1
                    fitted, residual = nnls(scaled_A[recovery].T, grad, maxiter=result["nnls_max_iterations"])
                    if fitted.shape != (len(recovery),) or not np.all(np.isfinite(fitted)):
                        raise ValueError("NNLS returned invalid multipliers")
                    mu[recovery] = fitted
                else:
                    residual = float(np.linalg.norm(grad))
                if not np.isfinite(residual):
                    raise ValueError("NNLS returned a nonfinite residual")
            except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as error:
                result.update(nnls_error=f"{type(error).__name__}: {error}")
                result["qualification_failures"].append("nnls_failed")
                return result
            multipliers = mu / row_scale
            dual_force = scaled_A.T @ mu
            stationarity = _inf(grad - dual_force)
            products = mu * slack
            complementarity = _inf(products)
            relative_stationarity = stationarity / max(1.0, _inf(grad), _inf(dual_force))
            relative_complementarity = complementarity / max(1.0, abs(value))
            gap = float(np.sum(products))
            if not all(
                np.all(np.isfinite(item))
                for item in (multipliers, dual_force, products, relative_stationarity, relative_complementarity, gap)
            ):
                raise FloatingPointError("Nonfinite QP certificate")
            result.update(
                multipliers=multipliers,
                scaled_multipliers=mu,
                objective_scaled_multipliers=mu / divisor,
                nnls_success=True,
                nnls_residual_norm=float(residual),
                stationarity_residual_inf=stationarity,
                stationarity_relative=relative_stationarity,
                complementarity_residual_inf=complementarity,
                complementarity_relative=relative_complementarity,
                complementarity_sum_gap=gap,
            )
            if np.any(mu < 0.0) or np.any(multipliers < 0.0):
                result["qualification_failures"].append("negative_multiplier")
            if relative_stationarity > tolerance:
                result["qualification_failures"].append("stationarity_failed")
            if relative_complementarity > tolerance:
                result["qualification_failures"].append("complementarity_failed")
    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
        result["certificate_error"] = f"{type(error).__name__}: {error}"
        result["qualification_failures"].append("invalid_certificate")
        return result
    result["qualified"] = not result["qualification_failures"]
    return result


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
        qp_objective_scaling_policy=QP_OBJECTIVE_SCALING_POLICY,
        dual_recovery_policy=DUAL_RECOVERY_POLICY,
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
            qp = solve_recovered_epigraph_qp(
                problem["B"],
                problem["g"],
                problem["A"],
                problem["b"],
                problem["initial_t"],
                objective_divisor=objective_divisor(problem, radius, mode),
                dual_recovery_mode=mode,
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
