"""Bounded causal trajectory search with alternating reserve and pose objectives.

All responses start from the same measured state through the supplied callback.
Numerical witnesses remain distinct from physical control qualification.
"""

from copy import deepcopy

import numpy as np

from s2a.accepted_point_sqp_core import solve_epigraph_qp
from s2a.geometry import finite
from s2a.restored_anchor_sqp_plan import canonical_initial_comparison
from s2a.terminal_only_tail_repair import ENTRY_FIELDS

DEFAULT_CONFIG = dict(
    seed=42,
    max_stages=24,
    max_responses=122,
    alphas=[1.0, 0.5, 0.25, 0.125, 0.0625],
    radius_initial=0.005,
    radius_min=0.00015625,
    radius_max=0.02,
    global_displacement_limit=0.1,
    reserve_target=1e-5,
    reserve_entry_fraction=0.9,
    terminal_time_scale=0.1,
    pose_ceiling_fraction=0.01,
    pose_reduction_fraction=0.1,
    safety_buffer=1e-8,
    accept_ratio=0.1,
    shrink_ratio=0.25,
    expand_ratio=0.75,
    minimum_predicted_reduction=1e-4,
    minimum_actual_reduction=1e-4,
    secant_min_squared_step=1e-12,
    secant_relative_limit=2.0,
    secant_row_norm_floor=1e-6,
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_config(config):
    """Require explicit finite bounds; smaller synthetic test budgets are legal."""
    config = deepcopy(dict(config))
    _require(set(config) == set(DEFAULT_CONFIG), "Complete explicit search configuration")
    _require(type(config["seed"]) is int and config["seed"] == 42, "Explicit seed42")
    _require(type(config["max_stages"]) is int and 1 <= config["max_stages"] <= 24, "Bounded stage budget")
    _require(type(config["max_responses"]) is int and 2 <= config["max_responses"] <= 122, "Initial and repeat budget")
    _require(tuple(config["alphas"]) == (1.0, 0.5, 0.25, 0.125, 0.0625), "Fixed five-alpha search")
    for key in set(config) - {"seed", "max_stages", "max_responses", "alphas"}:
        _require(float(finite(config[key], ())) > 0, f"Positive finite {key}")
    _require(config["radius_min"] <= config["radius_initial"] <= config["radius_max"], "Ordered trust radii")
    _require(0 < config["accept_ratio"] <= config["shrink_ratio"] < config["expand_ratio"] < 1, "Ordered ratios")
    _require(config["reserve_entry_fraction"] < 1, "Reserve mode entry below the target")
    _require(config["pose_reduction_fraction"] < 1, "Pose reduction fraction below one")
    return config


def causal_periods():
    """The unchanged safety ordering has four groups and one terminal row."""
    periods = np.arange(12)
    return np.r_[
        np.repeat(periods, 175), np.repeat(periods, 175), np.repeat(periods, 7), np.repeat(periods, 7), 11
    ], np.repeat(periods, 2)


def _causal_mask(periods):
    return np.repeat(np.arange(12), 6)[None, :] <= np.asarray(periods)[:, None]


def _jacobian(value, periods):
    value = finite(value, (len(periods), 72)).copy()
    _require(np.all(value[~_causal_mask(periods)] == 0), "Future command derivatives must already be exactly zero")
    return value


def _outputs(entry):
    safety = finite(entry["safety_slack"], (4369,))
    error = finite(entry["tcp_errors"], (12, 6))
    q = finite(
        np.column_stack((np.sum(error[:, :3] ** 2, axis=1) / 0.01**2, np.sum(error[:, 3:] ** 2, axis=1) / 0.05**2))
    )
    _require(np.array_equal(q, finite(entry["normalized_squared_errors"], (12, 2))), "Exact original pose-cap algebra")
    return safety, q.ravel()


def _entry(entry, commands):
    _require(set(entry) == set(ENTRY_FIELDS), "Complete original eleven response fields")
    entry = deepcopy(entry)
    observed = finite(entry["commands"], (12, 6))
    _require(observed.tobytes() == commands.tobytes(), "Response must describe the exact requested command")
    _outputs(entry)
    _require(type(entry["audit"]["feasible"]) is bool, "Strict original joint-audit flag")
    return entry


def _weights(config):
    weights = np.ones(4369)
    weights[-1] = config["terminal_time_scale"]
    return weights


def _metrics(safety, q, config):
    reserve = float(np.min(_weights(config) * (safety - config["safety_buffer"]) / config["reserve_target"]))
    return dict(
        reserve=reserve,
        violation=float(max(0.0, 1.0 - reserve)),
        pose_excess=float(max(0.0, np.max(q) - 1.0)),
        safety_min=float(np.min(safety)),
        pose_max=float(np.max(q)),
    )


def metrics(entry, config):
    """Dimensionless reserve uses the fixed terminal time conversion."""
    return _metrics(*_outputs(entry), config)


def build_stage_problem(anchor, Js, Jq, previous, global_lower, global_upper, pose_ceiling, radius, mode, config):
    """Rebuild one hard-constrained local QP without invoking a response."""
    _require(mode in ("restore", "pose"), "Explicit restore or pose objective")
    x = finite(anchor["commands"], (12, 6))
    safety, q = _outputs(anchor)
    Js, Jq = finite(Js, (4369, 72)), finite(Jq, (24, 72))
    previous = finite(previous, (6,))
    lo = np.maximum(finite(global_lower, (12, 6)), x - radius)
    hi = np.minimum(finite(global_upper, (12, 6)), x + radius)
    _require(np.all(lo <= hi), "Nonempty trust-region intersection")
    weights = _weights(config)
    reserve = metrics(anchor, config)["reserve"]
    floor = np.full(4369, config["safety_buffer"])
    if mode == "pose":
        floor += min(0.5 * reserve, 0.5) * config["reserve_target"] / weights
    identity = np.eye(72)
    difference, center = identity - np.eye(72, k=-6), np.r_[previous, np.zeros(66)]
    objective_matrix = weights[:, None] * Js / config["reserve_target"] if mode == "restore" else -Jq
    objective_offset = (
        weights * (safety - config["safety_buffer"]) / config["reserve_target"] - 1 if mode == "restore" else 1 - q
    )
    blocks = (
        ("safety", Js, safety - floor, False),
        ("pose_ceiling", -Jq, finite(pose_ceiling, (24,)) - q, False),
        ("mode_epigraph", objective_matrix, objective_offset, True),
        ("rate_lower", difference, 0.25 + difference @ x.ravel() - center, False),
        ("rate_upper", -difference, 0.25 - difference @ x.ravel() + center, False),
        ("command_lower", identity, (x - lo).ravel(), False),
        ("command_upper", -identity, (hi - x).ravel(), False),
    )
    matrices, offsets, row_blocks, start = [], [], {}, 0
    for name, matrix, offset, epigraph in blocks:
        matrices.append(np.c_[matrix, np.full(len(offset), float(epigraph))])
        offsets.append(offset)
        row_blocks[name] = [start, start + len(offset)]
        start += len(offset)
    matrices.append(np.r_[np.zeros(72), 1.0][None])
    offsets.append(np.zeros(1))
    row_blocks["epigraph"] = [start, start + 1]
    return dict(
        A=finite(np.vstack(matrices)),
        b=finite(np.concatenate(offsets)),
        B=identity,
        g=np.zeros(72),
        initial_t=metrics(anchor, config)["violation" if mode == "restore" else "pose_excess"],
        command_lower=lo.copy(),
        command_upper=hi.copy(),
        safety_floor=floor,
        row_blocks=row_blocks,
        mode=mode,
    )


def causal_secant_update(J, step, observed_change, row_periods, config):
    """Apply bounded Broyden corrections using each row's causal step prefix."""
    J = _jacobian(J, row_periods)
    step, change = finite(step, (72,)), finite(observed_change, (len(J),))
    masked = _causal_mask(row_periods) * step
    denominator = np.sum(masked**2, axis=1)
    residual = change - J @ step
    row_norm = np.linalg.norm(J, axis=1)
    valid = denominator >= config["secant_min_squared_step"]
    correction = np.zeros_like(J)
    correction[valid] = residual[valid, None] * masked[valid] / denominator[valid, None]
    correction_norm = np.linalg.norm(correction, axis=1)
    theta = np.zeros(len(J))
    theta[valid] = 1.0
    nonzero = valid & (correction_norm > 0)
    theta[nonzero] = np.minimum(
        1.0,
        config["secant_relative_limit"]
        * np.maximum(row_norm[nonzero], config["secant_row_norm_floor"])
        / correction_norm[nonzero],
    )
    updated = valid & (correction_norm > 0)
    result = finite(J + theta[:, None] * correction)
    _require(np.all(result[~_causal_mask(row_periods)] == 0), "Secant updates preserve exact causality")
    return result, dict(
        denominator=denominator,
        residual=residual,
        theta=theta,
        correction_norm=correction_norm,
        row_norm=row_norm,
        updated=updated,
    )


class SearchEvaluationError(RuntimeError):
    """Retain the charged response prefix when the supplied callback fails."""

    def __init__(self, message, answer):
        super().__init__(message)
        self.answer = answer


def search(evaluate, historical_initial, Js, Jq, previous, input_min, input_max, config):
    """Search finite full-H12 responses, always reserving the final initial repeat."""
    config = validate_config(config)
    initial_command = finite(historical_initial["commands"], (12, 6)).copy()
    historical_initial = _entry(historical_initial, initial_command)
    sp, qp_periods = causal_periods()
    Js, Jq = _jacobian(Js, sp), _jacobian(Jq, qp_periods)
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
    ceiling = np.maximum(initial_q, 1.0) * (1 + config["pose_ceiling_fraction"])
    answer = dict(
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
                termination=None,
            )
            answer["stages"].append(stage)
            answer["total_stages"] = len(answer["stages"])
            problem = build_stage_problem(
                anchor, Js, Jq, previous, global_lower, global_upper, ceiling, radius, mode, config
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
                metric_key = "violation" if mode == "restore" else "pose_excess"
                minimum_predicted = config["minimum_predicted_reduction"]
                minimum_actual = config["minimum_actual_reduction"]
                if mode == "pose":
                    minimum_predicted = min(minimum_predicted, config["pose_reduction_fraction"] * before[metric_key])
                    minimum_actual = min(minimum_actual, config["pose_reduction_fraction"] * before[metric_key])
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
                    predicted = _metrics(safety + Js @ step.ravel(), q + Jq @ step.ravel(), config)
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
                        stage["secant_update"] = dict(step=step.copy(), safety=safety_update, pose=pose_update)
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
