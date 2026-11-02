"""Strict independent native decisions for the fixed cap-distance arithmetic."""

import numpy as np

from scripts.tracking.trajectory_search.cap_distance_audit_policy import POLICY, cap_loss
from scripts.tracking.trajectory_search.check_tracking_v5_state_vector_gn_search import fresh_metrics as sumq_metrics
from scripts.verification.trajectory_search.check_tracking_v5_saved_direction_backtracking import NUMERIC_TOLERANCE
from scripts.verification.trajectory_search.check_tracking_v5_tail_translation_reserve_v2 import semantic_compare
from scripts.verification.trajectory_search.directional_derivative_audit import require
from scripts.verification.trajectory_search.tail_translation_reserve_audit import _comparison_report


def fresh_metrics(metric, config):
    result = sumq_metrics(metric, config)
    result["cap_loss"] = cap_loss(np.asarray(metric["normalized_squared_errors"], dtype=float).ravel())
    return result


def verify_cap_fresh_decisions(answer, rollouts, metrics, config, *, diagnostics=None):
    """Strictly reclassify actual loss, safety, priority, radius and full caps."""
    require(answer["pose_objective_policy"] == POLICY, "Fixed independently audited cap-distance policy")
    require(diagnostics is None or isinstance(diagnostics, list), "Mutable diagnostic event list required")

    def reserve_row(metric):
        safety = np.asarray(metric["safety_slack"], dtype=float)
        weights = np.ones(4369)
        weights[-1] = config["terminal_time_scale"]
        weighted = weights * (safety - config["safety_buffer"])
        values = weighted / config["reserve_target"]
        index = int(np.argmin(values))
        offset = 0
        for block, count, columns in (
            ("inner_lower", 2100, 7),
            ("inner_upper", 2100, 7),
            ("braking_lower", 84, 7),
            ("braking_upper", 84, 7),
            ("terminal", 1, 1),
        ):
            if index < offset + count:
                local = index - offset
                local_index = [local // columns, local % columns] if columns == 7 else [local]
                break
            offset += count
        return dict(
            flat_index=index,
            block=block,
            local_index=local_index,
            raw_slack=float(safety[index]),
            weight=float(weights[index]),
            safety_buffer=float(config["safety_buffer"]),
            weighted_slack=float(weights[index] * safety[index]),
            weighted_buffered_slack=float(weighted[index]),
            reserve=float(values[index]),
        )

    def gate_inputs(metric, *, saved=False):
        safety = np.asarray(metric["safety_slack"], dtype=float)
        squared = np.asarray(metric["normalized_squared_errors"], dtype=float).reshape(12, 2)
        joint = bool(metric["audit"]["feasible"] if saved else metric["joint_feasible"])
        buffered = bool(np.all(safety >= config["safety_buffer"]))
        pose = bool(np.all(squared <= ceiling))
        return dict(
            joint_feasible=joint,
            feasible=bool(metric["feasible"]),
            safety_buffer=float(config["safety_buffer"]),
            safety_min_raw=float(np.min(safety)),
            safety_min_buffer_margin=float(np.min(safety - config["safety_buffer"])),
            safety_buffer_passed=buffered,
            safe=bool(joint and buffered),
            witness=bool(metric["feasible"] and joint and buffered),
            normalized_squared_errors=squared.tolist(),
            pose_ceiling=np.asarray(ceiling, dtype=float).tolist(),
            pose_ceiling_passed=pose,
            pose_ceiling_min_margin=float(np.min(ceiling - squared)),
            pose_ceiling_flat_argmin=int(np.argmin(ceiling - squared)),
            admitted=bool(joint and buffered and pose),
        )

    def thresholds(base, mode):
        minimum = config["minimum_actual_reduction"]
        if mode == "pose":
            minimum = min(minimum, config["pose_reduction_fraction"] * base["cap_loss"])
        return dict(
            accept_ratio=float(config["accept_ratio"]),
            shrink_ratio=float(config["shrink_ratio"]),
            expand_ratio=float(config["expand_ratio"]),
            reserve_entry_fraction=float(config["reserve_entry_fraction"]),
            configured_minimum_actual_reduction=float(config["minimum_actual_reduction"]),
            minimum_actual_reduction=float(minimum),
        )

    require(len(metrics) == len(rollouts) == len(answer["evaluated_candidates"]), "Every actual response replayed")
    ceiling = np.asarray(answer["pose_ceiling"]).reshape(12, 2)
    for stage_number, stage in enumerate(answer["stages"]):
        base = fresh_metrics(metrics[stage["anchor_before_index"]], config)
        mode = "restore" if base["reserve"] < config["reserve_entry_fraction"] else "pose"
        if diagnostics is not None:
            saved_base_entry = answer["evaluated_candidates"][stage["anchor_before_index"]]
            saved_base = fresh_metrics(saved_base_entry, config)
            saved_mode = "restore" if saved_base["reserve"] < config["reserve_entry_fraction"] else "pose"
            stage_event = dict(
                event="stage",
                stage_index=int(stage.get("stage_index", stage_number)),
                anchor_before_index=int(stage["anchor_before_index"]),
                fresh_mode=mode,
                saved_mode=stage["mode"],
                saved_mode_from_metrics=saved_mode,
                fresh_base_metrics=base.copy(),
                saved_base_metrics=saved_base.copy(),
                merit_field="violation" if mode == "restore" else "cap_loss",
                fresh_base_merit=float(base["violation" if mode == "restore" else "cap_loss"]),
                saved_base_merit=float(saved_base["violation" if mode == "restore" else "cap_loss"]),
                fresh_base_reserve_row=reserve_row(metrics[stage["anchor_before_index"]]),
                saved_base_reserve_row=reserve_row(saved_base_entry),
                thresholds=thresholds(base, mode),
                boundary_distances=dict(
                    fresh_reserve_minus_entry=float(base["reserve"] - config["reserve_entry_fraction"]),
                    saved_reserve_minus_entry=float(saved_base["reserve"] - config["reserve_entry_fraction"]),
                ),
                numeric_check="stage_mode",
                numeric_check_passed=False,
            )
            diagnostics.append(stage_event)
        require(stage["mode"] == mode, "Independent actual reserve changes the stage mode")
        if diagnostics is not None:
            stage_event["numeric_check_passed"] = True
        field = "violation" if mode == "restore" else "cap_loss"
        minimum = config["minimum_actual_reduction"]
        if mode == "pose":
            minimum = min(minimum, config["pose_reduction_fraction"] * base["cap_loss"])
        radius = stage["radius_before"]
        accepted = False
        for alpha_number, attempt in enumerate(stage["attempts"]):
            index = attempt["response_index"]
            if index is None:
                continue
            metric = metrics[index]
            actual_metric = fresh_metrics(metric, config)
            actual = base[field] - actual_metric[field]
            predicted = attempt["predicted_reduction"]  # Independently reconstructed by audit_search.
            require(predicted > 0, "Positive preregistered predicted reduction")
            ratio = actual / predicted
            if diagnostics is not None:
                saved_entry = answer["evaluated_candidates"][index]
                saved_actual = fresh_metrics(saved_entry, config)
                step_inf = float(
                    np.max(
                        np.abs(
                            np.asarray(attempt["commands"])
                            - np.asarray(answer["evaluated_candidates"][stage["anchor_before_index"]]["commands"])
                        )
                    )
                )
                attempt_event = dict(
                    event="attempt",
                    stage_index=int(stage.get("stage_index", stage_number)),
                    anchor_before_index=int(stage["anchor_before_index"]),
                    response_index=int(index),
                    alpha_index=int(attempt.get("alpha_index", alpha_number)),
                    alpha=float(attempt["alpha"]) if "alpha" in attempt else None,
                    fresh_mode=mode,
                    saved_mode=stage["mode"],
                    saved_mode_from_metrics=saved_mode,
                    fresh_base_metrics=base.copy(),
                    saved_base_metrics=saved_base.copy(),
                    fresh_actual_metrics=actual_metric.copy(),
                    saved_actual_metrics=saved_actual,
                    merit_field=field,
                    fresh_base_merit=float(base[field]),
                    saved_base_merit=float(saved_base[field]),
                    fresh_actual_merit=float(actual_metric[field]),
                    saved_actual_merit=float(saved_actual[field]),
                    actual_reduction=float(actual),
                    saved_actual_reduction=float(attempt["actual_reduction"]),
                    actual_reduction_difference=float(actual - attempt["actual_reduction"]),
                    predicted_reduction=float(predicted),
                    ratio=float(ratio),
                    saved_ratio=float(attempt["ratio"]),
                    ratio_difference=float(ratio - attempt["ratio"]),
                    saved_accepted=bool(attempt["accepted"]),
                    saved_acceptance_reason=attempt["acceptance_reason"],
                    saved_radius_after=float(stage["radius_after"]),
                    numeric_tolerance=float(NUMERIC_TOLERANCE),
                    fresh_base_reserve_row=stage_event["fresh_base_reserve_row"].copy(),
                    saved_base_reserve_row=stage_event["saved_base_reserve_row"].copy(),
                    fresh_actual_reserve_row=reserve_row(metric),
                    saved_actual_reserve_row=reserve_row(saved_entry),
                    fresh_gate_inputs=gate_inputs(metric),
                    saved_gate_inputs=gate_inputs(saved_entry, saved=True),
                    thresholds=thresholds(base, mode),
                    radius_before=float(stage["radius_before"]),
                    step_norm_inf=step_inf,
                    boundary_distances=dict(
                        actual_reduction_minus_minimum=float(actual - minimum),
                        saved_actual_reduction_minus_minimum=float(attempt["actual_reduction"] - minimum),
                        ratio_minus_accept=float(ratio - config["accept_ratio"]),
                        saved_ratio_minus_accept=float(attempt["ratio"] - config["accept_ratio"]),
                        ratio_minus_shrink=float(ratio - config["shrink_ratio"]),
                        saved_ratio_minus_shrink=float(attempt["ratio"] - config["shrink_ratio"]),
                        ratio_minus_expand=float(ratio - config["expand_ratio"]),
                        saved_ratio_minus_expand=float(attempt["ratio"] - config["expand_ratio"]),
                        step_minus_expansion_boundary=float(step_inf - 0.8 * radius),
                    ),
                    numeric_check="fresh_actual_reduction",
                    numeric_check_passed=False,
                    fresh_actual_reduction_passed=False,
                    fresh_ratio_passed=False,
                )
                diagnostics.append(attempt_event)
            semantic_compare("fresh_actual_reduction", actual, attempt["actual_reduction"], NUMERIC_TOLERANCE)
            if diagnostics is not None:
                attempt_event["fresh_actual_reduction_passed"] = True
                attempt_event["numeric_check"] = "fresh_ratio"
            semantic_compare("fresh_ratio", ratio, attempt["ratio"], NUMERIC_TOLERANCE)
            if diagnostics is not None:
                attempt_event["fresh_ratio_passed"] = True
                attempt_event["numeric_check_passed"] = True
            safe = bool(
                metric["joint_feasible"] and np.all(np.asarray(metric["safety_slack"]) >= config["safety_buffer"])
            )
            witness = bool(metric["feasible"] and safe)
            admitted = bool(safe and np.all(np.asarray(metric["normalized_squared_errors"]) <= ceiling))
            if witness:
                reason, accepted = "original_witness", True
            elif not admitted:
                reason = "actual_admission_failed"
            elif actual < minimum:
                reason = "actual_reduction_insufficient"
            elif ratio < config["accept_ratio"]:
                reason = "ratio_insufficient"
            else:
                reason, accepted = "accepted_reduction", True
            if diagnostics is not None:
                attempt_event["fresh_acceptance_reason"] = reason
                attempt_event["fresh_accepted"] = bool(accepted)
                attempt_event["numeric_check"] = "acceptance_priority"
                attempt_event["numeric_check_passed"] = False
            require(
                accepted is attempt["accepted"] and reason == attempt["acceptance_reason"],
                "Independent actual response changes an acceptance gate or priority",
            )
            if diagnostics is not None:
                attempt_event["numeric_check_passed"] = True
            if accepted:
                step = np.asarray(attempt["commands"]) - np.asarray(
                    answer["evaluated_candidates"][stage["anchor_before_index"]]["commands"]
                )
                if ratio < config["shrink_ratio"]:
                    radius *= 0.5
                elif ratio >= config["expand_ratio"] and np.max(np.abs(step)) >= 0.8 * radius:
                    radius = min(config["radius_max"], 2 * radius)
        if not accepted:
            radius *= 0.5
        if diagnostics is not None:
            stage_event["fresh_radius_after"] = float(radius)
            stage_event["saved_radius_after"] = float(stage["radius_after"])
            stage_event["radius_check_passed"] = False
        require(radius == stage["radius_after"], "Independent actual ratio changes the radius")
        if diagnostics is not None:
            stage_event["radius_check_passed"] = True
    repeat = _comparison_report(
        dict(rollout=rollouts[0], metrics=metrics[0]), dict(rollout=rollouts[-1], metrics=metrics[-1]), len(metrics) - 1
    )
    require(repeat["exact"] is answer["repeatability"]["exact"], "Independent full initial repeat exact")
    qualified = [
        i
        for i, metric in enumerate(metrics)
        if metric["feasible"] and np.all(np.asarray(metric["safety_slack"]) >= config["safety_buffer"])
    ]
    if not (answer["initial_comparison"]["exact"] and repeat["exact"]):
        qualified = []
    require(qualified == answer["qualified_witness_indices"], "Independent full qualification indices")
    return dict(
        passed=True,
        verified_responses=len(metrics),
        strict_disagreements=[],
        repeat_exact=repeat["exact"],
        qualified_witness_indices=qualified,
        control_qualified=False,
        pose_objective_policy=POLICY,
        historical_scope="Saved-route history exact; independent arithmetic uses registered tolerances",
    )
