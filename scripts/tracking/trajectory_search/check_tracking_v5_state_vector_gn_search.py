"""Audit vector Gauss-Newton and independently replay every charged response."""

import argparse
import json
import os

import numpy as np

from s2a.infrastructure.native_environment import restore_cv2_environment, strict_native_environment
from s2a.infrastructure.publication import publish_result
from s2a.research.trajectory_search.accepted_point_sqp_evidence import PREFIX_CHECK_REF
from s2a.research.trajectory_search.restored_anchor_sqp_evidence import (
    CPU_ENVIRONMENT,
    REPO,
    finish,
    model_registration,
    read,
    reference,
    remember,
    require,
)
from scripts.tracking.trajectory_search.check_tracking_v5_rejected_response_search import verify_fresh_feedback
from scripts.tracking.trajectory_search.state_vector_gn_search_audit import (
    audit_search,
    reconstruct_error_vector_jacobian,
)
from scripts.verification.trajectory_search.check_tracking_v5_accepted_point_sqp import case_engine
from scripts.verification.trajectory_search.check_tracking_v5_prefix_trust_control import compare_contract
from scripts.verification.trajectory_search.check_tracking_v5_saved_direction_backtracking import (
    NUMERIC_TOLERANCE,
    TORQUE_TOLERANCE,
    native_math_proof,
)
from scripts.verification.trajectory_search.check_tracking_v5_tail_translation_reserve_v2 import semantic_compare
from scripts.verification.trajectory_search.check_tracking_v5_terminal_secant_probe import verify_response_trace
from scripts.verification.trajectory_search.tail_translation_reserve_audit import _comparison_report

ROOT = REPO / "outputs/tracking_v5_state_vector_gn_search_check_20261029"


def fresh_metrics(metric, config):
    """Derive the merit only from the independently recomputed response."""
    safety = np.asarray(metric["safety_slack"], dtype=float)
    squared = np.asarray(metric["normalized_squared_errors"], dtype=float)
    require(safety.shape == (4369,) and squared.shape == (12, 2), "All original independent constraint values")
    require(np.isfinite(safety).all() and np.isfinite(squared).all(), "Finite independent response")
    weights = np.ones(4369)
    weights[-1] = config["terminal_time_scale"]
    reserve = float(np.min(weights * (safety - config["safety_buffer"]) / config["reserve_target"]))
    return dict(
        reserve=reserve,
        violation=max(0.0, 1.0 - reserve),
        pose_excess=max(0.0, float(squared.max()) - 1.0),
        state_loss=float(0.5 * squared.sum()),
    )


def verify_fresh_decisions(answer, rollouts, metrics, config, *, diagnostics=None):
    """Strictly reclassify actual loss, safety, priority, radius and full caps."""
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
            minimum = min(minimum, config["pose_reduction_fraction"] * base["pose_excess"])
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
                merit_field="violation" if mode == "restore" else "state_loss",
                fresh_base_merit=float(base["violation" if mode == "restore" else "state_loss"]),
                saved_base_merit=float(saved_base["violation" if mode == "restore" else "state_loss"]),
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
        field = "violation" if mode == "restore" else "state_loss"
        minimum = config["minimum_actual_reduction"]
        if mode == "pose":
            minimum = min(minimum, config["pose_reduction_fraction"] * base["pose_excess"])
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
        pose_objective_policy="normalized_state_vector_gauss_newton",
        historical_scope="Saved-route history exact; independent arithmetic uses registered tolerances",
    )


def main(dry=False):
    from scripts.tracking.trajectory_search.probe_tracking_v5_state_vector_gn_search import (
        CONFIG,
        PLAN,
        initial_error_vector_jacobian,
        initial_model,
        load_inputs,
    )
    from scripts.tracking.trajectory_search.probe_tracking_v5_state_vector_gn_search import ROOT as PRODUCER_ROOT
    from scripts.tracking.trajectory_search.probe_tracking_v5_terminal_only_tail_repair import load_environment_guard

    strict_native_environment()
    require(
        all(os.environ.get(key) == value for key, value in CPU_ENVIRONMENT.items()), "Single-thread CPU environment"
    )
    bundle, config, refs = load_inputs()
    Js, Jq = initial_model(bundle)
    Je = reconstruct_error_vector_jacobian(bundle)
    np.testing.assert_array_equal(Je, initial_error_vector_jacobian(bundle))
    source, source_reg = bundle["source"], bundle["source_registration"]
    guard = load_environment_guard(refs)
    _, proof = model_registration(source_reg, refs)
    math_proof = native_math_proof()
    remember(math_proof["source_ref"], refs)
    prefix_check = read(remember(PREFIX_CHECK_REF, refs))
    prefix_check_reg = read(remember(prefix_check["registration_ref"], refs))
    finish(refs)
    if dry:
        print(
            json.dumps(dict(passed=True, dry=True, qp_solves=0, responses=0, queries=0, source_count=len(refs))),
            flush=True,
        )
        return
    require(not ROOT.exists(), "Exclusive checker output; no overwrite or automatic retry")
    producer_ref = remember(reference(PRODUCER_ROOT / "verified.json"), refs)
    producer = read(producer_ref)
    require(producer["complete"] is producer["passed"] is True, "Normally completed producer")
    registration = read(remember(producer["registration_ref"], refs))
    require(
        registration["experiment"] == "state_vector_gn_search"
        and registration["config"] == config
        and registration["config_ref"] == reference(CONFIG)
        and registration["plan_ref"] == reference(PLAN)
        and registration["historical_response_index"] == 3557
        and registration["seed"] == 42
        and registration["response_budget"] == config["max_responses"]
        and registration["qp_budget"] == config["max_stages"]
        and registration["qp_iteration_budget"] == 100 * config["max_stages"] == 6000
        and registration["automatic_retry"] is False,
        "Exact preregistered search",
    )
    require(
        registration["feedback_policy"] == "safety_qualified_pose_ceiling_rejection_minimum_alpha"
        and registration["pose_ceiling_policy"] == "fixed_initial_global_maximum"
        and registration["pose_objective_policy"] == "normalized_state_vector_gauss_newton"
        and type(registration["pose_damping"]) is float
        and registration["pose_damping"] == 1.0,
        "Registered feedback policy",
    )
    for ref in registration["source_refs"] + registration["input_refs"] + producer["source_refs_unchanged"]:
        remember(ref, refs)
    answer = producer["answer"]
    ledger = audit_search(
        answer,
        Js,
        Jq,
        Je,
        bundle["base"],
        np.asarray(source["previous_command"])[:6],
        bundle["control"]["input_min"],
        bundle["control"]["input_max"],
        config,
    )
    count = len(answer["evaluated_candidates"])
    require(
        count == producer["prediction_calls"] == producer["response_evaluations"] <= config["max_responses"]
        and producer["numerical_inner_steps"] == 300 * count
        and producer["query_count"] == 1 + 312 * count
        and producer["partial_response_queries"] == 0,
        "Every producer call including repeat is fully billed",
    )
    require(
        producer["unique_command_count"] == answer["unique_command_count"]
        and producer["qp_solves"] == len(answer["stages"])
        and producer["model_construction_attempts"] == producer["model_constructions_completed"] == 1
        and producer["automatic_retry"] is False,
        "One admitted numerical model and no hidden retry",
    )
    for key in ("physics_steps", "world_model_calls", "fresh_cases", "actual_consumed_reference_count"):
        require(type(producer[key]) is int and producer[key] == 0, "No unregistered activity: " + key)
    require(producer["control_qualified"] is False, "No physical control qualification")
    compare_contract(proof, producer["asset_loading_proof"], name="producer assets")
    finish(refs)
    registered = publish_result(
        ROOT / "registration.json",
        dict(
            experiment="state_vector_gn_search",
            producer_ref=producer_ref,
            producer_registration_ref=producer["registration_ref"],
            config_ref=reference(CONFIG),
            input_refs=list(refs.values()),
            math_proof=math_proof,
            numeric_tolerance=NUMERIC_TOLERANCE,
            torque_tolerance=TORQUE_TOLERANCE,
            expected_responses=count,
            automatic_retry=False,
        ),
    )
    result = dict(
        complete=False,
        passed=False,
        registration_ref=registered,
        ledger_check=ledger,
        response_evaluations=0,
        verified_responses=0,
        unique_command_count=0,
        numerical_inner_steps=0,
        mj_forward_queries=0,
        query_count=0,
        repeated_responses_checked=0,
        qp_solves=0,
        max_abs_difference={},
        max_jit_vs_python_goal_difference=0.0,
        jit_python_goal_differences=0,
        boundary_classification_disagreements=[],
        physics_steps=0,
        world_model_calls=0,
        fresh_cases=0,
        actual_consumed_reference_count=0,
        control_qualified=False,
    )

    def compare(label, actual, saved, tolerance=NUMERIC_TOLERANCE):
        difference = semantic_compare(label, actual, saved, tolerance)
        result["max_abs_difference"][label] = max(result["max_abs_difference"].get(label, 0.0), difference)
        return difference

    try:
        result["environment_admission"] = restore_cv2_environment(
            guard["exact_probe_ref"], guard["runtime_source_refs"]
        )
        os.environ.pop("MUJOCO_GL", None)
        os.environ.pop("PYOPENGL_PLATFORM", None)
        reconstruct, compare_rollout, classify, check_audit, current, compiled_goal = case_engine(
            "prefix", source, source_reg, prefix_check_reg, result, refs, compare
        )
        compare_contract(current, producer["current_quantities"], compare=compare, name="new_current")
        rollouts, metrics = [], []

        def retain_replay(commands):
            rollout, differences = reconstruct(commands)
            rollouts.append(rollout)
            return rollout, differences

        def retain_classification(commands, rollout):
            metric = classify(commands, rollout)
            metrics.append(metric)
            return metric

        verify_response_trace(
            answer, retain_replay, compare_rollout, retain_classification, check_audit, result, compare=compare
        )
        try:
            result["independent_search_check"] = verify_fresh_decisions(answer, rollouts, metrics, config)
            result["independent_feedback_check"] = verify_fresh_feedback(answer, metrics, config)
        except ValueError as error:
            result["boundary_classification_disagreements"].append(
                dict(scope="fresh search decision", error=str(error))
            )
            raise
        require(
            result["verified_responses"] == count and bool(compiled_goal.nopython_signatures),
            "All native responses checked",
        )
        result["native_quaternion_jit_signatures"] = [str(value) for value in compiled_goal.nopython_signatures]
        finish(refs)
        strict_native_environment()
        result.update(complete=True, passed=True, source_refs_unchanged=list(refs.values()))
    except BaseException as error:
        result["error"] = repr(error)
        raise
    finally:
        result["query_count"] = result["mj_forward_queries"]
        publish_result(ROOT / "verified.json", result)
    print(json.dumps(dict(passed=True, verified_responses=count, witness_found=result["witness_found"])), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry", action="store_true")
    main(parser.parse_args().dry)
