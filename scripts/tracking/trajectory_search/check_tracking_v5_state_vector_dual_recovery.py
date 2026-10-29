"""Audit pose-only dual recovery and replay every charged full response."""

import argparse
import json
import os

import numpy as np

from s2a.infrastructure.legacy_script_imports import CurrentScriptImports
from s2a.infrastructure.native_environment import restore_cv2_environment, strict_native_environment
from s2a.infrastructure.publication import publish_result
from s2a.research.trajectory_search.accepted_point_sqp_evidence import PREFIX_CHECK_REF
from s2a.research.trajectory_search.inputs import model_registration, validate_historical_registration
from s2a.research.trajectory_search.restored_anchor_sqp_evidence import (
    CPU_ENVIRONMENT,
    REPO,
    read,
    reference,
    remember,
    require,
)
from scripts.tracking.trajectory_search.check_tracking_v5_rejected_response_search import verify_fresh_feedback
from scripts.tracking.trajectory_search.check_tracking_v5_state_vector_gn_search import verify_fresh_decisions
from scripts.tracking.trajectory_search.state_vector_dual_recovery_audit import (
    audit_search,
    reconstruct_error_vector_jacobian,
)
from scripts.verification.trajectory_search.check_tracking_v5_prefix_trust_control import compare_contract
from scripts.verification.trajectory_search.check_tracking_v5_saved_direction_backtracking import (
    NUMERIC_TOLERANCE,
    TORQUE_TOLERANCE,
    case_engine,
    native_functions,
    native_math_proof,
)
from scripts.verification.trajectory_search.check_tracking_v5_tail_translation_reserve_v2 import semantic_compare
from scripts.verification.trajectory_search.check_tracking_v5_terminal_secant_probe import verify_response_trace
from scripts.verification.trajectory_search.native_replay_inputs import bind_native_ast_helper

ROOT = REPO / "outputs/tracking_v5_state_vector_dual_recovery_check_20261029"


def main(dry=False, *, diagnostic_config=None):
    from scripts.tracking.trajectory_search.probe_tracking_v5_state_vector_dual_recovery import (
        CONFIG,
        initial_error_vector_jacobian,
        initial_model,
        load_inputs,
    )
    from scripts.tracking.trajectory_search.probe_tracking_v5_state_vector_dual_recovery import ROOT as PRODUCER_ROOT
    from scripts.tracking.trajectory_search.probe_tracking_v5_terminal_only_tail_repair import load_environment_guard

    strict_native_environment()
    require(
        all(os.environ.get(key) == value for key, value in CPU_ENVIRONMENT.items()), "Single-thread CPU environment"
    )
    current_imports = CurrentScriptImports(root=REPO)
    current_imports.verify_present()
    bundle, config, context = load_inputs()
    refs = context.runtime_refs
    diagnostic = None
    output = ROOT
    if diagnostic_config is not None:
        from scripts.tracking.trajectory_search.state_vector_fresh_diagnostic_admission import OUTPUT, admit_replay

        diagnostic = admit_replay(context, diagnostic_config)
        output = OUTPUT
    Js, Jq = initial_model(bundle)
    Je = reconstruct_error_vector_jacobian(bundle)
    np.testing.assert_array_equal(Je, initial_error_vector_jacobian(bundle))
    source, source_reg = bundle["source"], bundle["source_registration"]
    guard = load_environment_guard(refs)
    migrated_registration, proof = model_registration(source_reg, context, bundle)
    math_proof = native_math_proof()
    remember(math_proof["source_ref"], refs)
    prefix_check = read(remember(PREFIX_CHECK_REF, refs))
    prefix_check_reg = read(remember(prefix_check["registration_ref"], refs))
    helper_identity = bind_native_ast_helper(context, prefix_check_reg["ast_helper_ref"], native_functions)
    context.finish()
    current_imports.finish()
    current_identity = current_imports.verify_present()
    if dry:
        print(
            json.dumps(dict(passed=True, dry=True, qp_solves=0, responses=0, queries=0, source_count=len(refs))),
            flush=True,
        )
        return
    require(not output.exists(), "Exclusive checker output; no overwrite or automatic retry")
    producer_ref = remember(reference(PRODUCER_ROOT / "verified.json"), refs)
    if diagnostic is not None:
        require(producer_ref == diagnostic["config"]["producer_ref"], "Exact original producer for diagnostic replay")
    producer = read(producer_ref)
    require(producer["complete"] is producer["passed"] is True, "Normally completed producer")
    registration = read(remember(producer["registration_ref"], refs))
    validate_historical_registration(
        context,
        registration,
        config=config,
        old_config_ref=context.historical_reference("configs/state_vector_dual_recovery_20261029.json"),
        old_plan_ref=context.historical_reference("plans/v0.0.1/state_vector_dual_recovery_20261029.md"),
    )
    require(
        registration["experiment"] == "state_vector_dual_recovery"
        and registration["config"] == config
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
        and registration["pose_damping"] == 1.0
        and registration["qp_objective_scaling_policy"] == "pose_whole_objective_divisor"
        and registration["objective_divisor_rule"] == "restore1_pose_max_1_gradient_inf_radius_hessian_inf"
        and registration["original_certificate_required"] is True
        and registration["dual_recovery_policy"] == "pose_slack_restricted_nnls_restore_full_active"
        and registration["pose_recovery_threshold_rule"] == "0.01_times_certificate_tolerance"
        and registration["restore_recovery_policy"] == "original_full_active"
        and registration["nnls_iteration_budget_rule"] == "3_times_original_full_active_count_or_one"
        and registration["single_fit_no_fallback"] is True,
        "Registered feedback policy",
    )
    for ref in registration["source_refs"] + registration["input_refs"] + producer["source_refs_unchanged"]:
        context.remember_historical(ref)
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
    qps = [stage["qp"] for stage in answer["stages"]]
    require(
        producer["nnls_fit_attempts"] == ledger["total_nnls_fit_attempts"] == sum(qp["nnls_fit_attempts"] for qp in qps)
        and producer["nnls_successful_fits"]
        == ledger["total_successful_nnls_fits"]
        == sum(qp["nnls_fit_attempts"] == 1 and qp["nnls_success"] for qp in qps)
        and 0 <= producer["nnls_successful_fits"] <= producer["nnls_fit_attempts"] <= len(qps),
        "Every fit including a failed attempt is charged once, with no hidden recovery",
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
    context.finish()
    registered = publish_result(
        output / "registration.json",
        dict(
            experiment="state_vector_dual_recovery",
            producer_ref=producer_ref,
            producer_registration_ref=producer["registration_ref"],
            config_ref=reference(CONFIG),
            input_refs=list(refs.values()),
            migration_input_identity=context.registration(),
            current_import_identity=current_identity,
            math_proof=math_proof,
            native_ast_helper_identity=helper_identity,
            numeric_tolerance=NUMERIC_TOLERANCE,
            torque_tolerance=TORQUE_TOLERANCE,
            expected_responses=count,
            automatic_retry=False,
            diagnostic_registration=diagnostic,
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
        nnls_fit_attempts=0,
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
    if diagnostic is not None:
        result["fresh_decision_diagnostics"] = []

    def compare(label, actual, saved, tolerance=NUMERIC_TOLERANCE):
        difference = semantic_compare(label, actual, saved, tolerance)
        result["max_abs_difference"][label] = max(result["max_abs_difference"].get(label, 0.0), difference)
        return difference

    rollouts, metrics = [], []
    try:
        result["environment_admission"] = restore_cv2_environment(
            guard["exact_probe_ref"], guard["runtime_source_refs"]
        )
        os.environ.pop("MUJOCO_GL", None)
        os.environ.pop("PYOPENGL_PLATFORM", None)
        result["asset_loading_proof"] = proof
        reconstruct, compare_rollout, classify, check_audit, current, compiled_goal = case_engine(
            "prefix", source, migrated_registration, prefix_check_reg, result, refs, compare, input_context=context
        )
        compare_contract(current, producer["current_quantities"], compare=compare, name="new_current")

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
            result["independent_search_check"] = verify_fresh_decisions(
                answer, rollouts, metrics, config, diagnostics=result.get("fresh_decision_diagnostics")
            )
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
        context.finish()
        current_imports.finish()
        strict_native_environment()
        result.update(
            complete=True,
            passed=True,
            source_refs_unchanged=list(refs.values()),
            migration_input_identity=context.registration(),
        )
    except BaseException as error:
        result["error"] = repr(error)
        raise
    finally:
        result["query_count"] = result["mj_forward_queries"]
        source_error = None
        try:
            context.finish()
            current_imports.finish()
            strict_native_environment()
            result["source_validation_after"] = True
            result["source_refs_unchanged"] = list(refs.values())
            result["migration_input_identity"] = context.registration()
        except BaseException as error:
            source_error = error
            result.update(complete=False, passed=False, source_validation_after=False, source_after_error=repr(error))
        if diagnostic is not None:
            result["fresh_metrics_ref"] = publish_result(
                output / "fresh_metrics.json",
                dict(
                    scope="Existing independent classifications; no additional model queries",
                    producer_ref=producer_ref,
                    response_count=len(metrics),
                    metrics=[{key: value for key, value in metric.items() if key != "groups"} for metric in metrics],
                ),
            )
        publish_result(output / "verified.json", result)
        if source_error is not None:
            raise source_error
    print(json.dumps(dict(passed=True, verified_responses=count, witness_found=result["witness_found"])), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry", action="store_true")
    main(parser.parse_args().dry)
