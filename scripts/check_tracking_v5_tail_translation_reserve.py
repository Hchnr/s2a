"""Independently replay14 tail translations and verify their strict restoration gates."""

import argparse
import json
import os

import numpy as np
from check_tracking_v5_accepted_point_sqp import case_engine
from check_tracking_v5_prefix_trust_control import compare_contract, numerical_compare
from check_tracking_v5_saved_direction_backtracking import NUMERIC_TOLERANCE, TORQUE_TOLERANCE, native_math_proof
from check_tracking_v5_terminal_secant_probe import verify_response_trace as verify_response_trace
from directional_derivative_audit import exact_array
from execute_tracking_v5_tail_translation_reserve import entry_command, validate_receipt
from tail_translation_reserve_audit import (
    _comparison_report,
    audit_tail_translation_grid,
    audit_tail_translation_ledger,
    audit_tail_translation_prior_residuals,
    compare_tail_causal_rollouts,
)

from s2a.native_environment import restore_cv2_environment, strict_native_environment
from s2a.preview_result_publication import publish_result
from s2a.tail_translation_reserve_evidence import (
    CPU_ENVIRONMENT,
    PLAN,
    PREFIX_CHECK_REF,
    PRIOR_ANALYSIS_REF,
    PRIOR_CLOSURE_REF,
    PRIOR_PRODUCER_REF,
    REPO,
    SOURCES,
    case_root,
    closed_owner,
    finish,
    load_prerequisites,
    model_registration,
    read,
    reference,
    remember,
    require,
)
from s2a.terminal_development_execution import CONTROL_REF


def verify_replayed_restoration(answer, rollouts, metrics):
    """Check strict gates within fresh arithmetic, separately from saved-value algebra."""
    require(len(rollouts) == len(metrics) == len(answer["causal_checks"]) == 14, "All fourteen fresh replays")
    require(len(answer["candidate_checks"]) == 12, "All twelve restoration candidates")
    causal = []
    for index, rollout in enumerate(rollouts):
        current = compare_tail_causal_rollouts(rollouts[0], rollout, index)
        for field in ("prefix_exact", "rotation_goals_exact", "passed"):
            require(current[field] is answer["causal_checks"][index][field], "Fresh causal strict flag: " + field)
        causal.append(current)
    repeat = _comparison_report(
        dict(rollout=rollouts[0], metrics=metrics[0]),
        dict(rollout=rollouts[-1], metrics=metrics[-1]),
        13,
    )
    require(repeat["exact"] is answer["repeatability"]["exact"], "Fresh complete anchor repeat strict flag")
    history = answer["historical_anchor_comparison"]["comparison"]["exact"]
    terminal = [float(np.asarray(row["safety_slack"])[4368]) for row in metrics]
    candidates = []
    for index in range(1, 13):
        metric = metrics[index]
        safe = np.asarray(metric["safety_slack"], dtype=float)
        squared = np.asarray(metric["normalized_squared_errors"], dtype=float)
        require(safe.shape == (4369,) and np.isfinite(safe).all(), "Full finite independent safety vector")
        require(squared.shape == (12, 2) and np.isfinite(squared).all(), "Full finite independent pose errors")
        row = dict(
            response_index=index,
            joint_feasible=metric["joint_feasible"],
            safety_buffer_satisfied=bool(np.all(safe >= 1e-8)),
            prefix9_preserved=bool(np.all(squared[:9] <= 1.0)),
            causal_exact=causal[index]["passed"],
            historical_exact=history,
            repeat_exact=repeat["exact"],
        )
        row["restoration_eligible"] = all(value for key, value in row.items() if key != "response_index")
        for key, value in row.items():
            if key != "response_index":
                require(
                    type(value) is bool and value is answer["candidate_checks"][index - 1][key],
                    "Independent strict restoration flag: " + key,
                )
        row.update(terminal=terminal[index], terminal_reserve=terminal[index] - 1e-8)
        candidates.append(row)
    improving = [row for row in candidates if row["restoration_eligible"] and row["terminal"] > terminal[0]]
    improvement = bool(improving)
    require(improvement is answer["improvement_found"], "Independent strict improvement classification")
    selected = answer["selected_response_index"]
    if selected is None:
        require(not improvement, "No omitted independently improving candidate")
        sufficient = False
    else:
        require(type(selected) is int and 1 <= selected <= 12, "One fixed selected response")
        require(
            candidates[selected - 1] in improving, "Selected response independently remains an eligible improvement"
        )
        sufficient = bool(terminal[selected] - 1e-8 >= 10 * answer["prior_max_abs_residual"])
    require(sufficient is answer["sufficient_reserve_restored"], "Independent strict empirical reserve threshold")
    return dict(
        passed=True,
        response_evaluations=14,
        causal_checks=causal,
        repeatability=repeat,
        candidate_checks=candidates,
        selected_response_index=selected,
        improvement_found=improvement,
        sufficient_reserve_restored=sufficient,
        strict_disagreements=[],
        ranking_scope="Exact ordering is independently rebuilt from verified saved-precision values by ledger_check",
        control_qualified=False,
    )


def validate_prior_binding(registration, answer, prior_check, analysis):
    """Bind the fixed empirical threshold to every prior original-model residual."""
    residuals = prior_check["original_residuals"]
    error = prior_check["prior_max_abs_residual"]
    points = analysis["analysis"]["points"]
    require(len(points) == len(residuals) == 15, "All fifteen closed prior validation points")
    require(
        all(
            point["request_index"] == i + 1 and point["old_residual"] == residuals[i] for i, point in enumerate(points)
        ),
        "Exact all-point closed prior analysis binding",
    )
    expected = dict(
        prior_producer_ref=PRIOR_PRODUCER_REF,
        prior_analysis_ref=PRIOR_ANALYSIS_REF,
        request_indices=list(range(1, 16)),
        original_residuals=residuals,
        validation_point_count=15,
        terminal_row_index=4368,
        historical_response_index=0,
        original_sqp_anchor_response_index=1819,
        prior_max_abs_residual=error,
        reserve_multiplier=10.0,
        safety_buffer=1e-8,
        reserve_target=10 * error,
        terminal_slack_target=1e-8 + 10 * error,
    )
    require(registration["prior_error_binding"] == expected, "Exact independently reconstructed E17 source binding")
    require(
        registration["prior_max_abs_residual"] == answer["prior_max_abs_residual"] == error,
        "Unmodified prior E17 before fresh model construction",
    )
    require(
        answer["reserve_target"] == 10 * error and answer["terminal_threshold"] == 1e-8 + 10 * error,
        "Fixed unrounded empirical reserve thresholds",
    )
    return expected


def main(dry=False):
    strict_native_environment()
    require(all(os.environ.get(key) == value for key, value in CPU_ENVIRONMENT.items()), "Fixed single-thread CPU")
    original, source_reg, prior, closure, refs = load_prerequisites()
    proof = native_math_proof()
    remember(proof["source_ref"], refs)
    prefix_check = read(remember(PREFIX_CHECK_REF, refs))
    prefix_check_reg = read(remember(prefix_check["registration_ref"], refs))
    _, expected_assets = model_registration(source_reg, refs)
    finish(refs)
    if dry:
        print(
            json.dumps(
                dict(
                    dry_run=True,
                    passed=True,
                    source_count=len(refs),
                    new_responses=0,
                    new_queries=0,
                    numerical_math_proof=proof,
                )
            ),
            flush=True,
        )
        return
    output = case_root(check=True)
    require(not output.exists(), "Exclusive new independent replay; no overwrite or retry")
    owner_ref = remember(reference(case_root(owner=True) / "result.json"), refs)
    owner, producer = closed_owner(owner_ref, refs)
    producer_ref = remember(reference(case_root() / "verified.json"), refs)
    require(
        owner["result_ref"] == producer_ref
        and owner["stage"] == "diagnose"
        and read(remember(owner["owned_command"]["launch"], refs))["command"] == entry_command("diagnose"),
        "Exact normally closed fixed-grid producer command",
    )
    validate_receipt(producer, "diagnose")
    registration = read(remember(producer["registration_ref"], refs))
    require(
        registration["experiment"] == "tail_translation_reserve"
        and registration["plan_ref"] == reference(PLAN)
        and registration["seed"] == 42
        and registration["prior_producer_ref"] == PRIOR_PRODUCER_REF
        and registration["prior_closure_ref"] == PRIOR_CLOSURE_REF
        and registration["prior_analysis_ref"] == PRIOR_ANALYSIS_REF
        and registration["prior_closure"] == closure
        and registration["endpoint"] == 368
        and registration["horizon"] == 12
        and registration["anchor_response_index"] == 1819
        and registration["response_budget"] == registration["planned_request_count"] == 14
        and registration["unique_command_budget"] == 13
        and registration["automatic_retry"] is False
        and registration["actual_consumed_reference_count"] == 0
        and registration["original_prefix_registration_ref"] == original["registration_ref"],
        "Exact tail-only registration and closed original provenance",
    )
    for ref in registration["source_refs"] + registration["input_refs"] + producer["source_refs_unchanged"]:
        remember(ref, refs)
    compare_contract(expected_assets, registration["asset_loading_proof"], name="registered_assets")
    compare_contract(expected_assets, producer["asset_loading_proof"], name="producer_assets")
    answer = producer["answer"]
    require(registration["signed_steps"] == [0.001, -0.001, 0.0005, -0.0005], "Fixed signed grid steps")
    exact_array("registered_steps", answer["grid"]["signed_steps"], registration["signed_steps"])
    require(answer["grid"] == registration["grid"], "Exact original complete grid registration")
    exact_array("old_anchor", answer["grid"]["anchor_commands"], prior["answer"]["geometry"]["anchor_commands"])
    exact_array(
        "completed367_previous",
        np.asarray(original["previous_command"], dtype=float)[:6],
        answer["grid"]["previous_command"],
    )
    control = read(remember(source_reg["control_ref"], refs))["contract"]
    for field in ("input_min", "input_max"):
        exact_array("original_" + field, control[field], answer["grid"][field])
    grid_check = audit_tail_translation_grid(answer["grid"])
    prior_residual_check = audit_tail_translation_prior_residuals(prior["answer"])
    prior_binding = validate_prior_binding(registration, answer, prior_residual_check, read(PRIOR_ANALYSIS_REF))
    n = producer["response_evaluations"]
    new_registration = publish_result(
        output / "registration.json",
        dict(
            experiment="tail_translation_reserve",
            producer_ref=producer_ref,
            producer_owner_ref=owner_ref,
            producer_registration_ref=producer["registration_ref"],
            plan_ref=reference(PLAN),
            seed=42,
            prior_producer_ref=PRIOR_PRODUCER_REF,
            prior_closure_ref=PRIOR_CLOSURE_REF,
            prior_analysis_ref=PRIOR_ANALYSIS_REF,
            prior_error_binding=prior_binding,
            prior_closure=closure,
            source_refs=[reference(REPO / path) for path in SOURCES],
            input_refs=list(refs.values()),
            numerical_math_proof=proof,
            numeric_tolerance=NUMERIC_TOLERANCE,
            torque_tolerance=TORQUE_TOLERANCE,
            expected_responses=n,
            expected_unique_commands=n - 1,
            expected_inner_steps=300 * n,
            expected_queries=1 + 312 * n,
            expected_repeated_responses=1,
            evidence_scope="All14 full calls including repeat; independent grid, causal prefix, strict restoration and prior E17 binding",
            automatic_retry=False,
        ),
    )
    result = dict(
        complete=False,
        passed=False,
        experiment="tail_translation_reserve",
        registration_ref=new_registration,
        response_evaluations=0,
        verified_responses=0,
        unique_command_count=0,
        repeated_responses_checked=0,
        numerical_inner_steps=0,
        mj_forward_queries=0,
        query_count=0,
        total_iterations=0,
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
        difference = numerical_compare(label, actual, saved, tolerance)
        result["max_abs_difference"][label] = max(result["max_abs_difference"].get(label, 0.0), difference)
        return difference

    try:
        guard = read(remember(CONTROL_REF, refs))["cv2_guard_registration"]
        result["environment_admission"] = restore_cv2_environment(
            guard["exact_probe_ref"], guard["runtime_source_refs"]
        )
        for ref in [guard["exact_probe_ref"], *guard["runtime_source_refs"]]:
            remember(ref, refs)
        os.environ.pop("MUJOCO_GL", None)
        os.environ.pop("PYOPENGL_PLATFORM", None)
        reconstruct, compare_rollout, classify, check_audit, current, compiled_goal = case_engine(
            "prefix", original, source_reg, prefix_check_reg, result, refs, compare
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
        result["grid_check"] = grid_check
        result["prior_residual_check"] = prior_residual_check
        result["ledger_check"] = audit_tail_translation_ledger(
            answer,
            historical_anchor=prior["answer"]["evaluated_candidates"][0],
            prior_answer=prior["answer"],
            compare=compare,
        )
        try:
            result["independent_restoration_check"] = verify_replayed_restoration(answer, rollouts, metrics)
        except ValueError as error:
            result["boundary_classification_disagreements"].append(
                dict(scope="fresh restoration/causality/repeat", error=str(error))
            )
            raise
        require(bool(compiled_goal.nopython_signatures), "Actual native quaternion compilation")
        result["native_quaternion_jit_signatures"] = [str(value) for value in compiled_goal.nopython_signatures]
        require(
            result["verified_responses"] == n and result["unique_command_count"] == producer["unique_command_count"],
            "No omitted response or repeat",
        )
        finish(refs)
        strict_native_environment()
        result.update(
            complete=True,
            passed=True,
            source_refs_unchanged=list(refs.values()),
            evidence="Independent complete local response and restoration audit; no control or task qualification",
        )
    except BaseException as error:
        result["error"] = repr(error)
        raise
    finally:
        result["query_count"] = result["mj_forward_queries"]
        publish_result(output / "verified.json", result)
    print(json.dumps(dict(complete=True, passed=True, verified_responses=n)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry", action="store_true")
    main(parser.parse_args().dry)
