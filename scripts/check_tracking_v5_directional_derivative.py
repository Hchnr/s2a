"""Independently replay every fixed-grid call, including the final anchor repeat."""

import argparse
import json
import os

import numpy as np
from check_tracking_v5_accepted_point_sqp import case_engine
from check_tracking_v5_prefix_trust_control import (
    check_full_candidate,
    command_key,
    compare_contract,
    numerical_compare,
    validate_trace_entry,
)
from check_tracking_v5_saved_direction_backtracking import NUMERIC_TOLERANCE, TORQUE_TOLERANCE, native_math_proof
from directional_derivative_audit import (
    _audit_repeat,
    audit_directional_derivative_ledger,
    audit_saved_stencil_comparison,
    exact_array,
)
from execute_tracking_v5_directional_derivative import entry_command, validate_receipt

from s2a.directional_derivative_evidence import (
    CPU_ENVIRONMENT,
    PLAN,
    PREFIX_CHECK_REF,
    REPO,
    SOURCES,
    SQP_PRODUCER_REF,
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
from s2a.native_environment import restore_cv2_environment, strict_native_environment
from s2a.preview_result_publication import publish_result
from s2a.terminal_development_execution import CONTROL_REF


def verify_response_trace(answer, reconstruct, compare_rollout, classify, check_audit, result, *, compare):
    """No deduplication: every billed call gets its own independent full replay."""
    entries = answer["evaluated_candidates"]
    for index, saved in enumerate(entries):
        commands = np.asarray(saved["commands"], dtype=float)
        command_key(commands, 12)
        rollout, _ = reconstruct(commands)
        result["response_evaluations"] += 1
        result["query_count"] = result["mj_forward_queries"]
        metrics = classify(commands, rollout)
        compact = dict(
            commands=saved["commands"],
            safety_min=float(np.min(saved["safety_slack"])),
            joint_feasible=saved["audit"]["feasible"],
            tcp_errors=saved["tcp_errors"],
            prefix_length=saved["prefix_length"],
            feasible=saved["feasible"],
        )
        try:
            validate_trace_entry(compact, metrics, compare=compare)
            check_full_candidate(saved, commands, rollout, metrics, compare_rollout, check_audit, compare=compare)
        except ValueError:
            result["boundary_classification_disagreements"].append(
                dict(
                    response_index=index,
                    producer_prefix=saved["prefix_length"],
                    independent_prefix=metrics["prefix_length"],
                    producer_joint=saved["audit"]["feasible"],
                    independent_joint=metrics["joint_feasible"],
                    producer_witness=saved["feasible"],
                    independent_witness=metrics["feasible"],
                    evidence="Full numerical or strict classification check rejected; no control inference",
                )
            )
            raise
        result["verified_responses"] += 1
        require(
            result["numerical_inner_steps"] == 300 * (index + 1)
            and result["mj_forward_queries"] == 1 + 312 * (index + 1),
            "Every actual call including repeat completes the independent H12",
        )
        if (index + 1) % 10 == 0 or index + 1 == len(entries):
            print(json.dumps(dict(checked=index + 1, total=len(entries))), flush=True)
    # Different replay arithmetic was first checked against every full saved record.
    # Difference algebra now uses the saved original-precision values, not replay derivatives.
    result["ledger_check"] = audit_directional_derivative_ledger(answer)
    result["repeated_responses_checked"] = 1
    result["unique_command_count"] = answer["unique_command_count"]
    result["witness_indices"] = [index for index, entry in enumerate(entries) if entry["feasible"]]
    result["witness_found"] = bool(result["witness_indices"])


def verify_historical_anchor(sqp, answer, saved):
    old = dict(sqp["evaluated_candidates"][1819])
    anchors = [row for row in sqp["accepted_anchors"] if row["response_index"] == 1819]
    require(len(anchors) == 1, "Exact saved accepted anchor")
    old["audit"] = anchors[0]["audit"]
    new = answer["evaluated_candidates"][0]
    projection = {
        key: new[key] for key in ("commands", "tcp_errors", "safety_slack", "prefix_length", "feasible", "audit")
    }
    projection.update(safety_min=float(np.min(new["safety_slack"])), joint_feasible=new["audit"]["feasible"])
    require(
        saved["scope"] == "Complete archived trace and accepted-anchor audit versus new response0; no old full rollout"
        and saved["historical_response_index"] == 1819
        and saved["new_response_index"] == 0,
        "Historical comparison is restricted to the actual archived trace and audit",
    )
    comparison = dict(saved["comparison"], initial_response_index=0, repeated_response_index=1)
    _audit_repeat(
        dict(repeatability=comparison, derivative_reliability_limited=not comparison["exact"]),
        [old, projection],
        numerical_compare,
    )
    return dict(
        passed=True,
        historical_response_index=1819,
        compared_fields=sorted(old),
        exact=comparison["exact"],
        max_abs_difference=comparison["max_abs_difference"],
    )


def main(dry=False):
    strict_native_environment()
    require(all(os.environ.get(key) == value for key, value in CPU_ENVIRONMENT.items()), "Fixed single-thread CPU")
    original, source_reg, sqp, closure, refs = load_prerequisites()
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
        registration["experiment"] == "directional_derivative_consistency"
        and registration["plan_ref"] == reference(PLAN)
        and registration["seed"] == 42
        and registration["sqp_producer_ref"] == SQP_PRODUCER_REF
        and registration["sqp_closure"] == closure
        and registration["endpoint"] == 368
        and registration["horizon"] == 12
        and registration["anchor_response_index"] == 1819
        and registration["source_stages"] == [24, 25, 26]
        and registration["response_budget"] == registration["planned_request_count"] == 80
        and registration["unique_command_budget"] == 79
        and registration["automatic_retry"] is False
        and registration["actual_consumed_reference_count"] == 0
        and registration["original_prefix_registration_ref"] == original["registration_ref"],
        "Exact fixed-grid registration and closed original provenance",
    )
    for ref in registration["source_refs"] + registration["input_refs"] + producer["source_refs_unchanged"]:
        remember(ref, refs)
    compare_contract(expected_assets, registration["asset_loading_proof"], name="registered_assets")
    compare_contract(expected_assets, producer["asset_loading_proof"], name="producer_assets")
    answer = producer["answer"]
    exact_array("registered_steps", answer["step_sizes"], registration["step_sizes"])
    exact_array(
        "completed367_previous", np.asarray(original["previous_command"], dtype=float)[:6], answer["previous_command"]
    )
    control = read(remember(source_reg["control_ref"], refs))["contract"]
    for field in ("input_min", "input_max"):
        exact_array("original_" + field, control[field], answer[field])
    n = producer["response_evaluations"]
    new_registration = publish_result(
        output / "registration.json",
        dict(
            experiment="directional_derivative_consistency",
            producer_ref=producer_ref,
            producer_owner_ref=owner_ref,
            producer_registration_ref=producer["registration_ref"],
            plan_ref=reference(PLAN),
            seed=42,
            sqp_producer_ref=SQP_PRODUCER_REF,
            sqp_closure=closure,
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
            evidence_scope="Every actual complete rollout and audit including final repeat; independent grid and saved-FD algebra",
            automatic_retry=False,
        ),
    )
    result = dict(
        complete=False,
        passed=False,
        experiment="directional_derivative_consistency",
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
        verify_response_trace(answer, reconstruct, compare_rollout, classify, check_audit, result, compare=compare)
        result["saved_stencil_check"] = audit_saved_stencil_comparison(
            sqp["answer"], answer, producer["saved_stencil_comparison"]
        )
        result["historical_anchor_check"] = verify_historical_anchor(
            sqp["answer"], answer, producer["historical_anchor_comparison"]
        )
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
            evidence="Independent complete local numerical and derivative diagnosis; no control or task qualification",
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
