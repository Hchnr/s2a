"""Independently replay all17 held-out secant calls and reconstruct their algebra."""

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
from directional_derivative_audit import exact_array
from execute_tracking_v5_terminal_secant_probe import entry_command, validate_receipt
from terminal_secant_audit import (
    audit_terminal_secant_fit,
    audit_terminal_secant_geometry,
    audit_terminal_secant_ledger,
)

from s2a.native_environment import restore_cv2_environment, strict_native_environment
from s2a.preview_result_publication import publish_result
from s2a.terminal_development_execution import CONTROL_REF
from s2a.terminal_secant_evidence import (
    CPU_ENVIRONMENT,
    DIAGNOSTIC_CLOSURE_REF,
    DIAGNOSTIC_REF,
    PLAN,
    PREFIX_CHECK_REF,
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
    result["repeated_responses_checked"] = 1
    result["unique_command_count"] = answer["unique_command_count"]
    result["witness_indices"] = [index for index, entry in enumerate(entries) if entry["feasible"]]
    result["witness_found"] = bool(result["witness_indices"])


def main(dry=False):
    strict_native_environment()
    require(all(os.environ.get(key) == value for key, value in CPU_ENVIRONMENT.items()), "Fixed single-thread CPU")
    original, source_reg, diagnostic, closure, refs = load_prerequisites()
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
        registration["experiment"] == "terminal_secant_probe"
        and registration["plan_ref"] == reference(PLAN)
        and registration["seed"] == 42
        and registration["diagnostic_ref"] == DIAGNOSTIC_REF
        and registration["diagnostic_closure_ref"] == DIAGNOSTIC_CLOSURE_REF
        and registration["prior_closure"] == closure
        and registration["endpoint"] == 368
        and registration["horizon"] == 12
        and registration["anchor_response_index"] == 1819
        and registration["response_budget"] == registration["planned_request_count"] == 17
        and registration["unique_command_budget"] == 16
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
    exact_array("registered_steps", answer["geometry"]["step_sizes"], registration["step_sizes"])
    require(registration["step_sizes"] == [0.001, 0.00075, 0.0005, 0.00025, 0.0001], "All five fixed validation scales")
    compare_contract(answer["geometry"], registration["geometry"], name="registered_geometry")
    compare_contract(answer["fit"], registration["fit"], name="registered_fit")
    exact_array("old_anchor", answer["geometry"]["anchor_commands"], diagnostic["answer"]["anchor_commands"])
    exact_array(
        "old_source_directions", answer["geometry"]["source_unit_directions"], diagnostic["answer"]["unit_directions"]
    )
    exact_array(
        "old_terminal_row",
        answer["fit"]["original_terminal_row"],
        diagnostic["saved_stencil_comparison"]["jacobian_safety_slack"][4368],
    )
    training = []
    for step in (0.001, 0.0005):
        candidates = [
            r
            for r in diagnostic["answer"]["request_ledger"]
            if r["role"] == "grid" and r["source_stage"] == 24 and r["step"] == step and r["sign"] == 1
        ]
        require(len(candidates) == 1, "Two exact source training slots")
        training.append(candidates[0])
    require(
        registration["fit_source"]
        == dict(
            diagnostic_ref=DIAGNOSTIC_REF,
            source_stage=24,
            sign=1,
            step_sizes=[0.001, 0.0005],
            request_indices=[r["request_index"] for r in training],
            response_indices=[r["response_index"] for r in training],
            terminal_index=4368,
            diagnostic_anchor_response_index=0,
            original_sqp_anchor_response_index=1819,
        ),
        "Exact source-only fit registration",
    )
    geometry_check = audit_terminal_secant_geometry(answer["geometry"])
    fit_check = audit_terminal_secant_fit(answer["fit"], source_answer=diagnostic["answer"])
    exact_array(
        "completed367_previous",
        np.asarray(original["previous_command"], dtype=float)[:6],
        answer["geometry"]["previous_command"],
    )
    control = read(remember(source_reg["control_ref"], refs))["contract"]
    for field in ("input_min", "input_max"):
        exact_array("original_" + field, control[field], answer["geometry"][field])
    n = producer["response_evaluations"]
    new_registration = publish_result(
        output / "registration.json",
        dict(
            experiment="terminal_secant_probe",
            producer_ref=producer_ref,
            producer_owner_ref=owner_ref,
            producer_registration_ref=producer["registration_ref"],
            plan_ref=reference(PLAN),
            seed=42,
            diagnostic_ref=DIAGNOSTIC_REF,
            diagnostic_closure_ref=DIAGNOSTIC_CLOSURE_REF,
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
            evidence_scope="All17 complete calls including final repeat; independently reconstructed fit, geometry and holdout residuals",
            automatic_retry=False,
        ),
    )
    result = dict(
        complete=False,
        passed=False,
        experiment="terminal_secant_probe",
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
        result["geometry_check"] = geometry_check
        result["fit_check"] = fit_check
        result["ledger_check"] = audit_terminal_secant_ledger(
            answer,
            historical_anchor=diagnostic["answer"]["evaluated_candidates"][0],
            source_answer=diagnostic["answer"],
            compare=compare,
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
