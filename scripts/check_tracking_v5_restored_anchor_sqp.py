"""Independently reconstruct every accepted-point SQP response and full ray record.

The native dynamics/classification helpers are frozen independent mathematics;
the new ray ledger is audited separately, without importing the producer solver.
"""

import argparse
import hashlib
import json
import os

import numpy as np
from check_tracking_v5_prefix_trust_control import (
    check_full_candidate,
    command_key,
    compare_contract,
    normalized_errors,
    numerical_compare,
    validate_trace_entry,
)
from check_tracking_v5_saved_direction_backtracking import (
    NUMERIC_TOLERANCE,
    TORQUE_TOLERANCE,
    native_math_proof,
)
from check_tracking_v5_saved_direction_backtracking import (
    case_engine as frozen_case_engine,
)
from restored_anchor_sqp_ledger_audit import audit_initial_record_guard, audit_restored_anchor_sqp_ledger

from s2a.native_environment import restore_cv2_environment, strict_native_environment
from s2a.preview_result_publication import publish_result as publish
from s2a.restored_anchor_sqp_evidence import REPO, model_registration, require
from s2a.terminal_development_execution import CONTROL_REF


def case_engine(name, original, source_reg, prefix_check_reg, result, refs, compare):
    migrated, proof = model_registration(source_reg, refs)
    result["asset_loading_proof"] = proof
    return frozen_case_engine(name, original, migrated, prefix_check_reg, result, refs, compare)


def full_ray_records(answer):
    """List all archived full ray responses; an uncompleted request has no data."""
    records = []
    for stage, phase in enumerate(answer["phases"]):
        for index, attempt in enumerate(phase["ray_attempts"]):
            if attempt["response_index"] is None:
                require(
                    attempt["entry"] is None and attempt["outcome"] == "budget_rejected",
                    "Only an explicit denied ray request has no completed data",
                )
                continue
            require(
                type(attempt["response_index"]) is int
                and 0 <= attempt["response_index"] < len(answer["evaluated_candidates"])
                and isinstance(attempt["entry"], dict),
                "Every completed ray has its full record",
            )
            records.append((stage, index, attempt))
    return records


def main(dry=False):
    case = "candidate"
    from s2a.restored_anchor_sqp_evidence import (
        CPU_ENVIRONMENT,
        NATIVE_PYTHON,
        PLAN,
        PREFIX_CHECK_REF,
        PREFIX_OPTIMIZATION_MARGIN,
        SOURCES,
        TAIL_CLOSURE_REF,
        TAIL_PRODUCER_REF,
        case_root,
        closed_owner,
        finish,
        load_prerequisites,
        read,
        reference,
        remember,
    )

    require(case == "candidate", "One registered accepted-point SQP candidate")
    strict_native_environment()
    require(
        {key: os.environ.get(key) for key in CPU_ENVIRONMENT} == CPU_ENVIRONMENT,
        "Registered CPU-only single-thread environment",
    )
    original, prefix_reg, tail, refs = load_prerequisites()
    proof = native_math_proof()
    remember(proof["source_ref"], refs)
    require(tail["answer"]["selected_response_index"] == 5, "Only the closed restored tail candidate")
    expected_initial = tail["answer"]["evaluated_candidates"][5]
    expected_ref = dict(**TAIL_PRODUCER_REF, locator="answer.evaluated_candidates[5]")
    warm = np.asarray(expected_initial["commands"], dtype=float)
    warm_key = command_key(warm, 12)
    prefix_check = read(remember(PREFIX_CHECK_REF, refs))
    prefix_check_reg = read(remember(prefix_check["registration_ref"], refs))
    finish(refs)
    if dry:
        print(
            json.dumps(
                dict(
                    ready=True,
                    dry=True,
                    case_name=case,
                    source_refs=len(refs),
                    warm_sha256=hashlib.sha256(warm.tobytes()).hexdigest(),
                    new_responses=0,
                    new_queries=0,
                    numerical_math_proof=proof,
                )
            ),
            flush=True,
        )
        return

    output = case_root(case, check=True)
    require(not output.exists(), "New independent result path; no overwrite or retry")
    owner_ref = reference(case_root(case, owner=True) / "result.json")
    owner, producer = closed_owner(owner_ref, refs)
    producer_ref = reference(case_root(case) / "verified.json")
    require(owner["result_ref"] == producer_ref, "Closed owner binds the intended new producer")
    launch = read(remember(owner["owned_command"]["launch"], refs))
    require(
        launch["command"] == [str(NATIVE_PYTHON), str(REPO / "scripts/probe_tracking_v5_restored_anchor_sqp.py")],
        "Only the registered producer command and case",
    )
    registration = read(remember(producer["registration_ref"], refs))
    require(
        registration["case_name"] == producer["case_name"] == case
        and registration["plan_ref"] == reference(PLAN)
        and registration["seed"] == 42
        and registration["experiment"] == "restored_anchor_sqp"
        and registration["prefix_optimization_margin"]
        == producer["prefix_optimization_margin"]
        == PREFIX_OPTIMIZATION_MARGIN,
        "Fixed common-warm case, preregistration, and seed",
    )
    require(
        registration["tail_producer_ref"] == TAIL_PRODUCER_REF
        and registration["warm_locator"] == expected_ref["locator"] + ".commands"
        and registration["expected_initial_ref"] == expected_ref
        and registration["tail_closure_ref"] == TAIL_CLOSURE_REF
        and registration["original_prefix_registration_ref"] == original["registration_ref"]
        and registration["endpoint"] == 368
        and registration["horizon"] == 12
        and registration["response_budget"] == 4000
        and registration["iteration_budget"] == registration["stage_budget"] == 60
        and registration["qp_iteration_limit"] == 100
        and registration["qp_total_iteration_limit"] == 6000
        and registration["qp_certificate_tolerance"] == 1e-6
        and registration["actual_consumed_reference_count"] == 0
        and registration["automatic_retry"] is False,
        "The previously qualified restored warm has the exact registered source",
    )
    for ref in registration["source_refs"] + registration["input_refs"]:
        remember(ref, refs)
    _, expected_asset_proof = model_registration(prefix_reg, refs)
    compare_contract(expected_asset_proof, producer["asset_loading_proof"], name="producer_assets")
    compare_contract(expected_asset_proof, registration["asset_loading_proof"], name="registered_assets")
    answer = producer["answer"]
    initial_check = audit_initial_record_guard(answer, expected_initial, expected_ref)
    require(
        producer["initial_record_guard"] == answer["initial_record_guard"],
        "The producer retains its complete first-record guard",
    )
    require(
        command_key(producer["warm"], 12) == warm_key,
        "New candidate uses the exact previously qualified common warm bytes",
    )
    for key in ("supplied_warm", "used_start"):
        require(command_key(answer[key], 12) == warm_key, f"No clipping or replacement of {key}")
    compare_contract(np.zeros((12, 6)), answer["start_roundoff_clip"], name="no_warm_clip")
    require(
        answer["horizon"] == answer["pose_caps_horizon"] == 12
        and answer["all_pose_caps_enforced"] is True
        and answer["max_evaluations"] == 4000
        and answer["max_iterations"] == 60
        and answer["finite_difference_step"] == 1e-4
        and answer["optimization_safety_buffer"] == 1e-8
        and answer["prefix_optimization_margin"] == PREFIX_OPTIMIZATION_MARGIN
        and answer["method"] == "restored_anchor_lagrangian_sqp"
        and answer["qp_iteration_limit"] == 100
        and answer["qp_certificate_tolerance"] == 1e-6
        and 0 <= answer["total_qp_iterations"] <= 6000,
        "Original complete H12 caps and fixed search budgets",
    )
    if case == "candidate":
        require(answer["max_stages"] == 60, "Registered finite stage budget")
    compare_contract(original["terminal_normals"], answer["terminal_normals"], name="fixed_terminal")
    compare_contract(original["past_inner_error_rad"], answer["past_inner_error_rad"], name="zero_history")
    compare_contract(original["answer"]["lower_command"], answer["lower_command"], name="base_lower_bounds")
    compare_contract(original["answer"]["upper_command"], answer["upper_command"], name="base_upper_bounds")
    trace = answer["evaluated_candidates"]
    n = len(trace)
    require(1 <= n <= 4000, "Registered finite full response count")
    require(
        producer["prediction_calls"] == producer["response_evaluations"] == answer["response_evaluations"] == n
        and producer["total_iterations"] == answer["total_iterations"] <= 60
        and producer["numerical_inner_steps"] == 300 * n
        and producer["query_count"] == 1 + 312 * n,
        "Every unique complete response is charged exactly once",
    )
    require(
        producer["control_qualified"] is False
        and all(
            producer[key] == 0
            for key in ("physics_steps", "world_model_calls", "fresh_cases", "actual_consumed_reference_count")
        ),
        "Numerical search does not qualify actual control or task performance",
    )
    new_registration = publish(
        output / "registration.json",
        dict(
            case_name=case,
            producer_ref=producer_ref,
            producer_owner_ref=owner_ref,
            producer_registration_ref=producer["registration_ref"],
            plan_ref=reference(PLAN),
            expected_initial_ref=expected_ref,
            initial_record_check=initial_check,
            seed=42,
            source_refs=[reference(REPO / path) for path in SOURCES],
            input_refs=list(refs.values()),
            numerical_math_proof=proof,
            numeric_tolerance=NUMERIC_TOLERANCE,
            torque_tolerance=TORQUE_TOLERANCE,
            expected_responses=n,
            expected_inner_steps=300 * n,
            expected_queries=1 + 312 * n,
            evidence_scope="Every unique response independently reconstructed; full initial/selected/closest and every completed ray rollout/audit checked, plus all accepted anchor audits",
            automatic_retry=False,
        ),
    )
    result = dict(
        complete=False,
        passed=False,
        case_name=case,
        registration_ref=new_registration,
        prefix_optimization_margin=PREFIX_OPTIMIZATION_MARGIN,
        initial_record_check=initial_check,
        response_evaluations=0,
        verified_responses=0,
        numerical_inner_steps=0,
        mj_forward_queries=0,
        query_count=0,
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
            "prefix", original, prefix_reg, prefix_check_reg, result, refs, compare
        )
        compare_contract(current, producer["current_quantities"], compare=compare, name="new_current")
        full = {key: answer[key] for key in ("initial", "selected", "closest_safety_attempt")}
        full_keys = {key: command_key(value["commands"], 12) for key, value in full.items()}
        checked_full, classifications, saved_safety_ranks = set(), [], []
        anchors = answer["accepted_anchors"]
        checked_anchors = set()
        rays = full_ray_records(answer)
        checked_rays = set()
        previous = np.asarray(original["previous_command"][:6], dtype=float)
        saved_previous = np.asarray(answer["previous_command"], dtype=float)
        require(
            previous.shape == saved_previous.shape == (6,) and previous.tobytes() == saved_previous.tobytes(),
            "Exact completed367 command bytes for rate and algorithm reconstruction",
        )
        control = read(remember(prefix_reg["control_ref"], refs))["contract"]
        lower, upper = np.asarray(control["input_min"]), np.asarray(control["input_max"])
        for index, saved in enumerate(trace):
            commands = np.asarray(saved["commands"], dtype=float)
            key = command_key(commands, 12)
            rollout, _ = reconstruct(commands)
            result["response_evaluations"] += 1
            result["query_count"] = result["mj_forward_queries"]
            metrics = classify(commands, rollout)
            require(
                set(saved)
                == {
                    "commands",
                    "safety_min",
                    "safety_slack",
                    "joint_feasible",
                    "tcp_errors",
                    "prefix_length",
                    "feasible",
                },
                "Exact trace schema including all nonlinear safety values",
            )
            try:
                normalized = validate_trace_entry(
                    {name: value for name, value in saved.items() if name != "safety_slack"}, metrics, compare=compare
                )
            except ValueError:
                result["boundary_classification_disagreements"].append(
                    dict(
                        trace_index=index,
                        producer_prefix=saved["prefix_length"],
                        independent_prefix=metrics["prefix_length"],
                        producer_joint=saved["joint_feasible"],
                        independent_joint=metrics["joint_feasible"],
                        producer_feasible=saved["feasible"],
                        independent_feasible=metrics["feasible"],
                        producer_normalized=normalized_errors(saved["tcp_errors"], 12),
                        independent_normalized=metrics["normalized_squared_errors"],
                        evidence="Numerical check rejected; no inference of actual physical violation or search success",
                    )
                )
                raise
            compare("trace_safety_slack", metrics["safety_slack"], saved["safety_slack"])
            classifications.append(
                {
                    key: metrics[key]
                    for key in (
                        "joint_feasible",
                        "feasible",
                        "prefix_length",
                        "normalized_squared_errors",
                        "tcp_errors",
                        "safety_min",
                        "safety_slack",
                    )
                }
            )
            rate = np.diff(np.vstack((previous, commands)), axis=0)
            worst = min(
                saved["safety_min"],
                float(np.min(commands - lower)),
                float(np.min(upper - commands)),
                float(np.min(0.25 - rate)),
                float(np.min(0.25 + rate)),
            )
            saved_safety_ranks.append((max(0.0, -worst), float(np.sum(normalized)), index))
            for label, expected in full_keys.items():
                if key == expected:
                    check_full_candidate(
                        full[label], commands, rollout, metrics, compare_rollout, check_audit, compare=compare
                    )
                    checked_full.add(label)
            for anchor_index, anchor in enumerate(anchors):
                if anchor["response_index"] != index:
                    continue
                require(command_key(anchor["commands"], 12) == key, "Accepted anchor binds exact response bytes")
                check_audit(anchor["audit"], metrics)
                for field in ("tcp_errors", "normalized_squared_errors", "safety_min"):
                    compare("anchor_" + field, metrics[field], anchor[field])
                require(
                    anchor["joint_feasible"] is metrics["joint_feasible"]
                    and anchor["feasible"] is metrics["feasible"]
                    and type(anchor["prefix_length"]) is int
                    and anchor["prefix_length"] == metrics["prefix_length"],
                    "Independent strict accepted-anchor classification",
                )
                checked_anchors.add(anchor_index)
            for stage_index, attempt_index, attempt in rays:
                if attempt["response_index"] != index:
                    continue
                require(command_key(attempt["commands"], 12) == key, "Ray point binds its exact response")
                check_full_candidate(
                    attempt["entry"], commands, rollout, metrics, compare_rollout, check_audit, compare=compare
                )
                checked_rays.add((stage_index, attempt_index))
            result["verified_responses"] += 1
            require(
                result["numerical_inner_steps"] == 300 * (index + 1)
                and result["mj_forward_queries"] == 1 + 312 * (index + 1),
                "Every trace response independently resets and completes all 12 predicted periods",
            )
            if (index + 1) % 25 == 0 or index + 1 == n:
                print(json.dumps(dict(case_name=case, checked=index + 1, total=n)), flush=True)
        require(checked_full == set(full), "Every retained full candidate is present in the unique trace")
        require(
            checked_anchors == set(range(len(anchors))),
            "Every accepted anchor has an independently checked complete audit",
        )
        require(
            checked_rays == {(stage, attempt) for stage, attempt, _ in rays},
            "Every completed ray has its entire archived rollout and audit checked",
        )
        closest_index = min(range(n), key=lambda i: saved_safety_ranks[i])
        require(
            command_key(trace[closest_index]["commands"], 12) == full_keys["closest_safety_attempt"],
            "Exact saved minimum-violation then full-loss closest candidate ranking",
        )
        result["ledger_check"] = audit_restored_anchor_sqp_ledger(
            answer, classifications, expected_initial=expected_initial, expected_ref=expected_ref, compare=compare
        )
        result["ledger_check"]["passed"] = True
        require(bool(compiled_goal.nopython_signatures), "Native quaternion goal actually compiled")
        result["native_quaternion_jit_signatures"] = [str(value) for value in compiled_goal.nopython_signatures]
        for ref in producer["source_refs_unchanged"]:
            remember(ref, refs)
        finish(refs)
        strict_native_environment()
        result.update(
            complete=True,
            passed=True,
            full_candidates_checked=sorted(checked_full),
            accepted_anchor_audits_checked=len(checked_anchors),
            complete_ray_rollouts_checked=len(checked_rays),
            closest_trace_index=closest_index,
            source_refs_unchanged=list(refs.values()),
            witness_found=bool(result["ledger_check"]["witness_indices"]),
            evidence="Independent local numerical and search-ledger qualification only; no physical control or task qualification",
        )
    except BaseException as error:
        result["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        result["query_count"] = result["mj_forward_queries"]
        publish(output / "verified.json", result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry", action="store_true")
    args = parser.parse_args()
    main(args.dry)
