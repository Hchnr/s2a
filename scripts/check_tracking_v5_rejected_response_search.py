"""Audit the feedback variant and independently replay every charged response."""

import argparse
import json
import os

import numpy as np
from check_tracking_v5_accepted_point_sqp import case_engine
from check_tracking_v5_interior_trajectory_search import verify_fresh_decisions
from check_tracking_v5_prefix_trust_control import compare_contract
from check_tracking_v5_saved_direction_backtracking import NUMERIC_TOLERANCE, TORQUE_TOLERANCE, native_math_proof
from check_tracking_v5_tail_translation_reserve_v2 import semantic_compare
from check_tracking_v5_terminal_secant_probe import verify_response_trace
from rejected_response_search_audit import audit_search

from s2a.accepted_point_sqp_evidence import PREFIX_CHECK_REF
from s2a.native_environment import restore_cv2_environment, strict_native_environment
from s2a.preview_result_publication import publish_result
from s2a.restored_anchor_sqp_evidence import (
    CPU_ENVIRONMENT,
    REPO,
    finish,
    model_registration,
    read,
    reference,
    remember,
    require,
)

ROOT = REPO / "outputs/tracking_v5_rejected_response_search_check_20261029"


def verify_fresh_feedback(answer, metrics, config):
    """Re-select each feedback source using independently recomputed actual values."""
    require(len(metrics) == len(answer["evaluated_candidates"]), "Every feedback source has a fresh full response")
    ceiling = np.asarray(answer["pose_ceiling"])
    sources = []
    for stage in answer["stages"]:
        eligible = []
        if stage["accepted_response_index"] is None:
            for attempt in stage["attempts"]:
                index = attempt["response_index"]
                if index is None or attempt["accepted"] or attempt["acceptance_reason"] != "actual_admission_failed":
                    continue
                metric = metrics[index]
                safety = np.asarray(metric["safety_slack"])
                squared = np.asarray(metric["normalized_squared_errors"]).ravel()
                safe = metric["joint_feasible"] and np.all(safety >= config["safety_buffer"])
                step = np.asarray(attempt["actual_step"])
                if safe and np.any(squared > ceiling) and np.sum(step**2) >= config["secant_min_squared_step"]:
                    eligible.append((attempt["alpha"], index))
        selected = min(eligible)[1] if eligible else None
        saved = stage["rejected_update"]
        require((saved is None) == (selected is None), "Fresh rejected-feedback availability")
        if selected is not None:
            require(saved["source_response_index"] == selected, "Fresh minimum-alpha safe feedback source")
            sources.append(selected)
    require(len(sources) == answer["rejected_update_count"], "Fresh feedback update count")
    return dict(
        passed=True,
        source_response_indices=sources,
        update_count=len(sources),
        strict_disagreements=[],
        new_response_calls=0,
        control_qualified=False,
    )


def main(dry=False):
    from probe_tracking_v5_rejected_response_search import CONFIG, PLAN, initial_model, load_inputs
    from probe_tracking_v5_rejected_response_search import ROOT as PRODUCER_ROOT
    from probe_tracking_v5_terminal_only_tail_repair import load_environment_guard

    strict_native_environment()
    require(
        all(os.environ.get(key) == value for key, value in CPU_ENVIRONMENT.items()), "Single-thread CPU environment"
    )
    bundle, config, refs = load_inputs()
    Js, Jq = initial_model(bundle)
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
        registration["experiment"] == "rejected_response_search"
        and registration["config"] == config
        and registration["config_ref"] == reference(CONFIG)
        and registration["plan_ref"] == reference(PLAN)
        and registration["historical_response_index"] == 3557
        and registration["seed"] == 42
        and registration["response_budget"] == config["max_responses"]
        and registration["qp_budget"] == config["max_stages"]
        and registration["automatic_retry"] is False,
        "Exact preregistered search",
    )
    require(
        registration["feedback_policy"] == "safety_qualified_pose_ceiling_rejection_minimum_alpha",
        "Registered feedback policy",
    )
    for ref in registration["source_refs"] + registration["input_refs"] + producer["source_refs_unchanged"]:
        remember(ref, refs)
    answer = producer["answer"]
    ledger = audit_search(
        answer,
        Js,
        Jq,
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
            experiment="rejected_response_search",
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
