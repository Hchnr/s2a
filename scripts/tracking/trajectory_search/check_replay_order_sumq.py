"""Finite same-predictor sum-q baseline with independent full decision audit."""

import copy
import os
import time
import traceback
from pathlib import Path

import numpy as np

from s2a.controllers.osc import OSCInterface
from s2a.dynamics.mujoco_robot import MujocoRobotDynamics
from s2a.infrastructure.native_environment import restore_cv2_environment, strict_native_environment
from s2a.infrastructure.publication import publish_result, result_json
from s2a.research.trajectory_search.accepted_point_sqp_evidence import PREFIX_CHECK_REF
from s2a.research.trajectory_search.inputs import model_registration
from s2a.research.trajectory_search.replay_order_trajectory import rollout_replay_order_trajectory
from s2a.research.trajectory_search.response_stencil import build_response_stencil
from s2a.research.trajectory_search.restored_anchor_sqp_evidence import (
    CPU_ENVIRONMENT,
    REPO,
    checked,
    read,
    reference,
    remember,
    require,
)
from s2a.research.trajectory_search.state_vector_dual_recovery import search
from scripts.tracking.trajectory_search.check_replay_order_stencil import compare_values
from scripts.tracking.trajectory_search.check_tracking_v5_rejected_response_search import verify_fresh_feedback
from scripts.tracking.trajectory_search.check_tracking_v5_state_vector_gn_search import verify_fresh_decisions
from scripts.tracking.trajectory_search.probe_tracking_v5_interior_trajectory_search import run_search
from scripts.tracking.trajectory_search.probe_tracking_v5_state_vector_dual_recovery import load_inputs
from scripts.tracking.trajectory_search.probe_tracking_v5_terminal_only_tail_repair import load_environment_guard
from scripts.tracking.trajectory_search.state_vector_dual_recovery_audit import audit_search
from scripts.verification.trajectory_search.check_tracking_v5_saved_direction_backtracking import case_engine
from scripts.verification.trajectory_search.check_tracking_v5_tail_translation_reserve_v2 import semantic_compare
from scripts.verification.trajectory_search.check_tracking_v5_terminal_secant_probe import verify_response_trace

OUTPUT = REPO / "outputs/replay_order_sumq_20261102"
CONFIG = REPO / "configs/tracking/trajectory_search/replay_order_sumq_20261102.json"
PLAN = REPO / "plans/v0.0.1/tracking/trajectory_search/replay_order_sumq_20261102.md"
STENCIL = REPO / "outputs/replay_order_stencil_20261102/stencil.json"
QUALIFICATION = REPO / "outputs/replay_order_stencil_20261102/result.json"


def validate_spec(spec, original):
    require(
        spec["search_config"] == original and original["seed"] == spec["seed"] == 42,
        "Unchanged complete original search config",
    )
    expected = dict(
        producer_response_budget=122,
        independent_response_budget=122,
        total_response_budget=244,
        total_inner_step_budget=73200,
        total_query_budget=76130,
        model_budget=2,
        qp_budget=60,
        qp_iteration_budget=6000,
        tool_policy="native_solve",
        automatic_retry=False,
    )
    require(all(spec.get(key) == value for key, value in expected.items()), "Fixed independent new-run budgets/policy")
    require(original["max_stages"] == 60 and original["max_responses"] == 122, "Original stage/response budget")
    return original


def main():
    strict_native_environment()
    require(all(os.environ.get(k) == v for k, v in CPU_ENVIRONMENT.items()), "Single-thread CPU")
    require(not OUTPUT.exists(), "Exclusive output; no overwrite or retry")
    bundle, original, context = load_inputs()
    refs = context.runtime_refs
    spec = read(reference(CONFIG))
    config = validate_spec(spec, original)
    qualification_ref = remember(reference(QUALIFICATION), refs)
    require(
        qualification_ref["sha256"] == "80e1fd703fe67a8d44ad90d5b603e86e1af7d9fba50084dc6009ea71157792c5",
        "Exact new stencil qualification",
    )
    qualified = read(qualification_ref)
    require(
        qualified["complete"]
        and qualified["passed"]
        and qualified["all_values_exact"]
        and qualified["sources_unchanged"],
        "New predictor qualification",
    )
    require(
        len(qualified["processes"]) == 32
        and all(p["actual_wait"] and p["reaped"] and p["exitcode"] == 0 for p in qualified["processes"]),
        "All stencil workers reaped",
    )
    stencil_ref = remember(reference(STENCIL), refs)
    require(stencil_ref == qualified["stencil_ref"], "Exact qualified stencil")
    del qualified
    stencil = read(stencil_ref)
    matrices = build_response_stencil(stencil["base"], stencil["probes"], stencil["nominal_deltas"])
    for key, value in matrices.items():
        np.testing.assert_array_equal(value, stencil["matrices"][key])
    source, source_reg, control = (bundle[k] for k in ("source", "source_registration", "control"))
    model_reg, proof = model_registration(source_reg, context, bundle)
    source = copy.deepcopy(source)
    control_record = read(remember(source_reg["control_ref"], refs))
    tool_record = read(remember(source_reg["tool_ref"], refs))
    source["site_from_tcp"] = np.linalg.solve(
        np.asarray(control_record["source_from_site"]), np.asarray(tool_record["source_from_tcp"])
    )
    guard = load_environment_guard(refs)
    prefix = read(remember(PREFIX_CHECK_REF, refs))
    prefix_reg = read(remember(prefix["registration_ref"], refs))
    local_paths = [
        Path(__file__),
        CONFIG,
        PLAN,
        REPO / "src/s2a/research/trajectory_search/response_stencil.py",
        REPO / "src/s2a/research/trajectory_search/replay_order_trajectory.py",
        REPO / "src/s2a/research/response_models/forward/native_replay_joint_response.py",
        REPO / "src/s2a/research/response_models/forward/native_osc_torque.py",
    ]
    before = [reference(p) for p in local_paths]
    context.finish()
    registration = publish_result(
        OUTPUT / "registration.json",
        dict(
            experiment="replay_order_sumq",
            config=spec,
            source_refs=before,
            input_refs=list(refs.values()),
            qualification_ref=qualification_ref,
            stencil_ref=stencil_ref,
            asset_proof=proof,
            tool=source["site_from_tcp"],
        ),
    )
    producer = dict(
        complete=False,
        passed=False,
        prediction_calls=0,
        response_evaluations=0,
        unique_command_count=0,
        numerical_inner_steps=0,
        query_count=0,
        answer=None,
        control_qualified=False,
    )
    checker = dict(
        complete=False,
        passed=False,
        response_evaluations=0,
        verified_responses=0,
        mj_forward_queries=0,
        query_count=0,
        numerical_inner_steps=0,
        repeated_responses_checked=0,
        unique_command_count=0,
        max_jit_vs_python_goal_difference=0.0,
        jit_python_goal_differences=0,
        boundary_classification_disagreements=[],
        fresh_decision_diagnostics=[],
        control_qualified=False,
    )
    result = dict(
        complete=False,
        passed=False,
        registration_ref=registration,
        models=0,
        model_attempts=0,
        qp_solves=0,
        nnls_fit_attempts=0,
        physics_steps=0,
        world_model_calls=0,
        fresh_cases=0,
        automatic_retry=False,
        control_qualified=False,
    )
    started = time.monotonic()
    dynamics = None
    try:
        result["environment_admission"] = restore_cv2_environment(
            guard["exact_probe_ref"], guard["runtime_source_refs"]
        )
        os.environ.pop("MUJOCO_GL", None)
        os.environ.pop("PYOPENGL_PLATFORM", None)
        result["model_attempts"] += 1
        dynamics = MujocoRobotDynamics(
            checked(model_reg["xml_ref"]),
            control["controller_site"],
            [f"robot0_joint{k}" for k in range(1, 8)],
            source["fixed_gripper_state"],
        )
        result["models"] += 1
        interface = OSCInterface(*(control[k] for k in ("input_min", "input_max", "output_min", "output_max")))

        def vector_search(evaluate, historical, Js, Jq, previous, lo, hi, cfg):
            return search(evaluate, historical, Js, Jq, matrices["Je"].copy(), previous, lo, hi, cfg)

        run_search(
            source,
            stencil["base"],
            matrices["Js"],
            matrices["Jq"],
            config,
            interface,
            dynamics,
            producer,
            search_fn=vector_search,
            rollout_fn=rollout_replay_order_trajectory,
        )
        producer.update(complete=True, passed=True)
        result["producer_ref"] = publish_result(OUTPUT / "producer.json", producer)
        answer = producer["answer"]
        result["ledger"] = audit_search(
            answer,
            matrices["Js"],
            matrices["Jq"],
            matrices["Je"],
            stencil["base"],
            np.asarray(source["previous_command"])[:6],
            interface.input_min,
            interface.input_max,
            config,
        )
        result["model_attempts"] += 1
        reconstruct, compare_rollout, classify, check_audit, _, _ = case_engine(
            "prefix",
            source,
            model_reg,
            prefix_reg,
            checker,
            refs,
            lambda label, a, b, tolerance=1e-10: semantic_compare(label, a, b, tolerance),
            input_context=context,
        )
        result["models"] += 1
        rollouts, metrics = [], []

        def retain_replay(commands):
            rollout, differences = reconstruct(commands)
            rollouts.append(rollout)
            return rollout, differences

        def retain_classify(commands, rollout):
            metric = classify(commands, rollout)
            metrics.append(metric)
            return metric

        verify_response_trace(
            answer,
            retain_replay,
            compare_rollout,
            retain_classify,
            check_audit,
            checker,
            compare=lambda name, a, b, tolerance=1e-10: semantic_compare(name, a, b, tolerance),
        )
        checker["independent_search_check"] = verify_fresh_decisions(
            answer, rollouts, metrics, config, diagnostics=checker["fresh_decision_diagnostics"]
        )
        checker["independent_feedback_check"] = verify_fresh_feedback(answer, metrics, config)
        checker["exact_response_values"] = all(
            r["exact"]
            for entry, rollout in zip(answer["evaluated_candidates"], rollouts, strict=True)
            for r in compare_values(entry["rollout"], rollout)
        )
        checker.update(complete=True, passed=True)
        count = producer["response_evaluations"]
        require(checker["verified_responses"] == count <= 122, "All actual responses independently audited")
        result["total_queries"] = producer["query_count"] + checker["mj_forward_queries"]
        require(result["total_queries"] == 2 + 624 * count <= spec["total_query_budget"], "Every new query charged")
        result.update(complete=True, passed=True)
    except BaseException as error:
        result.update(error=repr(error), traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        if dynamics is not None:
            producer["actual_model_queries"] = dynamics.queries
        if producer["answer"] is not None:
            qps = [s["qp"] for s in producer["answer"]["stages"] if s["qp"] is not None]
            result["qp_solves"] = len(qps)
            result["nnls_fit_attempts"] = sum(q["nnls_fit_attempts"] for q in qps)
        try:
            context.finish()
            strict_native_environment()
            result["sources_unchanged"] = before == [reference(p) for p in local_paths]
            result["passed"] = result["passed"] and result["sources_unchanged"]
        except BaseException as error:
            result.update(source_validation_error=repr(error), passed=False)
        result["producer_final_ref"] = publish_result(OUTPUT / "producer_final.json", producer)
        result["checker_ref"] = publish_result(OUTPUT / "checker.json", checker)
        result["elapsed_seconds"] = time.monotonic() - started
        receipt = publish_result(OUTPUT / "result.json", result)
        print(result_json(dict(result_ref=receipt, complete=result["complete"], passed=result["passed"])), flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
