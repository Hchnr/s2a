"""Matched finite searches, each independently replayed through native control math."""

import argparse
import copy
import os
import signal
import traceback

import numpy as np

from s2a.controllers.osc import OSCInterface
from s2a.dynamics.mujoco_robot import MujocoRobotDynamics
from s2a.infrastructure.native_environment import restore_cv2_environment, strict_native_environment
from s2a.infrastructure.publication import publish_result, result_json
from s2a.research.trajectory_search.cap_distance_objective import CapDistanceObjective
from s2a.research.trajectory_search.objective_search_driver import search
from s2a.research.trajectory_search.replay_order_trajectory import rollout_replay_order_trajectory
from s2a.research.trajectory_search.reserve_aligned_cap_objective import POLICY as ALIGNED_POLICY
from s2a.research.trajectory_search.reserve_aligned_cap_objective import ReserveAlignedCapObjective
from s2a.research.trajectory_search.reserve_entry_acceptance import InheritedSearchDomain, ReserveEntryAcceptance
from s2a.research.trajectory_search.reserve_safety_feedback import POLICY as SAFETY_FEEDBACK_POLICY
from s2a.research.trajectory_search.reserve_safety_feedback import ReserveSafetyFeedback
from s2a.research.trajectory_search.restored_anchor_sqp_evidence import REPO, checked, read, reference, require
from scripts.maintenance.run_cpu_tests import snapshot
from scripts.tracking.trajectory_search.cap_distance_search_audit import audit_cap_search
from scripts.tracking.trajectory_search.check_cap_distance_gn import phase_deadline
from scripts.tracking.trajectory_search.check_replay_order_stencil import compare_values
from scripts.tracking.trajectory_search.probe_tracking_v5_interior_trajectory_search import run_search
from scripts.tracking.trajectory_search.reserve_aligned_cap_audit import ReserveAlignedCapAuditPolicy
from scripts.tracking.trajectory_search.response_probe_runtime import ResponseProbeRuntime
from scripts.verification.trajectory_search.check_tracking_v5_prefix_trust_control import check_full_candidate
from scripts.verification.trajectory_search.check_tracking_v5_saved_direction_backtracking import case_engine
from scripts.verification.trajectory_search.check_tracking_v5_tail_translation_reserve_v2 import semantic_compare

CONFIG = REPO / "configs/tracking/trajectory_search/reserve_entry_search_20261102.json"
PLAN = REPO / "plans/v0.0.1/tracking/trajectory_search/reserve_entry_search_20261102.md"
QA = REPO / "outputs/qa/reserve_entry_search_final_20261102/result.json"
ROOT = REPO / "outputs/reserve_entry_search_20261102"


def registered_arms(aligned_floor=False, safety_feedback=False, progress_floor=None):
    require(not safety_feedback or aligned_floor, "Registered feedback constraint mode")
    if progress_floor is not None:
        require(
            progress_floor == 1e-5 and aligned_floor and safety_feedback, "Only the fixed two-field progress variant"
        )
        return ["small_progress"]
    return ["safety_feedback"] if safety_feedback else (["aligned"] if aligned_floor else ["baseline", "reserve"])


def registered_config(original, progress_floor=None):
    config = copy.deepcopy(original)
    if progress_floor is not None:
        require(progress_floor == 1e-5, "Fixed progress floor")
        for key in ("minimum_predicted_reduction", "minimum_actual_reduction"):
            require(original[key] == 1e-4, "Original progress floor identity")
            config[key] = progress_floor
    return config


def validate_spec(spec, original, *, aligned_floor=False, safety_feedback=False, progress_floor=None):
    require(not safety_feedback or aligned_floor, "Safety feedback uses the aligned constraint registration")
    expected = dict(
        experiment="small_progress_search"
        if progress_floor is not None
        else "reserve_safety_search"
        if safety_feedback
        else ("aligned_reserve_search" if aligned_floor else "reserve_entry_search"),
        seed=42,
        arms=registered_arms(aligned_floor, safety_feedback, progress_floor),
        search_config=registered_config(original, progress_floor),
        producer_response_budget_per_arm=122,
        independent_response_budget_per_arm=122,
        total_response_budget=244 if aligned_floor else 488,
        total_inner_step_budget=73200 if aligned_floor else 146400,
        total_query_budget=76130 if aligned_floor else 152260,
        model_budget=2 if aligned_floor else 4,
        qp_budget_per_arm=60,
        qp_iteration_budget_per_arm=6000,
        nnls_fit_budget_per_arm=60,
        preflight_timeout_seconds=300,
        initialization_timeout_seconds=300,
        execution_timeout_seconds=1200,
        owner_timeout_seconds=1800,
        automatic_retry=False,
    )
    if aligned_floor:
        expected["problem_policy"] = ALIGNED_POLICY
    if safety_feedback:
        expected["reserve_feedback_policy"] = SAFETY_FEEDBACK_POLICY
    if progress_floor is not None:
        expected["progress_policy"] = "minimum_cap_decrease_1e_5"
    require(
        spec == expected and original["max_stages"] == 60 and original["max_responses"] == 122,
        "Fixed matched arms, original config and all finite budgets",
    )


def native_metric_entry(entry, rollout, metric):
    """Replace all fields used by decision audits with independently obtained values."""
    rebuilt = copy.deepcopy(entry)
    rebuilt["rollout"] = rollout
    for key in [
        "safety_slack",
        "tcp_errors",
        "normalized_squared_errors",
        "feasible",
        "prefix_length",
        "maximum_normalized_squared_error",
        "maximum_pose_cap_violation",
        "pose_loss",
    ]:
        require(all(r["exact"] for r in compare_values(entry[key], metric[key])), "Exact native decision field: " + key)
        rebuilt[key] = copy.deepcopy(metric[key])
    require(entry["audit"]["feasible"] == metric["joint_feasible"], "Exact native joint feasibility")
    rebuilt["audit"]["feasible"] = metric["joint_feasible"]
    return rebuilt


def main(
    arm,
    *,
    config_path=CONFIG,
    plan_path=PLAN,
    qa_path=QA,
    output_root=ROOT,
    aligned_floor=False,
    safety_feedback=False,
    progress_floor=None,
):
    phase_deadline(300)
    require(arm in registered_arms(aligned_floor, safety_feedback, progress_floor), "Explicit registered arm")
    spec = read(reference(config_path))
    output = output_root / arm
    runtime = ResponseProbeRuntime(output, qa_path)
    prior = runtime.load(
        reference(REPO / "outputs/cap_distance_gn_20261102/result.json"),
        "80f58cde84778d1e3b2f7ff3ca0b85c5414419da976a057710421d23f03012e9",
    )
    cap = runtime.load(prior["producer_final_ref"])["answer"]
    cfg = spec["search_config"]
    validate_spec(
        spec, cap["config"], aligned_floor=aligned_floor, safety_feedback=safety_feedback, progress_floor=progress_floor
    )
    require(
        cfg == registered_config(cap["config"], progress_floor)
        and cfg["max_stages"] == 60
        and cfg["max_responses"] == 122,
        "Exact matched original config",
    )
    qualified = runtime.load(
        reference(REPO / "outputs/anchor85_coupled_stencil_20261102/result.json"),
        "bee4d9399fbbda691218c8f4ec9f9739f0026d072c1d9c51f3ab097bee7c79ce",
    )
    require(
        qualified["passed"]
        and qualified["complete"]
        and qualified["sources_unchanged"]
        and qualified["exact_native_responses"],
        "Qualified refreshed model",
    )
    stencil = runtime.load(qualified["stencil_ref"], "f4bc18043a6c1de5ae40e3c266eb7b5cb4cbe1ea9ac9af137ecf6e2058cac417")
    anchor = stencil["base"]
    models = {k: np.asarray(stencil["matrices"][k]) for k in ("Js", "Jq", "Je")}
    root_entry = cap["evaluated_candidates"][0]
    domain = InheritedSearchDomain.from_entry(root_entry)
    previous = np.asarray(runtime.source["previous_command"])[:6]
    interface = OSCInterface(*(runtime.control[k] for k in ("input_min", "input_max", "output_min", "output_max")))
    lower, upper = domain.bounds(previous, interface.input_min, interface.input_max, cfg)
    for left, key in [(lower, "global_lower"), (upper, "global_upper"), (domain.ceiling(cfg), "pose_ceiling")]:
        np.testing.assert_array_equal(left, cap[key])
    runtime.context.finish()
    registration = publish_result(
        output / "registration.json",
        dict(
            arm=arm,
            spec=spec,
            config_ref=reference(config_path),
            plan_ref=reference(plan_path),
            qa_ref=runtime.qa_ref,
            input_refs=list(runtime.refs.values()),
            source_before_ref=publish_result(output / "source_before.json", runtime.before),
            domain=domain.metadata(),
            initial_model_ref=qualified["stencil_ref"],
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
    native = dict(
        mj_forward_queries=0,
        numerical_inner_steps=0,
        max_jit_vs_python_goal_difference=0.0,
        jit_python_goal_differences=0,
    )
    result = dict(
        complete=False,
        passed=False,
        pid=os.getpid(),
        arm=arm,
        registration_ref=registration,
        models=0,
        model_attempts=0,
        qp_solves=0,
        nnls_fit_attempts=0,
        verified_responses=0,
        physics_steps=0,
        world_model_calls=0,
        fresh_cases=0,
        control_qualified=False,
        automatic_retry=False,
    )
    dynamics = None
    try:
        result["environment_admission"] = restore_cv2_environment(
            runtime.guard["exact_probe_ref"], runtime.guard["runtime_source_refs"]
        )
        os.environ.pop("MUJOCO_GL", None)
        os.environ.pop("PYOPENGL_PLATFORM", None)
        phase_deadline(spec["initialization_timeout_seconds"])
        result["model_attempts"] += 1
        dynamics = MujocoRobotDynamics(
            checked(runtime.model_reg["xml_ref"]),
            runtime.control["controller_site"],
            [f"robot0_joint{k}" for k in range(1, 8)],
            runtime.source["fixed_gripper_state"],
        )
        result["models"] += 1

        def vector_search(evaluate, historical, Js, Jq, previous, lo, hi, config):
            return search(
                evaluate,
                historical,
                Js,
                Jq,
                models["Je"].copy(),
                previous,
                lo,
                hi,
                config,
                objective=ReserveAlignedCapObjective() if aligned_floor else CapDistanceObjective(),
                domain=domain,
                pose_acceptance=ReserveEntryAcceptance() if arm != "baseline" else None,
                reserve_feedback=ReserveSafetyFeedback() if safety_feedback else None,
            )

        phase_deadline(spec["execution_timeout_seconds"])
        run_search(
            runtime.source,
            anchor,
            models["Js"],
            models["Jq"],
            cfg,
            interface,
            dynamics,
            producer,
            search_fn=vector_search,
            rollout_fn=rollout_replay_order_trajectory,
        )
        producer.update(complete=True, passed=True)
        answer = producer["answer"]

        def audit(record):
            return audit_cap_search(
                record,
                models["Js"],
                models["Jq"],
                models["Je"],
                anchor,
                previous,
                interface.input_min,
                interface.input_max,
                cfg,
                domain_reference=root_entry,
                preserve_pose_reserve=arm != "baseline",
                reserve_safety_feedback=safety_feedback,
                progress_reference=cap["config"] if progress_floor is not None else None,
                objective_audit=ReserveAlignedCapAuditPolicy() if aligned_floor else None,
            )

        result["production_ledger"] = audit(answer)
        publish_result(output / "producer.json", producer)
        phase_deadline(spec["initialization_timeout_seconds"])
        result["model_attempts"] += 1
        reconstruct, compare_rollout, classify, check_audit, _, _ = case_engine(
            "prefix",
            runtime.source,
            runtime.model_reg,
            runtime.prefix_reg,
            native,
            runtime.refs,
            lambda n, a, b, tolerance=1e-10: semantic_compare(n, a, b, tolerance),
            input_context=runtime.context,
        )
        result["models"] += 1
        phase_deadline(spec["execution_timeout_seconds"])
        fresh = copy.deepcopy(answer)
        exact = True
        for i, entry in enumerate(answer["evaluated_candidates"]):
            command = np.asarray(entry["commands"])
            rollout, _ = reconstruct(command)
            metric = classify(command, rollout)
            check_full_candidate(
                entry,
                command,
                rollout,
                metric,
                compare_rollout,
                check_audit,
                compare=lambda n, a, b, tolerance=1e-10: semantic_compare(n, a, b, tolerance),
            )
            exact = exact and all(r["exact"] for r in compare_values(entry["rollout"], rollout))
            fresh["evaluated_candidates"][i] = native_metric_entry(entry, rollout, metric)
            result["verified_responses"] += 1
            publish_result(output / "independent" / f"response_{i:03d}.json", dict(rollout=rollout, metrics=metric))
            if i % 12 == 0:
                print(result_json(dict(arm=arm, verified=i + 1, total=len(answer["evaluated_candidates"]))), flush=True)
        require(exact, "Every native response value exact")
        result["native_decision_ledger"] = audit(fresh)
        result["native_repeat_exact"] = all(
            r["exact"] for r in compare_values(fresh["evaluated_candidates"][0], fresh["evaluated_candidates"][-1])
        )
        require(
            result["native_repeat_exact"]
            and answer["initial_comparison"]["exact"]
            and answer["repeatability"]["exact"],
            "Exact historical/initial repeat",
        )
        n = producer["response_evaluations"]
        qps = [s["qp"] for s in answer["stages"]]
        result.update(
            qp_solves=len(qps),
            nnls_fit_attempts=sum(q["nnls_fit_attempts"] for q in qps),
            exact_native_responses=True,
            total_queries=producer["query_count"] + native["mj_forward_queries"],
        )
        require(
            n == result["verified_responses"] <= 122 and len(qps) <= 60 and result["nnls_fit_attempts"] <= 60,
            "All finite response/solver costs",
        )
        require(producer["query_count"] == native["mj_forward_queries"] == 1 + 312 * n, "Every query counted")
        require(
            producer["numerical_inner_steps"] + native["numerical_inner_steps"] == 600 * n and result["models"] == 2,
            "Every inner/model cost",
        )
        result.update(complete=True, passed=True)
    except BaseException as error:
        result.update(error=repr(error), traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        signal.alarm(0)
        if dynamics is not None:
            producer["actual_model_queries"] = dynamics.queries
        if producer["answer"] is not None:
            retained_qps = [stage["qp"] for stage in producer["answer"]["stages"] if stage["qp"] is not None]
            result["qp_solves"] = len(retained_qps)
            result["nnls_fit_attempts"] = sum(qp["nnls_fit_attempts"] for qp in retained_qps)
        try:
            runtime.context.finish()
            strict_native_environment()
            after = snapshot(REPO)
            result["source_after_ref"] = publish_result(output / "source_after.json", after)
            result["sources_unchanged"] = all(
                runtime.before[k] == after[k] for k in ("sources", "current_map_ref", "driver_ref")
            )
            result["passed"] = result["passed"] and result["sources_unchanged"]
        except BaseException as error:
            result.update(source_error=repr(error), passed=False)
        result["producer_final_ref"] = publish_result(output / "producer_final.json", producer)
        result["independent_counts"] = native
        receipt = publish_result(output / "result.json", result)
        print(
            result_json(dict(arm=arm, result_ref=receipt, complete=result["complete"], passed=result["passed"])),
            flush=True,
        )
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=["baseline", "reserve"], required=True)
    raise SystemExit(main(parser.parse_args().arm))
