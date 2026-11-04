"""Bounded nonlinear restoration, with every response independently replayed."""

import os
import signal
import traceback

import numpy as np

from s2a.controllers.osc import OSCInterface
from s2a.dynamics.mujoco_robot import MujocoRobotDynamics
from s2a.infrastructure.native_environment import restore_cv2_environment, strict_native_environment
from s2a.infrastructure.publication import publish_result
from s2a.research.trajectory_search.constraint_residual_repair import evaluate_repair_entry
from s2a.research.trajectory_search.nonlinear_feasibility_projection import project
from s2a.research.trajectory_search.replay_order_trajectory import rollout_replay_order_trajectory
from s2a.research.trajectory_search.restored_anchor_sqp_evidence import REPO, checked, read, reference, require
from scripts.maintenance.run_cpu_tests import snapshot
from scripts.tracking.trajectory_search.check_cap_distance_gn import phase_deadline
from scripts.tracking.trajectory_search.check_replay_order_stencil import compare_values
from scripts.tracking.trajectory_search.check_reserve_entry_search import native_metric_entry
from scripts.tracking.trajectory_search.nonlinear_projection_audit import audit_projection
from scripts.tracking.trajectory_search.response_probe_runtime import ResponseProbeRuntime
from scripts.verification.trajectory_search.check_tracking_v5_prefix_trust_control import check_full_candidate
from scripts.verification.trajectory_search.check_tracking_v5_saved_direction_backtracking import case_engine
from scripts.verification.trajectory_search.check_tracking_v5_tail_translation_reserve_v2 import semantic_compare

OUTPUT = REPO / "outputs/nonlinear_feasibility_projection_20261102"
CONFIG = REPO / "configs/tracking/trajectory_search/nonlinear_feasibility_projection_20261102.json"
PLAN = REPO / "plans/v0.0.1/tracking/trajectory_search/nonlinear_feasibility_projection_design_20261102.md"
QA = REPO / "outputs/qa/nonlinear_feasibility_projection_v2_20261102/result.json"


def main():
    phase_deadline(1800)
    spec = read(reference(CONFIG))
    require(
        spec
        == dict(
            seed=42,
            max_evaluations=320,
            max_iterations=4,
            maximum_responses_per_side=321,
            total_response_budget=642,
            total_inner_step_budget=192600,
            total_query_budget=200306,
            model_budget=2,
            timeout_seconds=1800,
            automatic_retry=False,
        ),
        "Exact restoration budget",
    )
    runtime = ResponseProbeRuntime(OUTPUT, QA)
    prior = runtime.load(
        reference(REPO / "outputs/local_support_direction_probe_20261102/result.json"),
        "a176f9e259e4dffe37cc09e193fd57a3974bc1daa0a08fa17299f163e896e10f",
    )
    require(
        prior["passed"] and prior["sources_unchanged"] and prior["exact_native_responses"], "Qualified unsafe source"
    )
    source = runtime.load(
        reference(REPO / "outputs/local_support_direction_probe_20261102/production/response_01.json"),
        "2f7a47e948345c9700b1115888e4cd4777f81fe090a68374b0b4e839a9698621",
    )
    require(not source["audit"]["feasible"], "Exact registered unsafe warm")
    s, c = runtime.source, runtime.control
    previous = np.asarray(s["previous_command"])[:6]
    interface = OSCInterface(*(c[k] for k in ["input_min", "input_max", "output_min", "output_max"]))
    registration = publish_result(
        OUTPUT / "registration.json",
        dict(
            spec=spec,
            config_ref=reference(CONFIG),
            plan_ref=reference(PLAN),
            qa_ref=runtime.qa_ref,
            input_refs=list(runtime.refs.values()),
            source_before_ref=publish_result(OUTPUT / "source_before.json", runtime.before),
        ),
    )
    result = dict(
        complete=False,
        passed=False,
        pid=os.getpid(),
        registration_ref=registration,
        models=0,
        model_attempts=0,
        prediction_calls=0,
        response_evaluations=0,
        verified_responses=0,
        numerical_inner_steps=0,
        physics_steps=0,
        world_model_calls=0,
        fresh_cases=0,
        actual_consumed_reference_count=0,
        control_qualified=False,
        automatic_retry=False,
    )
    native = dict(
        mj_forward_queries=0,
        numerical_inner_steps=0,
        max_jit_vs_python_goal_difference=0.0,
        jit_python_goal_differences=0,
    )
    entries, ledger = [], {}
    dynamics = None
    try:
        runtime.context.finish()
        result["environment_admission"] = restore_cv2_environment(
            runtime.guard["exact_probe_ref"], runtime.guard["runtime_source_refs"]
        )
        os.environ.pop("MUJOCO_GL", None)
        os.environ.pop("PYOPENGL_PLATFORM", None)
        result["model_attempts"] += 1
        dynamics = MujocoRobotDynamics(
            checked(runtime.model_reg["xml_ref"]),
            c["controller_site"],
            [f"robot0_joint{k}" for k in range(1, 8)],
            s["fixed_gripper_state"],
        )
        result["models"] += 1
        q, v = (np.asarray(s[k]) for k in ["current_q", "current_velocity"])
        dynamics(q, v)
        limits, torque = (np.asarray(s[k]) for k in ["joint_limits", "torque_limits"])

        def evaluate(commands):
            require(result["prediction_calls"] < 321, "Full registered producer budget")
            result["prediction_calls"] += 1
            rollout = rollout_replay_order_trajectory(
                q,
                v,
                commands,
                interface,
                np.asarray(s["initial_joint"]),
                np.asarray(s["kp"]),
                np.asarray(s["kd"]),
                dynamics,
                torque[:, 0],
                torque[:, 1],
                latched_rotation_goal=np.asarray(s["latched_rotation_goal"]),
            )
            result["numerical_inner_steps"] += 300
            entry = evaluate_repair_entry(
                rollout,
                commands,
                previous,
                interface,
                limits[:, 0],
                limits[:, 1],
                s["desired_tcp"],
                s["site_from_tcp"],
                s["past_inner_error_rad"],
                s["terminal_normals"],
            )
            publish_result(OUTPUT / "production" / f"response_{len(entries):03d}.json", entry)
            entries.append(entry)
            result["response_evaluations"] += 1
            if len(entries) == 1 or len(entries) % 40 == 0:
                print(dict(production=len(entries), maximum=321), flush=True)
            return entry

        project(evaluate, source["commands"], previous, c["input_min"], c["input_max"], ledger=ledger)
        result["production_audit"] = audit_projection(ledger, entries, previous, c["input_min"], c["input_max"])
        result["projection_ref"] = publish_result(OUTPUT / "projection.json", ledger)
        evaluate(np.asarray(ledger["used_warm"]))
        require(all(row["exact"] for row in compare_values(entries[0], entries[-1])), "Exact used warm repeat")
        result["model_attempts"] += 1
        reconstruct, compare_rollout, classify, check_audit, _, _ = case_engine(
            "prefix",
            s,
            runtime.model_reg,
            runtime.prefix_reg,
            native,
            runtime.refs,
            lambda n, a, b, tolerance=1e-10: semantic_compare(n, a, b, tolerance),
            input_context=runtime.context,
        )
        result["models"] += 1
        fresh = []
        for i, entry in enumerate(entries):
            commands = np.asarray(entry["commands"])
            rollout, _ = reconstruct(commands)
            metric = classify(commands, rollout)
            check_full_candidate(
                entry,
                commands,
                rollout,
                metric,
                compare_rollout,
                check_audit,
                compare=lambda n, a, b, tolerance=1e-10: semantic_compare(n, a, b, tolerance),
            )
            require(
                all(row["exact"] for row in compare_values(entry["rollout"], rollout)),
                "Exact independent full response",
            )
            fresh.append(native_metric_entry(entry, rollout, metric))
            publish_result(OUTPUT / "independent" / f"response_{i:03d}.json", dict(rollout=rollout, metrics=metric))
            result["verified_responses"] += 1
            if i % 40 == 0:
                print(dict(verified=i + 1, total=len(entries)), flush=True)
        result["independent_audit"] = audit_projection(ledger, fresh[:-1], previous, c["input_min"], c["input_max"])
        count = len(entries)
        require(
            result["prediction_calls"] == count <= 321
            and dynamics.queries == native["mj_forward_queries"] == 1 + 312 * count,
            "All queries and producer calls charged",
        )
        require(
            native["numerical_inner_steps"] == result["numerical_inner_steps"] == 300 * count, "Complete inner steps"
        )
        result.update(
            complete=True,
            passed=True,
            exact_native_responses=True,
            repeat_exact=True,
            total_responses=2 * count,
            total_queries=2 + 624 * count,
            total_inner_steps=600 * count,
        )
    except BaseException as error:
        result.update(error=repr(error), traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        signal.alarm(0)
        result["independent_counts"] = native
        result["producer_ledger_final_ref"] = publish_result(
            OUTPUT / "producer_final.json", dict(ledger=ledger, complete=result["complete"], responses=len(entries))
        )
        if dynamics is not None:
            result["actual_model_queries"] = dynamics.queries
        try:
            runtime.context.finish()
            strict_native_environment()
            after = snapshot(REPO)
            result["source_after_ref"] = publish_result(OUTPUT / "source_after.json", after)
            result["sources_unchanged"] = all(
                runtime.before[k] == after[k] for k in ["sources", "current_map_ref", "driver_ref"]
            )
            result["passed"] &= result["sources_unchanged"]
        except BaseException as error:
            result.update(source_error=repr(error), passed=False)
        print(publish_result(OUTPUT / "result.json", result), flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
