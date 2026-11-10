"""Matched, journaled H12 refinements with complete independent native replay."""

import argparse
import hashlib
import json
import os
import signal
import tempfile
import traceback

import numpy as np

from s2a.controllers.osc import OSCInterface
from s2a.dynamics.mujoco_robot import MujocoRobotDynamics
from s2a.infrastructure.native_environment import restore_cv2_environment, strict_native_environment
from s2a.infrastructure.publication import publish_result
from s2a.research.trajectory_search.constraint_residual_repair import evaluate_repair_entry
from s2a.research.trajectory_search.replay_order_trajectory import rollout_replay_order_trajectory
from s2a.research.trajectory_search.restored_anchor_sqp_evidence import REPO, checked, read, reference, require
from s2a.research.trajectory_search.whole_trajectory_refinement import common_start, refine
from scripts.maintenance.run_cpu_tests import snapshot
from scripts.tracking.trajectory_search.check_cap_distance_gn import phase_deadline
from scripts.tracking.trajectory_search.check_replay_order_stencil import compare_values
from scripts.tracking.trajectory_search.check_reserve_entry_search import native_metric_entry
from scripts.tracking.trajectory_search.response_probe_runtime import ResponseProbeRuntime
from scripts.tracking.trajectory_search.whole_trajectory_refinement_audit import audit_refinement
from scripts.verification.trajectory_search.check_tracking_v5_prefix_trust_control import check_full_candidate
from scripts.verification.trajectory_search.check_tracking_v5_saved_direction_backtracking import case_engine
from scripts.verification.trajectory_search.check_tracking_v5_tail_translation_reserve_v2 import semantic_compare

ROOT = REPO / "outputs/whole_trajectory_refinement_20261110"
CONFIG = REPO / "configs/tracking/trajectory_search/whole_trajectory_refinement_20261110.json"
PLAN = REPO / "plans/v0.0.1/tracking/trajectory_search/whole_trajectory_refinement_20261110.md"
QA = REPO / "outputs/qa/whole_trajectory_refinement_20261110/result.json"


def arm_bounds(arm, previous, input_min, input_max, root_lower, root_upper):
    require(arm in ("root", "full"), "Only the two registered domains")
    lo, hi = np.tile(np.asarray(input_min), (12, 1)), np.tile(np.asarray(input_max), (12, 1))
    lo[0], hi[0] = np.maximum(lo[0], previous - 0.25), np.minimum(hi[0], previous + 0.25)
    if arm == "root":
        lo, hi = np.maximum(lo, np.asarray(root_lower)), np.minimum(hi, np.asarray(root_upper))
    require(
        lo.shape == hi.shape == (12, 6) and np.isfinite(lo).all() and np.isfinite(hi).all() and np.all(lo < hi),
        "Finite ordered arm domain",
    )
    return lo, hi


def baseline_comparison(actual, historical):
    """Retain original numeric/boolean tolerances for the tiny common box repair."""
    rows = []
    for key in ("rollout", "tcp_errors", "normalized_squared_errors", "safety_slack"):
        rows.extend(compare_values(actual[key], historical[key], "/" + key))
    require(all(r["passed"] for r in rows), "Historical full-response numeric tolerances")
    require(
        actual["audit"]["feasible"] == historical["audit"]["feasible"]
        and actual["prefix_length"] == historical["prefix_length"]
        and actual["feasible"] == historical["feasible"],
        "Exact historical feasibility and prefix labels",
    )
    np.testing.assert_array_equal(
        np.asarray(actual["normalized_squared_errors"]) <= 1, np.asarray(historical["normalized_squared_errors"]) <= 1
    )
    return dict(
        passed=True,
        exact=all(r["exact"] for r in rows),
        max_abs_difference=max((r["max_abs_difference"] for r in rows), default=0.0),
    )


def main(arm):
    phase_deadline(1800)
    spec = read(reference(CONFIG))
    require(
        spec
        == dict(
            seed=42,
            arms=["root", "full"],
            max_evaluations_per_arm=640,
            max_iterations_per_arm=8,
            maximum_responses_per_side=641,
            total_response_budget_per_arm=1282,
            total_inner_step_budget_per_arm=384600,
            total_query_budget_per_arm=399986,
            model_budget_per_arm=2,
            timeout_seconds=1800,
            automatic_retry=False,
        ),
        "Exact matched budgets",
    )
    require(arm in spec["arms"], "Registered arm")
    output = ROOT / arm
    worker_tmp = REPO / "outputs/whole_trajectory_refinement_20261110_worker_tmp" / arm
    worker_tmp.mkdir(parents=True, exist_ok=False)
    os.environ["TMPDIR"] = str(worker_tmp)
    tempfile.tempdir = str(worker_tmp)
    runtime = ResponseProbeRuntime(output, QA)
    completed = runtime.load(
        reference(REPO / "outputs/interrupted_projection_candidate_replay_20261110/result.json"),
        "70968a498c1f7488368c44f01c12252ea124a2cd9a1cfd3b6db568676a1612e9",
    )
    require(
        completed["passed"]
        and completed["independently_safe_candidates"] == 13
        and completed["independently_qualified_witnesses"] == 0,
        "Previous numeric evidence remains closed",
    )
    prior = runtime.load(
        reference(REPO / "outputs/small_progress_search_20261102/small_progress/result.json"),
        "844c4e05a24b41dda33e36aac180bb421afa73e63f88f8eb70b213615f6e4c5a",
    )
    repaired = runtime.load(
        reference(REPO / "outputs/small_progress_native_repair_20261102/result.json"),
        "52a41511b116e958a3bb7ba43adbfaf1485eb351be1e88a5349e4a79eae0ac90",
    )
    require(repaired["passed"] and repaired["verified_responses"] == 36, "Independently verified historical anchor")
    answer = runtime.load(prior["producer_final_ref"])["answer"]
    require(answer["final_anchor_index"] == 23, "Exact shared anchor")
    historical = answer["evaluated_candidates"][23]
    s, c = runtime.source, runtime.control
    previous = np.asarray(s["previous_command"])[:6]
    start = common_start(
        historical["commands"], previous, c["input_min"], c["input_max"], answer["global_lower"], answer["global_upper"]
    )
    lo, hi = arm_bounds(arm, previous, c["input_min"], c["input_max"], answer["global_lower"], answer["global_upper"])
    require(np.all(start >= lo) and np.all(start <= hi), "Same exact warm in each domain")
    runtime.context.finish()
    registration = publish_result(
        output / "registration.json",
        dict(
            arm=arm,
            spec=spec,
            config_ref=reference(CONFIG),
            plan_ref=reference(PLAN),
            qa_ref=runtime.qa_ref,
            input_refs=list(runtime.refs.values()),
            raw_warm=historical["commands"],
            common_warm=start,
            warm_delta=start - np.asarray(historical["commands"]),
            lower=lo,
            upper=hi,
            source_before_ref=publish_result(output / "source_before.json", runtime.before),
        ),
    )
    result = dict(
        complete=False,
        passed=False,
        arm=arm,
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
    entries, ledger, events = [], {}, []
    dynamics = None
    journal = (output / "events.jsonl").open("x", buffering=1)

    def emit(event):
        event = dict(sequence=len(events), **event)
        journal.write(json.dumps(event, allow_nan=False, separators=(",", ":")) + "\n")
        journal.flush()
        events.append(event)

    def terminate(signum, frame):
        raise TimeoutError("Owner sent termination signal; completed journal prefix is preserved")

    signal.signal(signal.SIGTERM, terminate)
    try:
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
        interface = OSCInterface(*(c[k] for k in ["input_min", "input_max", "output_min", "output_max"]))
        limits, torque = (np.asarray(s[k]) for k in ["joint_limits", "torque_limits"])

        def evaluate(commands):
            require(result["prediction_calls"] < 641, "Registered producer limit")
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
            publish_result(output / "production" / f"response_{len(entries):03d}.json", entry)
            entries.append(entry)
            result["response_evaluations"] += 1
            if len(entries) == 1:
                result["historical_baseline_comparison"] = baseline_comparison(entry, historical)
            if len(entries) == 1 or len(entries) % 40 == 0:
                print(dict(arm=arm, production=len(entries), maximum=641), flush=True)
            return entry

        refine(evaluate, start, previous, lo, hi, ledger=ledger, event_sink=emit)
        result["production_audit"] = audit_refinement(ledger, entries, previous, lo, hi, events)
        result["ledger_ref"] = publish_result(output / "ledger.json", ledger)
        identity = hashlib.sha256(start.ravel().tobytes()).hexdigest()
        emit(dict(kind="repeat_request", command_sha256=identity))
        evaluate(start)
        emit(dict(kind="repeat_completed", command_sha256=identity))
        require(all(row["exact"] for row in compare_values(entries[0], entries[-1])), "Exact production warm repeat")
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
                all(row["exact"] for row in compare_values(entry["rollout"], rollout)), "Exact independent response"
            )
            fresh.append(native_metric_entry(entry, rollout, metric))
            publish_result(output / "independent" / f"response_{i:03d}.json", dict(rollout=rollout, metrics=metric))
            result["verified_responses"] += 1
            if i % 40 == 0:
                print(dict(arm=arm, verified=i + 1, total=len(entries)), flush=True)
        result["independent_audit"] = audit_refinement(ledger, fresh[:-1], previous, lo, hi, events)
        count = len(entries)
        require(
            result["prediction_calls"] == count <= 641
            and dynamics.queries == native["mj_forward_queries"] == 1 + 312 * count,
            "All full responses and queries charged",
        )
        require(
            result["numerical_inner_steps"] == native["numerical_inner_steps"] == 300 * count and result["models"] == 2,
            "Exact model and numerical inner-step accounting",
        )
        result.update(
            complete=True,
            passed=True,
            exact_native_responses=True,
            repeat_exact=True,
            total_responses=2 * count,
            total_queries=2 + 624 * count,
            total_inner_steps=600 * count,
            selected_index=ledger["selected_index"],
            witness_indices=ledger["witness_indices"],
        )
    except BaseException as error:
        result.update(error=repr(error), traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        signal.alarm(0)
        journal.close()
        result["journal_ref"] = reference(output / "events.jsonl")
        result["independent_counts"] = native
        result["producer_final_ref"] = publish_result(
            output / "producer_final.json", dict(ledger=ledger, complete=result["complete"], responses=len(entries))
        )
        if dynamics is not None:
            result["actual_model_queries"] = dynamics.queries
        try:
            runtime.context.finish()
            strict_native_environment()
            after = snapshot(REPO)
            result["source_after_ref"] = publish_result(output / "source_after.json", after)
            result["sources_unchanged"] = all(
                runtime.before[k] == after[k] for k in ("sources", "current_map_ref", "driver_ref")
            )
            result["passed"] &= result["sources_unchanged"]
        except BaseException as error:
            result.update(source_error=repr(error), passed=False)
        print(publish_result(output / "result.json", result), flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=("root", "full"))
    raise SystemExit(main(parser.parse_args().arm))
