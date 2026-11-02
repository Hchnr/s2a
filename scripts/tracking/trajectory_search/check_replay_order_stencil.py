"""Re-evaluate and independently audit the new predictor's complete FD stencil."""

import concurrent.futures
import multiprocessing
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
from s2a.research.trajectory_search.constraint_residual_repair import evaluate_repair_entry
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
from scripts.tracking.trajectory_search.probe_tracking_v5_state_vector_dual_recovery import load_inputs
from scripts.tracking.trajectory_search.probe_tracking_v5_terminal_only_tail_repair import load_environment_guard
from scripts.verification.trajectory_search.check_tracking_v5_saved_direction_backtracking import case_engine
from scripts.verification.trajectory_search.check_tracking_v5_tail_translation_reserve_v2 import semantic_compare

OUTPUT = REPO / "outputs/replay_order_stencil_20261102"
CONFIG = REPO / "configs/tracking/trajectory_search/replay_order_stencil_20261102.json"
PLAN = REPO / "plans/v0.0.1/tracking/trajectory_search/replay_order_stencil_20261102.md"


def compare_values(actual, expected, path=""):
    """Keep original tolerances and exact booleans; record every numeric field."""
    if isinstance(actual, dict):
        require(set(actual) == set(expected), "Same comparison fields: " + path)
        return [row for key in actual for row in compare_values(actual[key], expected[key], path + "/" + key)]
    if isinstance(actual, (list, tuple)):
        require(len(actual) == len(expected), "Same comparison length: " + path)
        return [
            row
            for i, (left, right) in enumerate(zip(actual, expected, strict=True))
            for row in compare_values(left, right, path + "/" + str(i))
        ]
    left, right = np.asarray(actual), np.asarray(expected)
    require(left.shape == right.shape, "Same comparison shape: " + path)
    require(left.dtype.kind in "bifu" and right.dtype.kind in "bifu", "Numeric comparison: " + path)
    difference = float(np.max(np.abs(left.astype(float) - right.astype(float)))) if left.size else 0.0
    exact = bool(np.array_equal(left, right))
    tolerance = 1e-8 if "torque" in path else 1e-10
    passed = exact if left.dtype.kind == "b" else difference <= tolerance
    return [dict(field=path, max_abs_difference=difference, exact=exact, passed=passed)]


def production_batch(rank, commands, source, control, xml_ref, tool):
    """One explicitly billed model and an exclusive directory for each batch."""
    strict_native_environment()
    require(all(os.environ.get(k) == v for k, v in CPU_ENVIRONMENT.items()), "Single-thread CPU worker")
    output = OUTPUT / f"batch_{rank:02d}"
    (output / "tmp").mkdir(parents=True, exist_ok=False)
    os.environ["TMPDIR"] = str(output / "tmp")
    result = dict(
        complete=False,
        passed=False,
        pid=os.getpid(),
        model_attempts=0,
        models=0,
        responses=0,
        inner_steps=0,
        queries=0,
        records=[],
    )
    dynamics = None
    try:
        require(2 <= len(commands) <= 3, "Fixed 32-batch response partition")
        result["model_attempts"] += 1
        dynamics = MujocoRobotDynamics(
            checked(xml_ref),
            control["controller_site"],
            [f"robot0_joint{k}" for k in range(1, 8)],
            source["fixed_gripper_state"],
        )
        result["models"] += 1
        q, velocity = (np.asarray(source[key]) for key in ("current_q", "current_velocity"))
        for key, value in dynamics(q, velocity).items():
            semantic_compare("initial_" + key, value, source["current_quantities"][key], 1e-10)
        interface = OSCInterface(*(control[key] for key in ("input_min", "input_max", "output_min", "output_max")))
        limits, torque = (np.asarray(source[key]) for key in ("joint_limits", "torque_limits"))
        for index, command in commands:
            rollout = rollout_replay_order_trajectory(
                q,
                velocity,
                command,
                interface,
                np.asarray(source["initial_joint"]),
                np.asarray(source["kp"]),
                np.asarray(source["kd"]),
                dynamics,
                torque[:, 0],
                torque[:, 1],
                latched_rotation_goal=np.asarray(source["latched_rotation_goal"]),
            )
            result["responses"] += 1
            result["inner_steps"] += 300
            entry = evaluate_repair_entry(
                rollout,
                command,
                np.asarray(source["previous_command"])[:6],
                interface,
                limits[:, 0],
                limits[:, 1],
                source["desired_tcp"],
                tool,
                source["past_inner_error_rad"],
                source["terminal_normals"],
            )
            ref = publish_result(output / f"response_{index:02d}.json", entry)
            result["records"].append(dict(index=index, ref=ref))
        require(dynamics.queries == 1 + 312 * len(commands), "Every batch query charged")
        result.update(complete=True, passed=True)
    except BaseException as error:
        result.update(error=repr(error), traceback=traceback.format_exc())
    finally:
        if dynamics is not None:
            result["queries"] = dynamics.queries
        strict_native_environment()
        result["result_ref"] = publish_result(output / "result.json", result)
    return result


def main():
    strict_native_environment()
    require(all(os.environ.get(k) == v for k, v in CPU_ENVIRONMENT.items()), "Single-thread CPU parent")
    require(not OUTPUT.exists(), "Exclusive output; no overwrite or automatic retry")
    bundle, _, context = load_inputs()
    refs = context.runtime_refs
    spec = read(reference(CONFIG))
    require(spec["jobs"] == 32 and spec["response_budget"] == 148 and spec["automatic_retry"] is False, "Fixed budget")
    source, source_reg, control = (bundle[key] for key in ("source", "source_registration", "control"))
    model_reg, proof = model_registration(source_reg, context, bundle)
    guard = load_environment_guard(refs)
    control_record = read(remember(source_reg["control_ref"], refs))
    tool_record = read(remember(source_reg["tool_ref"], refs))
    tool = np.linalg.solve(np.asarray(control_record["source_from_site"]), np.asarray(tool_record["source_from_tcp"]))
    commands = [np.asarray(bundle["base"]["commands"])] * 2 + [np.asarray(p["commands"]) for p in bundle["probes"]]
    require(len(commands) == 74, "Base, repeat, complete72 probes")
    nominal = np.asarray(bundle["nominal_deltas"])
    build_response_stencil(bundle["base"], bundle["probes"], nominal)  # Validate commands before any new query.
    prefix_check = read(remember(PREFIX_CHECK_REF, refs))
    prefix_reg = read(remember(prefix_check["registration_ref"], refs))
    prerequisite = reference(REPO / "outputs/native_replay_response_contract_v4_20261102/result.json")
    require(prerequisite["sha256"] == "9a651d11bdc90da6c4eb396f84a5679c9ce588ce2f6e0c8b145e488ab9442c7c", "Pinned v4")
    require(read(prerequisite)["all_values_exact"] is True, "Exact finite response prerequisite")
    local_paths = [
        Path(__file__),
        CONFIG,
        PLAN,
        REPO / "src/s2a/research/trajectory_search/response_stencil.py",
        REPO / "src/s2a/research/trajectory_search/replay_order_trajectory.py",
        REPO / "src/s2a/research/response_models/forward/native_replay_joint_response.py",
        REPO / "src/s2a/research/response_models/forward/native_osc_torque.py",
    ]
    before = [reference(path) for path in local_paths]
    context.finish()
    registration = publish_result(
        OUTPUT / "registration.json",
        dict(
            experiment="replay_order_stencil",
            config=spec,
            config_ref=reference(CONFIG),
            plan_ref=reference(PLAN),
            local_sources=before,
            input_refs=list(refs.values()),
            asset_proof=proof,
            prerequisite_ref=prerequisite,
            commands=commands,
            nominal_deltas=nominal,
            tool=tool,
            tool_difference_from_historical=float(np.max(abs(tool - np.asarray(source["site_from_tcp"])))),
        ),
    )
    result = dict(
        complete=False,
        passed=False,
        registration_ref=registration,
        batches=[],
        processes=[],
        comparisons=[],
        mj_forward_queries=0,
        numerical_inner_steps=0,
        max_jit_vs_python_goal_difference=0.0,
        jit_python_goal_differences=0,
        new_responses=0,
        new_queries=0,
        new_inner_steps=0,
        independent_responses=0,
        model_attempts=0,
        models=0,
        physics_steps=0,
        world_model_calls=0,
        qp_solves=0,
        nnls_fits=0,
        fresh_cases=0,
        control_qualified=False,
        automatic_retry=False,
    )
    started = time.monotonic()
    try:
        result["environment_admission"] = restore_cv2_environment(
            guard["exact_probe_ref"], guard["runtime_source_refs"]
        )
        os.environ.pop("MUJOCO_GL", None)
        os.environ.pop("PYOPENGL_PLATFORM", None)
        result["model_attempts"] += 1
        reconstruct, _, classify, _, _, _ = case_engine(
            "prefix",
            source,
            model_reg,
            prefix_reg,
            result,
            refs,
            lambda name, a, b, tolerance=1e-10: semantic_compare(name, a, b, tolerance),
            input_context=context,
        )
        result["models"] += 1
        batches = [list(enumerate(commands))[rank::32] for rank in range(32)]
        executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=32, mp_context=multiprocessing.get_context("spawn")
        )
        handles = []
        try:
            futures = [
                executor.submit(production_batch, rank, batch, source, control, model_reg["xml_ref"], tool)
                for rank, batch in enumerate(batches)
            ]
            handles = list(executor._processes.values())
            for future in concurrent.futures.as_completed(futures):
                try:
                    batch = future.result()
                except BaseException as error:
                    result.setdefault("worker_errors", []).append(dict(error=repr(error), costs_unknown=True))
                    continue
                result["batches"].append(batch)
                for target, key in [
                    ("new_responses", "responses"),
                    ("new_queries", "queries"),
                    ("new_inner_steps", "inner_steps"),
                    ("model_attempts", "model_attempts"),
                    ("models", "models"),
                ]:
                    result[target] += batch[key]
        finally:
            executor.shutdown(wait=True)
            for process in handles:
                process.join()
                result["processes"].append(
                    dict(pid=process.pid, exitcode=process.exitcode, actual_wait=True, reaped=True)
                )
        require(all(b["passed"] for b in result["batches"]) and len(result["batches"]) == 32, "Every batch passed")
        require(all(p["exitcode"] == 0 for p in result["processes"]), "All CPU workers exited normally")
        records = sorted([r for b in result["batches"] for r in b["records"]], key=lambda row: row["index"])
        require([r["index"] for r in records] == list(range(74)), "Every request exactly once")
        new_entries, independent_entries = [], []
        for index, record in enumerate(records):
            entry = read(record["ref"])
            native, _ = reconstruct(commands[index])
            result["independent_responses"] += 1
            metric = classify(commands[index], native)
            independent = dict(
                commands=commands[index],
                rollout=native,
                safety_slack=metric["safety_slack"],
                tcp_errors=metric["tcp_errors"],
                normalized_squared_errors=metric["normalized_squared_errors"],
                feasible=metric["feasible"],
                prefix_length=metric["prefix_length"],
                maximum_normalized_squared_error=metric["maximum_normalized_squared_error"],
                maximum_pose_cap_violation=metric["maximum_pose_cap_violation"],
                pose_loss=metric["pose_loss"],
            )
            comparable = {key: entry[key] for key in independent}
            comparisons = compare_values(comparable, independent)
            comparisons += compare_values(entry["audit"]["feasible"], metric["joint_feasible"], "/joint_feasible")
            result["comparisons"].append(dict(index=index, fields=comparisons))
            publish_result(
                OUTPUT / "independent" / f"response_{index:02d}.json", dict(entry=independent, metrics=metric)
            )
            new_entries.append(entry)
            independent_entries.append(independent)
            if index % 12 == 0:
                print(
                    result_json(
                        dict(verified=index + 1, max_difference=max(r["max_abs_difference"] for r in comparisons))
                    ),
                    flush=True,
                )
        fresh = build_response_stencil(new_entries[0], new_entries[2:], nominal)
        independent_model = build_response_stencil(independent_entries[0], independent_entries[2:], nominal)
        result["model_comparison"] = compare_values(fresh, independent_model)
        result["repeat_comparison"] = compare_values(new_entries[0], new_entries[1])
        result["independent_repeat_comparison"] = compare_values(independent_entries[0], independent_entries[1])
        result["stencil_ref"] = publish_result(
            OUTPUT / "stencil.json",
            dict(
                base=new_entries[0],
                probes=[
                    {k: e[k] for k in ("commands", "safety_slack", "tcp_errors", "normalized_squared_errors")}
                    for e in new_entries[2:]
                ],
                nominal_deltas=nominal,
                matrices=fresh,
            ),
        )
        rows = [r for c in result["comparisons"] for r in c["fields"]]
        rows += result["model_comparison"] + result["repeat_comparison"] + result["independent_repeat_comparison"]
        result["all_values_exact"] = all(r["exact"] for r in rows)
        result["total_queries"] = result["new_queries"] + result["mj_forward_queries"]
        require(result["new_responses"] == result["independent_responses"] == 74, "All response costs")
        require(
            result["new_inner_steps"] + result["numerical_inner_steps"] == spec["inner_step_budget"], "All inner costs"
        )
        require(
            result["total_queries"] == spec["query_budget"] and result["models"] == spec["model_budget"],
            "All model/query costs",
        )
        result.update(complete=True, passed=all(r["passed"] for r in rows) and result["all_values_exact"])
    except BaseException as error:
        result.update(error=repr(error), traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        try:
            context.finish()
            strict_native_environment()
            result["sources_unchanged"] = before == [reference(path) for path in local_paths]
            result["passed"] = result["passed"] and result["sources_unchanged"]
        except BaseException as error:
            result["source_validation_error"] = repr(error)
            result["passed"] = False
        result["elapsed_seconds"] = time.monotonic() - started
        result_ref = publish_result(OUTPUT / "result.json", result)
        print(
            result_json(dict(result_ref=result_ref, complete=result["complete"], passed=result["passed"])), flush=True
        )
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
