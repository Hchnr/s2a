"""Reconstruct, but do not dispatch, the original early-warning refiner decision."""

import argparse
import json
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from s2a.geometry import inverse
from s2a.infrastructure.native_environment import restore_cv2_environment
from s2a.infrastructure.publication import publish_result as publish
from s2a.research.joint_control.preview.completed_history import (
    CompletedHistory,
    recompute_completed_innovation,
    restore_completed_history,
)
from s2a.research.joint_control.preview.dependency_safe_preview_runtime import solve_preview
from s2a.research.native_runtime.development_continuation import current_joint_site_state
from scripts.maintenance.run_cpu_tests import snapshot
from scripts.simulation.runtime import compare_early_warning_tracking as early
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix

REPO = prefix.REPO
PLAN = REPO / "plans/v0.0.1/tracking/joint_control/early_refiner_reconstruction_20261110.md"
CORE = REPO / "src/s2a/research/joint_control/preview/completed_history.py"
CORE_TEST = REPO / "tests/research/joint_control/preview/test_completed_history.py"
INPUT_TEST = REPO / "tests/research/native_runtime/test_early_refiner_history.py"
QA = REPO / "outputs/qa/early_refiner_reconstruction_20261110/result.json"
DRY = REPO / "outputs/early_refiner_reconstruction_dry_20261110/result.json"
OUTPUT = REPO / "outputs/early_refiner_reconstruction_20261110"
ATOL = 1e-8


def source_refs():
    return [
        prefix.reference(p)
        for p in (
            Path(__file__),
            PLAN,
            CORE,
            CORE_TEST,
            INPUT_TEST,
            Path(early.__file__),
            early.PLAN,
            early.CONFIG,
            Path(prefix.__file__),
            prefix.CORE,
            prefix.RUNTIME,
            REPO / "src/s2a/research/joint_control/preview/tracking_preview_transaction.py",
            REPO / "src/s2a/research/joint_control/native/tracking_margin_native.py",
            REPO / "src/s2a/research/joint_control/preview/dependency_safe_preview_runtime.py",
        )
    ]


def completed_from_prefix(rows, commands):
    """Extract a narrow snapshot from the last previously completed period."""
    n = len(commands)
    prefix.require(n > 0 and len(rows) == n and rows[-1]["step"] == n - 1, "Exactly one completed prefix")
    row = rows[-1]
    prefix.require(np.array_equal(np.asarray(row["command"]), commands[-1]), "Recorded previous command mismatch")
    coupled = row["coupled_response_history"]
    prefix.require(coupled["completed_periods_after"] == n, "Coupled completed periods mismatch")
    return CompletedHistory(
        previous_command=row["command"],
        coupled_innovation=coupled["innovation_after"],
        tracking_innovation=row["tracking_history_after"],
        last_plan=row["tracking_refinement"]["planned_commands"],
        completed_periods=n,
    )


def load_inputs():
    inputs = early.load_inputs()
    for case in inputs["cases"]:
        path = case["physical_folder"] / "steps.jsonl"
        ref = prefix.reference(path)
        prefix.require(inputs["source_sha256"].get(ref["path"]) == ref["sha256"], "Reviewed completed log")
        n = case["spec"]["prefix_steps"]
        rows = prefix.read_recorded_prefix(path, n, 600)
        case["completed_history"] = completed_from_prefix(rows, case["commands"])
        previous_row = rows[-1]
        case["previous_evidence"] = {
            "site": previous_row["qualified_forward_input"]["x_site_pose"][0],
            "twist": previous_row["qualified_forward_input"]["x_site_twist"][0],
            "coupled_prediction": previous_row["coupled_response_history"]["commit"]["nominal_response"],
            "tracking_prediction": previous_row["tracking_refinement"]["nominal_response"],
        }
    return inputs


def differences(actual, expected):
    actual, expected = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    prefix.require(
        actual.shape == expected.shape and np.isfinite(actual).all() and np.isfinite(expected).all(),
        "Comparable finite arrays",
    )
    return float(np.max(np.abs(actual - expected)))


def compare_original(case, measured, seed, refinement, coupled, tracking):
    """Read the original current decision only after the new calculation has finished."""
    n = case["spec"]["prefix_steps"]
    row = prefix.read_recorded_prefix(case["physical_folder"] / "steps.jsonl", n + 1, 600)[-1]
    dynamics = row["coupled_dynamics"]
    old_seed = row["pose_preview_seed_inverse"]
    old_refined = row["tracking_refinement"]
    old_current = row["qualified_forward_input"]
    delta = {
        "mass": differences(measured["mass"], dynamics["mass"]),
        "jacobian": differences(measured["jacobian"], dynamics["jacobian"]),
        "q": differences(measured["q"], dynamics["robot_joint_position"]),
        "qvel": differences(measured["qvel"], dynamics["robot_joint_velocity"]),
        "site": differences(measured["site"], old_current["x_site_pose"][0]),
        "twist": differences(measured["twist"], old_current["x_site_twist"][0]),
        "coupled_innovation": differences(coupled, row["coupled_response_history"]["innovation_before"]),
        "tracking_innovation": differences(tracking, old_refined["history_before"]),
        "seed_command": differences(seed["command"], old_seed["command"]),
        "seed_plan": differences(seed["preview"]["planned_commands"], old_seed["preview"]["planned_commands"]),
        "refined_command": differences(refinement["command"], row["command"][:6]),
        "refined_plan": differences(refinement["planned_commands"], old_refined["planned_commands"]),
    }
    return {
        "max_absolute_difference": delta,
        "atol": ATOL,
        "rtol": 0,
        "same_route": refinement["route"] == old_refined["route"],
        "original_route": old_refined["route"],
        "all_within_tolerance": all(value <= ATOL for value in delta.values()),
        "original_comparison_read_after_computation": True,
    }


def numerical_reconstruction(env, case, receipt):
    """Prepare one action, forbid dispatch, and retain the actual model cost."""
    sim, robot = env.sim, env.robots[0]
    before = np.asarray(sim.get_state().flatten()).copy()
    controls, sensors = np.asarray(sim.data.ctrl).copy(), np.asarray(sim.data.sensordata).copy()
    goal = np.asarray(robot.controller.goal_ori).copy()
    original_step = env.step

    def no_dispatch(*_args, **_kwargs):
        raise RuntimeError("Refiner reconstruction forbids dispatch")

    env.step = no_dispatch
    try:
        q, limits, jacobian = current_joint_site_state(env)
        mass, qvel, site = early.current_dynamics(env, jacobian)
        twist = jacobian @ qvel
        historical = case["completed_history"]
        evidence = case["previous_evidence"]
        coupled_rebuilt = recompute_completed_innovation(
            evidence["site"], evidence["twist"], evidence["coupled_prediction"], site, twist
        )
        tracking_rebuilt = recompute_completed_innovation(
            evidence["site"], evidence["twist"], evidence["tracking_prediction"], site, twist
        )
        innovation_diff = {
            "coupled": differences(coupled_rebuilt, historical.coupled_innovation),
            "tracking": differences(tracking_rebuilt, historical.tracking_innovation),
        }
        prefix.require(all(value <= ATOL for value in innovation_diff.values()), "Prior innovations must recompute")
        rebuilt = CompletedHistory(
            historical.previous_command,
            coupled_rebuilt,
            tracking_rebuilt,
            historical.last_plan,
            historical.completed_periods,
        )
        coupled, tracking = restore_completed_history(rebuilt, site, twist)
        coupled_before = coupled.current(site, twist)
        tracking_before = tracking.history.current(site, twist)
        source_from_site = np.asarray(case["controller"]["source_from_site"], dtype=float)
        site_from_tcp = inverse(source_from_site) @ case["tool"]
        targets, anchor = case["targets"] @ inverse(site_from_tcp), case["anchor"] @ inverse(site_from_tcp)
        cursor = case["intervention_spec"]["cursor"]
        contract = case["controller"]["contract"]
        receipt["preview_calls"] = {"attempted": 1, "completed": 0}
        started = time.monotonic()
        seed = solve_preview(
            mass,
            jacobian,
            site,
            twist,
            targets,
            anchor,
            cursor,
            historical.previous_command[:6],
            contract,
            innovation=coupled_before,
        )
        receipt["preview_calls"]["completed"] = 1
        dynamics = {
            "mass": mass,
            "jacobian": jacobian,
            "robot_joint_position": q,
            "robot_joint_velocity": qvel,
            "kp": np.asarray(robot.controller.kp).copy(),
            "kd": np.asarray(robot.controller.kd).copy(),
        }
        receipt["refiner_calls"] = {"attempted": 1, "completed": 0}
        refinement = tracking.prepare(
            env,
            dynamics,
            site,
            twist,
            targets,
            anchor,
            cursor,
            seed["preview"]["planned_commands"],
            historical.previous_command[:6],
            contract,
            site_from_tcp=site_from_tcp,
        )
        receipt["refiner_calls"]["completed"] = 1
        prefix.require(refinement["evaluations"] <= 480, "Original per-case optimization budget")
        prefix.require(
            tracking.phase == "prepared" and tracking.history.pending is None,
            "Reconstructed decision was not committed or observed",
        )
        np.testing.assert_array_equal(before, sim.get_state().flatten())
        np.testing.assert_array_equal(controls, np.asarray(sim.data.ctrl))
        np.testing.assert_array_equal(sensors, np.asarray(sim.data.sensordata))
        np.testing.assert_array_equal(goal, robot.controller.goal_ori)
    finally:
        env.step = original_step
    measured = {"mass": mass, "jacobian": jacobian, "q": q, "qvel": qvel, "site": site, "twist": twist}
    comparison = compare_original(case, measured, seed, refinement, coupled_before, tracking_before)
    return {
        "preview_command": seed["command"],
        "preview_plan": seed["preview"]["planned_commands"],
        "refined_command": refinement["command"],
        "refined_plan": refinement["planned_commands"],
        "route": refinement["route"],
        "evaluations": refinement["evaluations"],
        "optimization": refinement["optimization"],
        "comparison": comparison,
        "completed_innovation_recomputation_max_difference": innovation_diff,
        "elapsed_seconds": time.monotonic() - started,
        "new_actions_dispatched": 0,
        "actual_state_controls_sensors_goal_unchanged": True,
        "current_runtime_initial_joint": np.asarray(robot.controller.initial_joint).copy(),
        "historical_initial_joint_verified": False,
    }


def worker(index):
    def interrupted(signum, _frame):
        raise InterruptedError(f"Worker received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    registration = prefix.read(prefix.reference(OUTPUT / "registration.json"))
    inputs = load_inputs()
    prefix.require(
        inputs["source_sha256"] == registration["source_sha256"]
        and inputs["asset_refs"] == registration["asset_refs"]
        and source_refs() == registration["source_refs"],
        "Frozen branch identities",
    )
    case = next(item for item in inputs["cases"] if item["spec"]["case_index"] == index)
    directory = OUTPUT / f"case_{index:02d}"
    result = {"complete": False, "passed": False, "case_index": index, "automatic_retry": False}
    try:
        cv2 = inputs["cv2_admission"]
        result["cv2_admission"] = restore_cv2_environment(cv2["cv2_probe_ref"], cv2["cv2_runtime_source_refs"])
        from s2a.research.native_runtime.development_prefix_runtime import PrefixReplayFailure, run_prefix_runtime

        def continuation(env, terminal, observers, receipt, *, progress=None):
            prefix.require(
                terminal["endpoint_receipt"]["endpoint_step"] == case["spec"]["prefix_steps"],
                "Verified warning endpoint",
            )
            return numerical_reconstruction(env, case, receipt)

        def progress(receipt):
            print(
                json.dumps(
                    {
                        "case": index,
                        "checked_endpoints": receipt.get("checked_endpoints", 0),
                        "controls": receipt.get("control_steps_completed", 0),
                        "preview_calls": receipt.get("preview_calls"),
                        "refiner_calls": receipt.get("refiner_calls"),
                        "native_steps": receipt["counters"]["mj_step"],
                    },
                    allow_nan=False,
                ),
                flush=True,
            )

        try:
            receipt = run_prefix_runtime(
                case["config"],
                case["record"],
                case["record_xml"],
                case["commands"],
                case["saved"],
                case["receipts"],
                directory / "runtime",
                inputs["asset_refs"],
                case["controller"],
                continuation=continuation,
                continuation_budget=1,
                progress=progress,
            )
        except PrefixReplayFailure as error:
            result["runtime"] = error.receipt
            raise
        prefix.require(
            receipt["control_steps_completed"] == case["spec"]["prefix_steps"]
            and receipt["counters"]["mj_step"]["completed"] == 25 * case["spec"]["prefix_steps"]
            and receipt["newly_selected_commands"] == 0,
            "No control was dispatched in reconstruction",
        )
        result.update(complete=True, passed=True, runtime=receipt)
    except BaseException as error:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        print(publish(directory / "result.json", result), flush=True)


def execute():
    prefix.require(not OUTPUT.exists(), "New output only; no retry")
    dry_ref, qa_ref = prefix.reference(DRY), prefix.reference(QA)
    dry, qa = prefix.read(dry_ref), prefix.read(qa_ref)
    prefix.require(dry["complete"] is dry["passed"] is True and dry["physics_steps"] == 0, "Dry closed")
    prefix.require(
        qa["complete"] is qa["passed"] is qa["source_unchanged"] is qa["actual_wait"] is qa["reaped"] is True,
        "CPU QA closed",
    )
    source_before = snapshot(REPO)["sources"]
    prefix.require(source_before == prefix.read(qa["source_before_ref"])["sources"], "Full QA source identity")
    sources, inputs = source_refs(), load_inputs()
    prefix.require(
        dry["source_refs"] == sources
        and dry["source_sha256"] == inputs["source_sha256"]
        and dry["asset_refs"] == inputs["asset_refs"],
        "Dry source/input identity",
    )
    registration = publish(
        OUTPUT / "registration.json",
        {
            "dry_ref": dry_ref,
            "qa_ref": qa_ref,
            "source_refs": sources,
            "source_sha256": inputs["source_sha256"],
            "asset_refs": inputs["asset_refs"],
            "cases": [2, 6],
            "max_control_steps": 95,
            "max_native_steps": 2375,
            "max_models": 6,
            "max_contexts": 6,
            "max_refiner_response_evaluations": 960,
            "new_actions_dispatched": 0,
            "world_model_calls": 0,
            "fresh_cases": 0,
            "automatic_retry": False,
        },
    )

    def interrupted(signum, _frame):
        raise KeyboardInterrupt(f"Owner received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    workers, launch_error = [], None
    try:
        for index in (2, 6):
            path = OUTPUT / f"case_{index:02d}"
            path.mkdir()
            log = (path / "worker.log").open("x")
            try:
                proc = subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "scripts.simulation.runtime.reconstruct_early_refiner",
                        "--worker",
                        str(index),
                    ],
                    cwd=REPO,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            except BaseException:
                log.close()
                raise
            workers.append({"index": index, "process": proc, "log": log})
    except BaseException:
        launch_error = traceback.format_exc()
        for item in workers:
            if item["process"].poll() is None:
                try:
                    item["process"].terminate()
                    item["signaled"] = True
                except ProcessLookupError:
                    pass
    lifecycle, deadline = [], time.monotonic() + 1800
    for item in workers:
        proc, sent = item["process"], item.get("signaled", False)
        try:
            code = proc.wait(timeout=max(0.01, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            sent = True
            proc.terminate()
            try:
                code = proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                code = proc.wait()
        except BaseException:
            launch_error = traceback.format_exc()
            for other in workers:
                if other["process"].poll() is None:
                    try:
                        other["process"].terminate()
                        other["signaled"] = True
                    except ProcessLookupError:
                        pass
            sent = item.get("signaled", False)
            try:
                code = proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                code = proc.wait()
        item["log"].close()
        lifecycle.append(
            {
                "case_index": item["index"],
                "pid": proc.pid,
                "returncode": code,
                "actual_wait": True,
                "reaped": True,
                "signal_sent": sent,
                "log_ref": prefix.reference(OUTPUT / f"case_{item['index']:02d}" / "worker.log"),
            }
        )
    workers_ref = publish(OUTPUT / "workers.json", {"workers": lifecycle, "launch_error": launch_error})
    result = {
        "complete": False,
        "passed": False,
        "registration_ref": registration,
        "workers_ref": workers_ref,
        "new_actions_dispatched": 0,
        "world_model_calls": 0,
        "fresh_cases": 0,
        "automatic_retry": False,
    }
    try:
        prefix.require(
            not launch_error
            and len(lifecycle) == 2
            and all(w["returncode"] == 0 and not w["signal_sent"] for w in lifecycle),
            "Both workers closed",
        )
        refs = [prefix.reference(OUTPUT / f"case_{i:02d}" / "result.json") for i in (2, 6)]
        cases = [prefix.read(ref) for ref in refs]
        prefix.require(all(c["complete"] is c["passed"] is True for c in cases), "Completed reconstruction cases")
        prefix.require(
            all(prefix.reference(path)["sha256"] == digest for path, digest in inputs["source_sha256"].items())
            and source_refs() == sources
            and snapshot(REPO)["sources"] == source_before,
            "Sources unchanged",
        )
        controls = sum(c["runtime"]["control_steps_completed"] for c in cases)
        native = sum(c["runtime"]["counters"]["mj_step"]["completed"] for c in cases)
        models = sum(c["runtime"]["counters"]["model_construct"]["completed"] for c in cases)
        contexts = sum(c["runtime"]["counters"]["context"]["completed"] for c in cases)
        responses = sum(c["runtime"]["continuation"]["evaluations"] for c in cases)
        prefix.require(
            controls == 95 and native == 2375 and models == contexts == 6 and responses <= 960,
            "Original reconstruction budgets",
        )
        result.update(
            complete=True,
            passed=True,
            sources_unchanged=True,
            case_refs=refs,
            control_steps=controls,
            native_steps=native,
            model_initializations=models,
            software_contexts=contexts,
            refiner_response_evaluations=responses,
            original_decisions_reproduced=all(
                c["runtime"]["continuation"]["comparison"]["all_within_tolerance"]
                and c["runtime"]["continuation"]["comparison"]["same_route"]
                for c in cases
            ),
        )
    finally:
        print(publish(OUTPUT / "result.json", result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true")
    group.add_argument("--execute", action="store_true")
    group.add_argument("--worker", type=int, choices=(2, 6))
    args = parser.parse_args()
    prefix.environment()
    if args.dry_run:
        prefix.require(not DRY.exists(), "New dry output only")
        inputs = load_inputs()
        print(
            publish(
                DRY,
                {
                    "complete": True,
                    "passed": True,
                    "source_refs": source_refs(),
                    "source_sha256": inputs["source_sha256"],
                    "asset_refs": inputs["asset_refs"],
                    "cases": [
                        dict(case=x["intervention_spec"], completed_periods=x["completed_history"].completed_periods)
                        for x in inputs["cases"]
                    ],
                    "physics_steps": 0,
                    "world_model_calls": 0,
                    "fresh_cases": 0,
                },
            ),
            flush=True,
        )
    elif args.execute:
        execute()
    else:
        worker(args.worker)


if __name__ == "__main__":
    main()
