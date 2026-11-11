"""Run and independently replay a six-period box-face feedback probe."""

import argparse
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from s2a.infrastructure.native_environment import restore_cv2_environment
from s2a.infrastructure.owned_processes import run_owned_workers
from s2a.infrastructure.publication import publish_result as publish
from scripts.maintenance.run_cpu_tests import snapshot
from scripts.simulation.runtime import prepare_box_face_feedback as admission
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_causal_stream_live_48 as frozen
from scripts.simulation.runtime import run_causal_stream_live_cached as cached
from scripts.simulation.runtime import run_causal_stream_recorded_physical as physical
from scripts.simulation.runtime import run_early_tabletop_live_feedback as earlier
from scripts.simulation.runtime import run_quality_rate_repair_recorded_feedback as cloning
from scripts.simulation.runtime.box_face_feedback_continuation import checker_isolation, execute_face_continuation

ROOT, OUTPUT, DRY = admission.ROOT, admission.OUTPUT, admission.DRY
QA = ROOT / "outputs/qa/box_face_feedback_after_qa_fix_20261111/result.json"
MODULE = "scripts.simulation.runtime.run_box_face_feedback"


def source_refs():
    return earlier.source_refs() + [
        prefix.reference(path)
        for path in (
            admission.CONFIG,
            admission.PLAN,
            Path(__file__).resolve(),
            ROOT / "scripts/simulation/runtime/prepare_box_face_feedback.py",
            ROOT / "scripts/simulation/runtime/box_face_feedback_continuation.py",
            ROOT / "scripts/simulation/runtime/box_face_feedback_measurement.py",
            ROOT / "src/s2a/research/joint_control/tracking/box_face_feedback.py",
            ROOT / "src/s2a/research/joint_control/tracking/box_face_reference_projection.py",
            ROOT / "tests/research/native_runtime/test_box_face_feedback_admission.py",
            ROOT / "tests/research/native_runtime/test_box_face_feedback_continuation.py",
            ROOT / "tests/research/native_runtime/test_box_face_feedback_measurement.py",
            ROOT / "tests/research/native_runtime/test_box_face_feedback_owner.py",
            ROOT / "tests/research/joint_control/tracking/test_box_face_feedback.py",
            ROOT / "tests/research/joint_control/tracking/test_box_face_reference_projection.py",
        )
    ]


def continuation_budget(prefix_count, config):
    budget = config["max_control_steps_per_phase"] - prefix_count
    prefix.require(
        prefix_count == 80 and budget == 46 and 25 * (prefix_count + budget) == config["max_native_steps_per_phase"],
        "The native runtime must enforce the registered step budget",
    )
    return budget


def bounded_prefix(values, directory, continuation, config):
    """Keep the same cached runtime while bounding all physical inner steps."""

    def fixed_budget(old, output, callback):
        from s2a.research.native_runtime.guarded_stream_prefix import GuardedStreamPrefixRuntime

        return GuardedStreamPrefixRuntime()._run(
            old["runtime_config"],
            old["record"],
            old["record_xml"],
            old["commands"],
            old["saved"],
            old["receipts"],
            output / "runtime",
            old["asset_refs"],
            old["controller"],
            continuation=callback,
            continuation_budget=continuation_budget(len(old["commands"]), config),
            continuation_limit=48,
            registered_seed=42,
            capture_final_prefix_inner=True,
        )

    run = cloning.clone(cached._run_cached_prefix, {"frozen": SimpleNamespace(_run_prefix=fixed_budget)})
    return run(values, directory, continuation)


def worker(phase):
    mode = "original"
    directory = OUTPUT / mode / phase / "simulation"
    temp, cache = directory / "tmp", directory / "cache"
    temp.mkdir()
    cache.mkdir()
    os.environ.update(TMPDIR=str(temp), XDG_CACHE_HOME=str(cache))
    tempfile.tempdir = None
    prefix.environment()
    if phase == "checker":
        checker_isolation()
    inputs = admission.load_inputs()
    registration = prefix.read(prefix.reference(OUTPUT / mode / phase / "registration.json"))
    prefix.require(
        registration["phase"] == phase
        and registration["inputs"] == admission.input_receipt(inputs)
        and registration["source_refs"] == source_refs()
        and phase in ("producer", "checker"),
        "Worker phase, inputs and sources must match the owner registration",
    )
    producer = None if registration["producer_ref"] is None else prefix.read(registration["producer_ref"])
    prefix.require(
        (phase == "producer" and producer is None)
        or (phase == "checker" and producer is not None and producer["complete"] is producer["passed"] is True),
        "Only the checker may read the closed producer",
    )
    values = inputs["old"]["old"]["base"]["previous"]
    config = inputs["box_face_feedback"]["config"]
    result = dict(complete=False, passed=False, mode=mode, phase=phase, control_qualified=False, automatic_retry=False)
    try:
        cv2 = values["cv2_admission"]
        result["cv2_admission"] = restore_cv2_environment(cv2["cv2_probe_ref"], cv2["cv2_runtime_source_refs"])
        from s2a.research.native_runtime.development_prefix_runtime_48 import PrefixReplayFailure

        def continuation(env, terminal, observers, receipt, *, progress=None):
            frozen._check_prefix(
                receipt,
                inputs["tabletop_projection_feedback_probe"]["previous_producer"]["runtime"]["prefix_inner_periods"],
            )
            return execute_face_continuation(
                env,
                terminal,
                observers,
                receipt,
                inputs,
                mode=mode,
                output=directory / "runtime",
                producer=producer,
                progress=progress,
            )

        try:
            runtime = bounded_prefix(values, directory, continuation, config)
        except PrefixReplayFailure as error:
            result["runtime"] = error.receipt
            raise
        if phase == "checker":
            checker_isolation()
        result.update(complete=True, passed=True, runtime=runtime, candidate_modules_unloaded=phase == "checker")
    except BaseException as error:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        print(publish(directory / "result.json", result), flush=True)


def summarize_periods(periods, samples, *, has_slide_conditions):
    prefix.require(len(samples) == 25 * len(periods), "Every compared period requires all 25 physical samples")
    table_pair = {"table_collision", "gripper0_finger2_collision"}
    stove_pair = {"flat_stove_1_collision_burner", "gripper0_finger2_collision"}

    def active_pair(sample, pair):
        return any(
            {record["geom1"], record["geom2"]} == pair and record["active_constraint"] for record in sample["contacts"]
        )

    errors = [float(np.linalg.norm(np.asarray(row["tcp_error"])[:3])) for row in periods]
    result = {
        "periods": len(periods),
        "original_pose_caps_satisfied": sum(row["pose_caps_satisfied"] for row in periods),
        "selected_pose_caps_satisfied": sum(
            row["completed_feedback"]["projected_pose_caps_satisfied"] for row in periods
        ),
        "mean_original_position_error_m": float(np.mean(errors)) if errors else None,
        "max_original_position_error_m": max(errors, default=None),
        "task_success_endpoints": sum(row["task_success"] for row in periods),
        "target_grasp_endpoints": sum(row["target_grasp"] for row in periods),
        "table_finger_active_contact_inner_samples": sum(active_pair(row, table_pair) for row in samples),
        "stove_finger_active_contact_inner_samples": sum(active_pair(row, stove_pair) for row in samples),
        "robot_environment_active_contact_inner_samples": sum(
            any(c["kind"] == "robot_environment" and c["active_constraint"] for c in row["contacts"]) for row in samples
        ),
        "torque_clipped_inner_samples": sum(
            np.max(np.abs(np.asarray(row["controller_raw_torque_nm"]) - row["commanded_torque_nm"])) > 1e-9
            for row in samples
        ),
        "inner_arm_below_buffer_samples": sum(row["inner_below_buffer_samples"] for row in periods),
    }
    if has_slide_conditions:
        conditions = [row["bounded_sample_conditions"] for row in samples]
        result.update(
            finger_joint_range_met_samples=sum(row["finger_joint_range_met"] for row in conditions),
            position_bound_met_samples=sum(row["position_bound_met"] for row in conditions),
            rotation_bound_met_samples=sum(row["rotation_bound_met"] for row in conditions),
            max_slide_vertex_residual_m=max((row["slide_vertex_residual_m"] for row in conditions), default=None),
            max_slide_axis_change=max((row["slide_axis_change_tcp"] for row in conditions), default=None),
        )
    return result


def summarize(branch, phase, config):
    runtime, continuation = branch["runtime"], branch["runtime"]["continuation"]
    periods, samples = continuation["probe"]["periods"], continuation["inner_samples"]
    prefix.require(
        branch["complete"]
        is branch["passed"]
        is runtime["complete"]
        is runtime["passed"]
        is runtime["cleanup_complete"]
        is True
        and branch["mode"] == continuation["mode"] == "original"
        and branch["phase"] == phase
        and continuation["strategy"] == "box_face_projection"
        and 0 <= len(periods) <= config["new_period_limit"]
        and len(samples) == 25 * len(periods)
        and runtime["control_steps_completed"] == 120 + len(periods)
        and runtime["counters"]["mj_step"]["completed"] == 25 * runtime["control_steps_completed"]
        and runtime["limits"]["mj_step"] == config["max_native_steps_per_phase"]
        and runtime["counters"]["model_construct"]["completed"] == runtime["counters"]["context"]["completed"] == 3
        and runtime["newly_selected_commands"] == (len(periods) if phase == "producer" else 0)
        and runtime["world_model_calls"] == runtime["fresh_cases"] == 0
        and continuation["model_state_sha256"] == config["model_state_sha256"]
        and continuation["model_image_ref"]["sha256"] == config["model_image_sha256"]
        and continuation["model_response_ref"]["sha256"] == config["model_response_sha256"]
        and [period["step"] for period in periods] == list(range(120, 120 + len(periods))),
        "The branch-bound reference, physical samples and hard budgets must hold",
    )
    result = summarize_periods(periods, samples, has_slide_conditions=True)
    result.update(
        stop=continuation["probe"]["stop"],
        native_steps=runtime["counters"]["mj_step"]["completed"],
        newly_selected_commands=runtime["newly_selected_commands"],
    )
    return result


def dry_run():
    inputs = admission.load_inputs()
    print(
        publish(
            DRY,
            dict(
                complete=True,
                passed=True,
                inputs=admission.input_receipt(inputs),
                source_refs=source_refs(),
                native_models=0,
                physics_steps=0,
                world_model_calls=0,
                fresh_cases=0,
                automatic_retry=False,
            ),
        ),
        flush=True,
    )


def execute():
    inputs = admission.load_inputs()
    item = inputs["box_face_feedback"]
    sources, before = source_refs(), snapshot(ROOT)["sources"]
    qa, dry = prefix.read(prefix.reference(QA)), prefix.read(prefix.reference(DRY))
    prefix.require(
        qa["complete"] is qa["passed"] is qa["all_tests_passed"] is qa["source_unchanged"] is True
        and qa["actual_wait"] is qa["reaped"] is True
        and qa["signal_sent"] is False
        and qa["configured_parallelism"] == 32
        and qa["skipped_count"] == qa["xfail_count"] == qa["xpass_count"] == 0
        and not qa["collection"]["collection_errors"]
        and prefix.read(qa["source_before_ref"])["sources"] == before
        and dry["complete"] is dry["passed"] is True
        and dry["inputs"] == admission.input_receipt(inputs)
        and dry["source_refs"] == sources
        and dry["native_models"] == dry["physics_steps"] == dry["world_model_calls"] == 0,
        "Source-matched 32-worker QA and zero-work dry are required",
    )
    OUTPUT.mkdir(exist_ok=False)
    result = dict(
        complete=False,
        passed=False,
        control_qualified=False,
        automatic_retry=False,
        new_world_model_calls=0,
        fresh_cases=0,
        prior_experiment_native_steps=30800,
        prior_model_requests=2,
        groups=[],
    )
    try:
        env = physical._cpu_environment(OUTPUT)
        env["MUJOCO_GL"] = "osmesa"

        def run_group(phase, producer_ref):
            directory = OUTPUT / "original" / phase
            (directory / "simulation").mkdir(parents=True)
            publish(
                directory / "registration.json",
                dict(
                    phase=phase, inputs=admission.input_receipt(inputs), source_refs=sources, producer_ref=producer_ref
                ),
            )
            spec = dict(key=phase, argv=[sys.executable, "-m", MODULE, "--worker", phase], log=directory / "worker.log")
            with physical._temporary_environment(env):
                lifecycle = run_owned_workers(
                    [spec], cwd=ROOT, timeout_seconds=item["config"]["worker_timeout_seconds"]
                )
            result["groups"].append(
                {"phase": phase, "workers_ref": publish(OUTPUT / f"{phase}_workers.json", lifecycle)}
            )
            prefix.require(
                lifecycle["launch_error"] is None
                and len(lifecycle["workers"]) == 1
                and all(
                    row["actual_wait"] is row["reaped"] is True
                    and row["returncode"] == 0
                    and row["signal_sent"] is False
                    for row in lifecycle["workers"]
                ),
                f"The {phase} worker must exit and be reaped",
            )
            ref = prefix.reference(directory / "simulation/result.json")
            return ref, prefix.read(ref)

        producer_ref, producer = run_group("producer", None)
        producer_summary = summarize(producer, "producer", item["config"])
        checker_ref, checker = run_group("checker", producer_ref)
        checker_summary = summarize(checker, "checker", item["config"])
        earlier.check_pair(producer, checker)
        n = producer_summary["periods"]
        baseline = item["baseline_producer"]["runtime"]["continuation"]
        baseline_summary = summarize_periods(
            baseline["probe"]["periods"][:n], baseline["inner_samples"][: 25 * n], has_slide_conditions=False
        )
        total = producer_summary["native_steps"] + checker_summary["native_steps"]
        prefix.require(
            total <= item["config"]["max_total_native_steps"]
            and producer_summary["newly_selected_commands"] <= item["config"]["max_new_commands"],
            "The actual total physical budget must hold",
        )
        prefix.require(
            source_refs() == sources and snapshot(ROOT)["sources"] == before,
            "Sources changed during the physical comparison",
        )
        for path, digest in item["source_sha256"].items():
            prefix.checked({"path": path, "sha256": digest})
        result.update(
            complete=True,
            passed=True,
            sources_unchanged=True,
            producer_ref=producer_ref,
            checker_ref=checker_ref,
            producer_summary=producer_summary,
            checker_summary=checker_summary,
            baseline_summary=baseline_summary,
            matched_periods=n,
            new_native_steps=total,
            new_selected_commands=producer_summary["newly_selected_commands"],
            producer_stop_independently_rechecked=False,
        )
    except BaseException as error:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        partial = []
        for phase in ("producer", "checker"):
            path = OUTPUT / "original" / phase / "simulation/result.json"
            if path.is_file():
                ref = prefix.reference(path)
                branch = prefix.read(ref)
                runtime = branch.get("runtime", {})
                partial.append(
                    dict(
                        phase=phase,
                        ref=ref,
                        native_steps_completed=runtime.get("counters", {}).get("mj_step", {}).get("completed"),
                        newly_selected_commands=runtime.get("newly_selected_commands"),
                    )
                )
        result["partial_branches"] = partial
        raise
    finally:
        print(publish(OUTPUT / "result.json", result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true")
    group.add_argument("--execute", action="store_true")
    group.add_argument("--worker", choices=("producer", "checker"))
    args = parser.parse_args()
    if args.dry_run:
        dry_run()
    elif args.execute:
        execute()
    else:
        worker(args.worker)


if __name__ == "__main__":
    main()
