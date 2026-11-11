"""Run branch-specific live WAM references through bounded physical feedback."""

import argparse
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

from s2a.infrastructure.native_environment import restore_cv2_environment
from s2a.infrastructure.owned_processes import run_owned_workers
from s2a.infrastructure.publication import publish_result as publish
from scripts.maintenance.run_cpu_tests import snapshot
from scripts.simulation.runtime import prepare_early_tabletop_live_feedback as admission
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_causal_stream_live_48 as frozen
from scripts.simulation.runtime import run_causal_stream_live_cached as cached
from scripts.simulation.runtime import run_causal_stream_recorded_physical as physical
from scripts.simulation.runtime import run_early_tabletop_capture as capture_owner
from scripts.simulation.runtime import run_tabletop_projection_feedback_probe as previous
from scripts.simulation.runtime.early_tabletop_live_continuation import execute_live_continuation
from scripts.wam.tracking.preview import early_tabletop_live_wam as model

ROOT, OUTPUT, DRY, QA = admission.ROOT, admission.OUTPUT, admission.DRY, admission.QA
MODULE = "scripts.simulation.runtime.run_early_tabletop_live_feedback"


def source_refs():
    return capture_owner.source_refs() + [
        prefix.reference(path)
        for path in (
            admission.CONFIG,
            admission.PLAN,
            Path(__file__).resolve(),
            Path(model.__file__).resolve(),
            ROOT / "scripts/simulation/runtime/prepare_early_tabletop_live_feedback.py",
            ROOT / "scripts/simulation/runtime/early_tabletop_live_continuation.py",
            ROOT / "tests/research/native_runtime/test_early_tabletop_live_feedback_admission.py",
            ROOT / "tests/research/native_runtime/test_early_tabletop_live_continuation.py",
            ROOT / "tests/research/native_runtime/test_early_tabletop_live_feedback.py",
            ROOT / "tests/research/wam/test_early_tabletop_live_wam.py",
        )
    ]


def worker(mode, phase):
    directory = OUTPUT / mode / phase / "simulation"
    temp, cache = directory / "tmp", directory / "cache"
    temp.mkdir()
    cache.mkdir()
    os.environ.update(TMPDIR=str(temp), XDG_CACHE_HOME=str(cache))
    tempfile.tempdir = None
    prefix.environment()
    if phase == "checker":
        previous.checker_isolation()
    inputs = admission.load_inputs()
    registration = prefix.read(prefix.reference(OUTPUT / mode / phase / "registration.json"))
    prefix.require(
        registration["mode"] == mode
        and registration["phase"] == phase
        and registration["inputs"] == admission.input_receipt(inputs)
        and registration["source_refs"] == source_refs(),
        "Worker registration, sources and branch-bound inputs must match",
    )
    item = inputs["early_tabletop_live_feedback"]
    prefix.require(mode in item["config"]["modes"] and phase in ("producer", "checker"), "Registered worker identity")
    producer = None if registration["producer_ref"] is None else prefix.read(registration["producer_ref"])
    if phase == "checker":
        prefix.require(
            producer is not None and producer["complete"] is producer["passed"] is True,
            "A closed matching producer is required",
        )
    else:
        prefix.require(producer is None, "A producer cannot consume its future actions")
    values = inputs["old"]["old"]["base"]["previous"]
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
            return execute_live_continuation(
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
            runtime = cached._run_cached_prefix(values, directory, continuation)
        except PrefixReplayFailure as error:
            result["runtime"] = error.receipt
            raise
        if phase == "checker":
            previous.checker_isolation()
        result.update(complete=True, passed=True, runtime=runtime, candidate_modules_unloaded=phase == "checker")
    except BaseException as error:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        print(publish(directory / "result.json", result), flush=True)


def summarize(branch, *, mode, phase, config):
    runtime, continuation = branch["runtime"], branch["runtime"]["continuation"]
    periods, samples = continuation["probe"]["periods"], continuation["inner_samples"]
    response = next(item for item in config["responses"] if item["mode"] == mode)
    prefix.require(
        branch["complete"]
        is branch["passed"]
        is runtime["complete"]
        is runtime["passed"]
        is runtime["cleanup_complete"]
        is True
        and branch["mode"] == mode
        and branch["phase"] == phase
        and continuation["mode"] == mode
        and 0 <= len(periods) <= 8
        and len(samples) == 25 * len(periods)
        and runtime["control_steps_completed"] == 120 + len(periods)
        and runtime["counters"]["mj_step"]["completed"] == 25 * runtime["control_steps_completed"]
        and runtime["limits"]["mj_step"] == config["max_native_steps_per_phase"]
        and runtime["counters"]["model_construct"]["completed"] == runtime["counters"]["context"]["completed"] == 3
        and runtime["newly_selected_commands"] == (len(periods) if phase == "producer" else 0)
        and runtime["world_model_calls"] == runtime["fresh_cases"] == 0
        and continuation["model_state_sha256"] == response["state_sha256"]
        and continuation["model_image_ref"]["sha256"] == response["image_sha256"]
        and continuation["model_response_ref"]["sha256"] == response["sha256"]
        and [period["step"] for period in periods] == list(range(120, 120 + len(periods)))
        and all(
            [sample["step"] for sample in samples[i * 25 : (i + 1) * 25]] == [120 + i] * 25 for i in range(len(periods))
        ),
        "The branch-specific reference, feedback and native budgets must hold",
    )
    table_pair = {"table_collision", "gripper0_finger2_collision"}
    stove_pair = {"flat_stove_1_collision_burner", "gripper0_finger2_collision"}

    def active_pair(sample, pair):
        return any(
            {record["geom1"], record["geom2"]} == pair and record["active_constraint"] for record in sample["contacts"]
        )

    return {
        "periods": len(periods),
        "stop": continuation["probe"]["stop"],
        "control_steps": runtime["control_steps_completed"],
        "native_steps": runtime["counters"]["mj_step"]["completed"],
        "newly_selected_commands": runtime["newly_selected_commands"],
        "original_pose_caps_satisfied": sum(row["pose_caps_satisfied"] for row in periods),
        "selected_pose_caps_satisfied": sum(
            row["completed_feedback"]["projected_pose_caps_satisfied"] for row in periods
        ),
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
        "max_reference_translation_m": max((row["decision"]["reference_translation_m"] for row in periods), default=0.0)
        if phase == "producer"
        else None,
        "max_observed_shape_change_m": max(
            (row["bounded_sample_conditions"]["max_material_vertex_displacement_tcp_m"] for row in samples), default=0.0
        ),
        "all_sample_bounds_met": sum(row["bounded_sample_conditions"]["all_sample_bounds_met"] for row in samples),
        "producer_stop_independently_rechecked": False,
    }


def check_pair(producer, checker):
    previous.check_pair(producer, checker)
    a, b = producer["runtime"]["continuation"], checker["runtime"]["continuation"]
    prefix.require(
        a["model_image_ref"] == b["model_image_ref"]
        and a["model_state_sha256"] == b["model_state_sha256"]
        and a["model_response_ref"] == b["model_response_ref"],
        "Physical replay must use the same branch-bound model reference",
    )
    np.testing.assert_allclose(a["original_targets"], b["original_targets"], atol=1e-12, rtol=0)


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
            ),
        ),
        flush=True,
    )


def execute():
    inputs = admission.load_inputs()
    item, sources, before = inputs["early_tabletop_live_feedback"], source_refs(), snapshot(ROOT)["sources"]
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
    OUTPUT.mkdir(parents=True, exist_ok=False)
    result = dict(
        complete=False,
        passed=False,
        control_qualified=False,
        automatic_retry=False,
        new_world_model_calls=0,
        fresh_cases=0,
        groups=[],
    )
    try:
        env = physical._cpu_environment(OUTPUT)
        env["MUJOCO_GL"] = "osmesa"

        def register(mode, phase, producer_ref):
            directory = OUTPUT / mode / phase
            (directory / "simulation").mkdir(parents=True)
            publish(
                directory / "registration.json",
                dict(
                    mode=mode,
                    phase=phase,
                    inputs=admission.input_receipt(inputs),
                    source_refs=sources,
                    producer_ref=producer_ref,
                ),
            )
            return dict(
                key=f"{mode}-{phase}",
                argv=[sys.executable, "-m", MODULE, "--worker", mode, phase],
                log=directory / "worker.log",
            )

        def group(name, specs):
            with physical._temporary_environment(env):
                lifecycle = run_owned_workers(specs, cwd=ROOT, timeout_seconds=item["config"]["worker_timeout_seconds"])
            result["groups"].append({"name": name, "workers_ref": publish(OUTPUT / f"{name}_workers.json", lifecycle)})
            prefix.require(
                lifecycle["launch_error"] is None
                and len(lifecycle["workers"]) == len(specs)
                and all(
                    row["actual_wait"] is row["reaped"] is True
                    and row["returncode"] == 0
                    and row["signal_sent"] is False
                    for row in lifecycle["workers"]
                ),
                f"Every {name} worker must exit and be reaped",
            )

        group("producers", [register(mode, "producer", None) for mode in item["config"]["modes"]])
        refs, producers, summaries = {}, {}, {}
        for mode in item["config"]["modes"]:
            refs[mode] = prefix.reference(OUTPUT / mode / "producer/simulation/result.json")
            producers[mode] = prefix.read(refs[mode])
            summaries[mode] = {
                "producer": summarize(producers[mode], mode=mode, phase="producer", config=item["config"])
            }
        group("checkers", [register(mode, "checker", refs[mode]) for mode in item["config"]["modes"]])
        checker_refs = {}
        for mode in item["config"]["modes"]:
            checker_refs[mode] = prefix.reference(OUTPUT / mode / "checker/simulation/result.json")
            checker = prefix.read(checker_refs[mode])
            summaries[mode]["checker"] = summarize(checker, mode=mode, phase="checker", config=item["config"])
            check_pair(producers[mode], checker)
        total = sum(
            summary[phase]["native_steps"] for summary in summaries.values() for phase in ("producer", "checker")
        )
        new = sum(summary["producer"]["newly_selected_commands"] for summary in summaries.values())
        prefix.require(total <= 12800 and new <= 16, "The live feedback physical budget must hold")
        prefix.require(
            source_refs() == sources and snapshot(ROOT)["sources"] == before, "Feedback source identity changed"
        )
        for path, digest in item["source_sha256"].items():
            prefix.checked({"path": path, "sha256": digest})
        result.update(
            complete=True,
            passed=True,
            sources_unchanged=True,
            producer_refs=refs,
            checker_refs=checker_refs,
            summaries=summaries,
            total_native_steps=total,
            newly_selected_commands=new,
            prior_capture_native_steps=12000,
            experiment_native_steps=12000 + total,
            prior_model_requests=2,
            prior_denoise_calls=70,
        )
    except BaseException as error:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        print(publish(OUTPUT / "result.json", result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true")
    group.add_argument("--execute", action="store_true")
    group.add_argument("--worker", nargs=2, metavar=("MODE", "PHASE"))
    args = parser.parse_args()
    if args.dry_run:
        dry_run()
    elif args.execute:
        execute()
    else:
        worker(*args.worker)


if __name__ == "__main__":
    main()
