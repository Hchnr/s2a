"""Run and independently replay the bounded tabletop feedback comparison."""

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
from scripts.simulation.runtime import prepare_tabletop_projection_feedback_probe as admission
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_causal_stream_live_48 as frozen
from scripts.simulation.runtime import run_causal_stream_live_cached as cached
from scripts.simulation.runtime import run_causal_stream_recorded_physical as physical
from scripts.simulation.runtime import run_causal_wam_two_window_feedback as historical
from scripts.simulation.runtime import run_quality_rate_repair_recorded_feedback as old_runner
from scripts.simulation.runtime.tabletop_projection_continuation import execute_continuation
from scripts.tracking.response_models import replay_contact_torque as contact

ROOT, OUTPUT, DRY, QA = admission.ROOT, admission.OUTPUT, admission.DRY, admission.QA
MODULE = "scripts.simulation.runtime.run_tabletop_projection_feedback_probe"
FORBIDDEN = contact.FORBIDDEN + (
    "s2a.research.joint_control.tracking.tabletop_projection_feedback",
    "s2a.research.joint_control.tracking.tabletop_reference_projection",
)


def source_refs():
    names = (
        admission.CONFIG,
        admission.PLAN,
        Path(__file__).resolve(),
        ROOT / "scripts/simulation/runtime/prepare_tabletop_projection_feedback_probe.py",
        ROOT / "scripts/simulation/runtime/tabletop_projection_continuation.py",
        ROOT / "scripts/simulation/runtime/tabletop_projection_measurement.py",
        ROOT / "scripts/simulation/runtime/tabletop_projection_inner_recording.py",
        ROOT / "src/s2a/research/joint_control/tracking/tabletop_projection_feedback.py",
        ROOT / "src/s2a/research/joint_control/tracking/tabletop_projection_observation.py",
        ROOT / "src/s2a/research/joint_control/tracking/tabletop_reference_projection.py",
        ROOT / "src/s2a/research/native_runtime/streaming_replay.py",
        ROOT / "src/s2a/research/native_runtime/development_continuation.py",
        ROOT / "tests/research/native_runtime/test_tabletop_projection_feedback_admission.py",
        ROOT / "tests/research/native_runtime/test_tabletop_projection_continuation.py",
        ROOT / "tests/research/native_runtime/test_tabletop_projection_measurement.py",
        ROOT / "tests/research/native_runtime/test_tabletop_projection_inner_recording.py",
        ROOT / "tests/research/native_runtime/test_tabletop_projection_feedback_probe.py",
        ROOT / "tests/research/joint_control/tracking/test_tabletop_projection_feedback.py",
        ROOT / "tests/research/joint_control/tracking/test_tabletop_projection_observation.py",
    )
    return contact.source_refs() + [prefix.reference(path) for path in names]


def checker_isolation():
    contact.isolation()
    prefix.require(
        all(name not in sys.modules for name in FORBIDDEN),
        "The independent checker cannot load candidate policy modules",
    )


def bounded_prefix(values, directory, continuation):
    """Retain the historical source while imposing the seven-period branch budget."""

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
            continuation_budget=47,
            continuation_limit=48,
            registered_seed=42,
            capture_final_prefix_inner=True,
        )

    run = old_runner.clone(cached._run_cached_prefix, {"frozen": SimpleNamespace(_run_prefix=fixed_budget)})
    return run(values, directory, continuation)


def worker(mode, phase):
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
        registration["mode"] == mode
        and registration["phase"] == phase
        and registration["inputs"] == admission.input_receipt(inputs)
        and registration["source_refs"] == source_refs(),
        "Worker registration, source and input identities must match",
    )
    item = inputs["tabletop_projection_feedback_probe"]
    prefix.require(mode in item["config"]["modes"] and phase in ("producer", "checker"), "Registered worker identity")
    baseline = None if registration["baseline_ref"] is None else prefix.read(registration["baseline_ref"])
    producer = None if registration["producer_ref"] is None else prefix.read(registration["producer_ref"])
    if baseline is not None:
        prefix.require(baseline["complete"] is baseline["passed"] is True, "Closed baseline producer required")
    if phase == "checker":
        prefix.require(
            producer is not None and producer["complete"] is producer["passed"] is True,
            "Closed matching producer required",
        )
    else:
        prefix.require(producer is None, "A producer cannot consume future decisions")
    values = inputs["old"]["old"]["base"]["previous"]
    result = dict(complete=False, passed=False, mode=mode, phase=phase, control_qualified=False, automatic_retry=False)
    try:
        cv2 = values["cv2_admission"]
        result["cv2_admission"] = restore_cv2_environment(cv2["cv2_probe_ref"], cv2["cv2_runtime_source_refs"])
        from s2a.research.native_runtime.development_prefix_runtime_48 import PrefixReplayFailure

        def continuation(env, terminal, observers, receipt, *, progress=None):
            frozen._check_prefix(receipt, item["previous_producer"]["runtime"]["prefix_inner_periods"])
            start = None if baseline is None else baseline["runtime"]["continuation"]["branch_start"]
            return execute_continuation(
                env,
                terminal,
                observers,
                receipt,
                inputs,
                mode=mode,
                producer=producer,
                baseline_start=start,
                progress=progress,
            )

        try:
            runtime = bounded_prefix(values, directory, continuation)
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


def assert_observations_match(left, right, *, path="sample"):
    """Compare all saved torque, contact, geometry and controller values."""
    if isinstance(left, dict) and isinstance(right, dict):
        prefix.require(set(left) == set(right), f"Observation fields differ at {path}")
        for key in left:
            assert_observations_match(left[key], right[key], path=f"{path}.{key}")
    elif isinstance(left, list) and isinstance(right, list):
        prefix.require(len(left) == len(right), f"Observation lengths differ at {path}")
        for index, (a, b) in enumerate(zip(left, right, strict=True)):
            assert_observations_match(a, b, path=f"{path}[{index}]")
    elif type(left) in (int, float) and type(right) in (int, float):
        np.testing.assert_allclose(left, right, atol=1e-9, rtol=0, err_msg=path)
    else:
        prefix.require(left == right, f"Observation values differ at {path}")


def summarize(branch, *, mode, phase, config):
    runtime = branch["runtime"]
    continuation = runtime["continuation"]
    periods, samples = continuation["probe"]["periods"], continuation["inner_samples"]
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
        and 0 <= len(periods) <= 7
        and len(samples) == 25 * len(periods)
        and runtime["control_steps_completed"] == 120 + len(periods)
        and runtime["counters"]["mj_step"]["completed"] == 25 * runtime["control_steps_completed"]
        and runtime["limits"]["mj_step"] == config["max_native_steps_per_phase"]
        and runtime["counters"]["model_construct"]["completed"] == runtime["counters"]["context"]["completed"] == 3
        and runtime["newly_selected_commands"] == (len(periods) if phase == "producer" else 0)
        and runtime["world_model_calls"] == runtime["fresh_cases"] == 0
        and [period["step"] for period in periods] == list(range(120, 120 + len(periods)))
        and all(
            [sample["step"] for sample in samples[i * 25 : (i + 1) * 25]] == [120 + i] * 25 for i in range(len(periods))
        ),
        "The registered physical, source and sample budget must hold",
    )
    pair = {"table_collision", "gripper0_finger2_collision"}
    contacts = [
        any({record["geom1"], record["geom2"]} == pair and record["active_constraint"] for record in sample["contacts"])
        for sample in samples
    ]
    clips = [
        np.max(np.abs(np.asarray(sample["controller_raw_torque_nm"]) - sample["commanded_torque_nm"])) > 1e-9
        for sample in samples
    ]
    return {
        "periods": len(periods),
        "stop": continuation["probe"]["stop"],
        "control_steps": runtime["control_steps_completed"],
        "native_steps": runtime["counters"]["mj_step"]["completed"],
        "newly_selected_commands": runtime["newly_selected_commands"],
        "table_finger_active_contact_inner_samples": sum(contacts),
        "torque_clipped_inner_samples": sum(clips),
        "original_pose_caps_satisfied": sum(row["pose_caps_satisfied"] for row in periods),
        "projected_pose_caps_satisfied": sum(
            row["completed_feedback"]["projected_pose_caps_satisfied"] for row in periods
        ),
        "max_reference_translation_m": (
            max((row["decision"]["reference_translation_m"] for row in periods), default=0.0)
            if phase == "producer"
            else None
        ),
        "stop_reason_evaluated_by_worker": phase == "producer",
        "producer_stop_independently_rechecked": False,
        "max_observed_shape_change_m": max(
            (row["bounded_sample_conditions"]["max_material_vertex_displacement_tcp_m"] for row in samples), default=0.0
        ),
        "all_sample_bounds_met": sum(
            sample["bounded_sample_conditions"]["all_sample_bounds_met"] for sample in samples
        ),
    }


def check_pair(producer, checker):
    left, right = producer["runtime"]["continuation"], checker["runtime"]["continuation"]
    prefix.require(
        checker["candidate_modules_unloaded"] is True
        and len(left["probe"]["periods"]) == len(right["probe"]["periods"]),
        "An isolated complete action replay is required",
    )
    for original, replay in zip(left["probe"]["periods"], right["probe"]["periods"], strict=True):
        historical.compare_first_period(original, replay)
        assert_observations_match(original["completed_feedback"], replay["completed_feedback"], path="endpoint")
    assert_observations_match(left["inner_samples"], right["inner_samples"], path="inner_samples")


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
    item, sources, before = inputs["tabletop_projection_feedback_probe"], source_refs(), snapshot(ROOT)["sources"]
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
        "Source-matched 32-worker QA and a zero-work dry are required",
    )
    OUTPUT.mkdir(exist_ok=False)
    result = dict(
        complete=False,
        passed=False,
        control_qualified=False,
        automatic_retry=False,
        world_model_calls=0,
        fresh_cases=0,
        groups=[],
    )
    try:
        env = physical._cpu_environment(OUTPUT)
        env["MUJOCO_GL"] = "osmesa"

        def register(mode, phase, baseline_ref, producer_ref):
            directory = OUTPUT / mode / phase
            (directory / "simulation").mkdir(parents=True)
            publish(
                directory / "registration.json",
                dict(
                    mode=mode,
                    phase=phase,
                    inputs=admission.input_receipt(inputs),
                    source_refs=sources,
                    baseline_ref=baseline_ref,
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

        group("baseline_producer", [register("original", "producer", None, None)])
        baseline_ref = prefix.reference(OUTPUT / "original/producer/simulation/result.json")
        baseline = prefix.read(baseline_ref)
        summaries = {
            "original": {"producer": summarize(baseline, mode="original", phase="producer", config=item["config"])}
        }
        group(
            "projected_producers",
            [register(mode, "producer", baseline_ref, None) for mode in ("nominal", "pose_bounded")],
        )
        refs, producers = {}, {"original": baseline}
        for mode in item["config"]["modes"]:
            refs[mode] = prefix.reference(OUTPUT / mode / "producer/simulation/result.json")
            producers[mode] = prefix.read(refs[mode])
            summaries.setdefault(mode, {})["producer"] = summarize(
                producers[mode], mode=mode, phase="producer", config=item["config"]
            )
        group("checkers", [register(mode, "checker", baseline_ref, refs[mode]) for mode in item["config"]["modes"]])
        checker_refs = {}
        for mode in item["config"]["modes"]:
            checker_refs[mode] = prefix.reference(OUTPUT / mode / "checker/simulation/result.json")
            checked = prefix.read(checker_refs[mode])
            summaries[mode]["checker"] = summarize(checked, mode=mode, phase="checker", config=item["config"])
            check_pair(producers[mode], checked)
        total_steps = sum(
            summary[phase]["native_steps"] for summary in summaries.values() for phase in ("producer", "checker")
        )
        new_commands = sum(summary["producer"]["newly_selected_commands"] for summary in summaries.values())
        prefix.require(
            total_steps <= 19050 and new_commands <= item["config"]["max_new_commands"],
            "Total physical and new-command budgets must hold",
        )
        prefix.require(
            source_refs() == sources and snapshot(ROOT)["sources"] == before, "Source identity changed during the probe"
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
            total_native_steps=total_steps,
            newly_selected_commands=new_commands,
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
