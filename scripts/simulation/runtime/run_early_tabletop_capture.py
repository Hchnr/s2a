"""Run and independently replay two early interventions before live WAM."""

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
from scripts.simulation.runtime import prepare_early_tabletop_intervention as admission
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_causal_stream_live_48 as frozen
from scripts.simulation.runtime import run_causal_stream_live_cached as cached
from scripts.simulation.runtime import run_causal_stream_recorded_physical as physical
from scripts.simulation.runtime import run_quality_rate_repair_recorded_feedback as old_runner
from scripts.simulation.runtime import run_tabletop_projection_feedback_probe as previous
from scripts.simulation.runtime.early_tabletop_continuation import execute_early_continuation

ROOT = admission.ROOT
OUTPUT = admission.OUTPUT / "phase_a"
DRY = ROOT / "outputs/early_tabletop_capture_dry_20261111/result.json"
QA = ROOT / "outputs/qa/early_tabletop_capture_20261111/result.json"
MODULE = "scripts.simulation.runtime.run_early_tabletop_capture"


def source_refs():
    return previous.source_refs() + [
        prefix.reference(path)
        for path in (
            admission.CONFIG,
            admission.PLAN,
            Path(__file__).resolve(),
            ROOT / "scripts/simulation/runtime/prepare_early_tabletop_intervention.py",
            ROOT / "scripts/simulation/runtime/early_tabletop_continuation.py",
            ROOT / "tests/research/native_runtime/test_early_tabletop_intervention_admission.py",
            ROOT / "tests/research/native_runtime/test_early_tabletop_continuation.py",
            ROOT / "tests/research/native_runtime/test_early_tabletop_inner_recording.py",
            ROOT / "tests/research/native_runtime/test_early_tabletop_capture.py",
        )
    ]


def bounded_prefix(values, directory, continuation):
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
            continuation_budget=40,
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
        previous.checker_isolation()
    inputs = admission.load_inputs()
    registration = prefix.read(prefix.reference(OUTPUT / mode / phase / "registration.json"))
    prefix.require(
        registration["mode"] == mode
        and registration["phase"] == phase
        and registration["inputs"] == admission.input_receipt(inputs)
        and registration["source_refs"] == source_refs(),
        "Worker registration, source and input identities must match",
    )
    item = inputs["early_tabletop_intervention"]
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
            start = None if baseline is None else baseline["runtime"]["continuation"]["branch_start"]
            return execute_early_continuation(
                env,
                terminal,
                observers,
                receipt,
                inputs,
                mode=mode,
                output=directory / "runtime",
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
            previous.checker_isolation()
        result.update(complete=True, passed=True, runtime=runtime, candidate_modules_unloaded=phase == "checker")
    except BaseException as error:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        print(publish(directory / "result.json", result), flush=True)


def compare_capture(producer, checker):
    """Verify that both independently rendered the same current endpoint."""
    left, right = (
        producer["runtime"]["continuation"]["next_current_image"],
        checker["runtime"]["continuation"]["next_current_image"],
    )
    prefix.require(left is not None and right is not None, "Both verified endpoint captures are required")
    a, b = left["capture"], right["capture"]
    prefix.require(
        a["frame_index"] == b["frame_index"] == 120
        and left["endpoint"]["sim_state_sha256"] == right["endpoint"]["sim_state_sha256"]
        and left["endpoint"]["sim_time"] == right["endpoint"]["sim_time"]
        and a["canvas_png"]["sha256"] == b["canvas_png"]["sha256"]
        and a["canvas_rgb_sha256"] == b["canvas_rgb_sha256"]
        and a["future_rgb_frames"] == b["future_rgb_frames"] == 0,
        "Replay image, state, time and absence of future frames must match",
    )
    np.testing.assert_allclose(a["current_source_pose"], b["current_source_pose"], atol=1e-10, rtol=0)
    np.testing.assert_allclose(a["current_opening"], b["current_opening"], atol=1e-10, rtol=0)
    prefix.checked(a["canvas_png"])
    prefix.checked(b["canvas_png"])
    return {
        "state_sha256": left["endpoint"]["sim_state_sha256"],
        "sim_time_s": left["endpoint"]["sim_time"],
        "image_ref": a["canvas_png"],
        "replay_image_ref": b["canvas_png"],
    }


def summarize(branch, *, mode, phase, config):
    runtime, continuation = branch["runtime"], branch["runtime"]["continuation"]
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
        and 0 <= len(periods) <= 2
        and len(samples) == 25 * len(periods)
        and runtime["control_steps_completed"] == 118 + len(periods)
        and runtime["counters"]["mj_step"]["completed"] == 25 * runtime["control_steps_completed"]
        and runtime["limits"]["mj_step"] == config["stage_a_max_native_steps_per_phase"]
        and runtime["counters"]["model_construct"]["completed"] == runtime["counters"]["context"]["completed"] == 3
        and runtime["newly_selected_commands"] == (len(periods) if phase == "producer" else 0)
        and runtime["world_model_calls"] == runtime["fresh_cases"] == 0
        and [period["step"] for period in periods] == list(range(118, 118 + len(periods)))
        and all(
            [sample["step"] for sample in samples[i * 25 : (i + 1) * 25]] == [118 + i] * 25 for i in range(len(periods))
        ),
        "The early branch, captured samples and native budget must hold",
    )
    return {
        "periods": len(periods),
        "control_steps": runtime["control_steps_completed"],
        "native_steps": runtime["counters"]["mj_step"]["completed"],
        "newly_selected_commands": runtime["newly_selected_commands"],
        "stop": continuation["probe"]["stop"],
        "current_image_captured": continuation["next_current_image"] is not None,
    }


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
    item, sources, before = inputs["early_tabletop_intervention"], source_refs(), snapshot(ROOT)["sources"]
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
    OUTPUT.mkdir(parents=True, exist_ok=False)
    result = dict(
        complete=False,
        passed=False,
        ready_for_model=False,
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
        summarize(baseline, mode="original", phase="producer", config=item["config"])
        group("projection_producer", [register("pose_bounded", "producer", baseline_ref, None)])
        producers, refs, summaries = {}, {}, {}
        for mode in item["config"]["modes"]:
            refs[mode] = prefix.reference(OUTPUT / mode / "producer/simulation/result.json")
            producers[mode] = prefix.read(refs[mode])
            summaries[mode] = {
                "producer": summarize(producers[mode], mode=mode, phase="producer", config=item["config"])
            }
        group("checkers", [register(mode, "checker", baseline_ref, refs[mode]) for mode in item["config"]["modes"]])
        checker_refs, captures = {}, {}
        for mode in item["config"]["modes"]:
            checker_refs[mode] = prefix.reference(OUTPUT / mode / "checker/simulation/result.json")
            checker = prefix.read(checker_refs[mode])
            summaries[mode]["checker"] = summarize(checker, mode=mode, phase="checker", config=item["config"])
            previous.check_pair(producers[mode], checker)
            if summaries[mode]["producer"]["periods"] == 2:
                captures[mode] = compare_capture(producers[mode], checker)
        total = sum(
            summary[phase]["native_steps"] for summary in summaries.values() for phase in ("producer", "checker")
        )
        new = sum(summary["producer"]["newly_selected_commands"] for summary in summaries.values())
        prefix.require(total <= 12000 and new <= 4, "The stage-A physical budget must hold")
        prefix.require(
            source_refs() == sources and snapshot(ROOT)["sources"] == before, "Stage-A source identity changed"
        )
        for path, digest in item["source_sha256"].items():
            prefix.checked({"path": path, "sha256": digest})
        result.update(
            complete=True,
            passed=True,
            sources_unchanged=True,
            ready_for_model=len(captures) == 2,
            producer_refs=refs,
            checker_refs=checker_refs,
            captures=captures,
            summaries=summaries,
            total_native_steps=total,
            newly_selected_commands=new,
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
