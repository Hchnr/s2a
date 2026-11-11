"""Run and independently replay two early branches on one WAM reference."""

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
from scripts.simulation.runtime import prepare_early_box_face_feedback as admission
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_box_face_feedback as previous
from scripts.simulation.runtime import run_causal_stream_live_48 as frozen
from scripts.simulation.runtime import run_causal_stream_recorded_physical as physical
from scripts.simulation.runtime import run_early_tabletop_capture as capture
from scripts.simulation.runtime import run_tabletop_projection_feedback_probe as audit
from scripts.simulation.runtime.early_box_face_feedback_continuation import checker_isolation, execute_early_box_face
from scripts.simulation.runtime.tabletop_projection_continuation import compare_branch_start

ROOT, OUTPUT, DRY, QA = admission.ROOT, admission.OUTPUT, admission.DRY, admission.QA
MODULE = "scripts.simulation.runtime.run_early_box_face_feedback"


def source_refs():
    return (
        previous.source_refs()
        + capture.source_refs()
        + [
            prefix.reference(path)
            for path in (
                admission.CONFIG,
                admission.PLAN,
                Path(__file__).resolve(),
                ROOT / "scripts/simulation/runtime/prepare_early_box_face_feedback.py",
                ROOT / "scripts/simulation/runtime/early_box_face_feedback_continuation.py",
                ROOT / "tests/research/native_runtime/test_early_box_face_feedback_admission.py",
                ROOT / "tests/research/native_runtime/test_early_box_face_feedback_continuation.py",
                ROOT / "tests/research/native_runtime/test_early_box_face_inner_recording.py",
                ROOT / "tests/research/native_runtime/test_early_box_face_feedback_owner.py",
            )
        ]
    )


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
    config = inputs["early_box_face_feedback"]["config"]
    prefix.require(
        mode in config["modes"]
        and phase in ("producer", "checker")
        and registration["mode"] == mode
        and registration["phase"] == phase
        and registration["inputs"] == admission.input_receipt(inputs)
        and registration["source_refs"] == source_refs(),
        "The worker identity and sources must match registration",
    )
    producer = None if registration["producer_ref"] is None else prefix.read(registration["producer_ref"])
    prefix.require(
        (phase == "producer" and producer is None)
        or (phase == "checker" and producer is not None and producer["complete"] is producer["passed"] is True),
        "Only the checker may read its closed producer",
    )
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
            return execute_early_box_face(
                env, terminal, observers, receipt, inputs, mode=mode, producer=producer, progress=progress
            )

        try:
            runtime = previous.bounded_prefix(values, directory, continuation, config)
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


def compare_common_reference(first, second):
    """Check the common start and full already-generated target sequence."""
    compare_branch_start(first["branch_start"], second["branch_start"])
    prefix.require(
        first["reference_step"] == second["reference_step"] == 112
        and first["reference_image_ref"] == second["reference_image_ref"]
        and first["reference_response_ref"] == second["reference_response_ref"],
        "The prior current-image WAM response must be identical",
    )
    np.testing.assert_array_equal(first["original_targets"], second["original_targets"])


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
        and branch["mode"] == continuation["mode"] == mode
        and branch["phase"] == phase
        and 0 <= len(periods) <= config["new_period_limit"]
        and len(samples) == 25 * len(periods)
        and runtime["control_steps_completed"] == 118 + len(periods)
        and runtime["counters"]["mj_step"]["completed"] == 25 * runtime["control_steps_completed"]
        and runtime["limits"]["mj_step"] == config["max_native_steps_per_phase"]
        and runtime["counters"]["model_construct"]["completed"] == runtime["counters"]["context"]["completed"] == 3
        and runtime["newly_selected_commands"] == (len(periods) if phase == "producer" else 0)
        and runtime["world_model_calls"] == runtime["fresh_cases"] == 0
        and continuation["branch_start"]["state_sha256"] == config["branch_state_sha256"]
        and continuation["reference_step"] == config["reference_step"]
        and continuation["reference_image_ref"]["sha256"] == config["reference_image_sha256"]
        and continuation["reference_response_ref"]["sha256"] == config["reference_response_sha256"]
        and [period["step"] for period in periods] == list(range(118, 118 + len(periods))),
        "The common source, complete physical samples and hard budgets must hold",
    )
    result = previous.summarize_periods(periods, samples, has_slide_conditions=True)
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
    item, sources, before = inputs["early_box_face_feedback"], source_refs(), snapshot(ROOT)["sources"]
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
        prior_related_native_steps=37050,
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

        modes = item["config"]["modes"]
        group("producers", [register(mode, "producer", None) for mode in modes])
        refs, producers, summaries = {}, {}, {}
        for mode in modes:
            refs[mode] = prefix.reference(OUTPUT / mode / "producer/simulation/result.json")
            producers[mode] = prefix.read(refs[mode])
            summaries[mode] = {
                "producer": summarize(producers[mode], mode=mode, phase="producer", config=item["config"])
            }
        compare_common_reference(*(producers[mode]["runtime"]["continuation"] for mode in modes))
        group("checkers", [register(mode, "checker", refs[mode]) for mode in modes])
        checker_refs = {}
        for mode in modes:
            checker_refs[mode] = prefix.reference(OUTPUT / mode / "checker/simulation/result.json")
            checker = prefix.read(checker_refs[mode])
            summaries[mode]["checker"] = summarize(checker, mode=mode, phase="checker", config=item["config"])
            audit.check_pair(producers[mode], checker)
            compare_common_reference(producers[mode]["runtime"]["continuation"], checker["runtime"]["continuation"])
        n = min(summaries[mode]["producer"]["periods"] for mode in modes)
        matched = {}
        for mode in modes:
            c = producers[mode]["runtime"]["continuation"]
            matched[mode] = previous.summarize_periods(
                c["probe"]["periods"][:n], c["inner_samples"][: 25 * n], has_slide_conditions=True
            )
        total = sum(summaries[mode][phase]["native_steps"] for mode in modes for phase in ("producer", "checker"))
        new = sum(summaries[mode]["producer"]["newly_selected_commands"] for mode in modes)
        prefix.require(
            total <= item["config"]["max_total_native_steps"] and new <= item["config"]["max_new_commands"],
            "The registered total physical budget must hold",
        )
        prefix.require(
            source_refs() == sources and snapshot(ROOT)["sources"] == before, "Sources changed during the comparison"
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
            matched_periods=n,
            matched_summaries=matched,
            new_native_steps=total,
            new_selected_commands=new,
            producer_stop_independently_rechecked=False,
        )
    except BaseException as error:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        partial = []
        for mode in item["config"]["modes"]:
            for phase in ("producer", "checker"):
                path = OUTPUT / mode / phase / "simulation/result.json"
                if path.is_file():
                    ref = prefix.reference(path)
                    branch = prefix.read(ref)
                    runtime = branch.get("runtime", {})
                    partial.append(
                        dict(
                            mode=mode,
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
