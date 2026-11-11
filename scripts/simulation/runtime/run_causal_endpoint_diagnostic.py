"""Independently replay saved actions and measure the step120 endpoint effect."""

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from s2a.infrastructure.native_environment import restore_cv2_environment
from s2a.infrastructure.owned_processes import run_owned_workers
from s2a.infrastructure.publication import publish_result as publish
from scripts.maintenance.run_cpu_tests import snapshot
from scripts.simulation.runtime import causal_endpoint_diagnostic_admission as admission
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_causal_stream_live_48 as live_physical
from scripts.simulation.runtime import run_causal_stream_recorded_physical as physical
from scripts.simulation.runtime import run_ordered_correct_stream as prior

ROOT, OUTPUT, QA, DRY = admission.ROOT, admission.OUTPUT, admission.QA, admission.DRY
MODULE = "scripts.simulation.runtime.run_causal_endpoint_diagnostic"


def source_refs():
    names = (
        admission.CONFIG,
        admission.PLAN,
        Path(__file__).resolve(),
        ROOT / "scripts/simulation/runtime/causal_endpoint_diagnostic_admission.py",
        ROOT / "src/s2a/research/native_runtime/guarded_endpoint_prefix.py",
        ROOT / "src/s2a/research/native_runtime/endpoint_observation_probe.py",
        ROOT / "src/s2a/research/native_runtime/streaming_replay_48.py",
        ROOT / "tests/research/native_runtime/test_causal_endpoint_diagnostic_admission.py",
        ROOT / "tests/research/native_runtime/test_causal_endpoint_diagnostic.py",
        ROOT / "tests/research/native_runtime/test_guarded_endpoint_prefix.py",
        ROOT / "tests/research/native_runtime/test_endpoint_observation_probe.py",
    )
    return [prefix.reference(path) for path in names]


def dry_run():
    inputs = admission.load_inputs()
    print(
        publish(
            DRY,
            {
                "complete": True,
                "passed": True,
                "inputs": admission.input_receipt(inputs),
                "source_refs": source_refs(),
                "new_model_requests": 0,
                "new_denoise_calls": 0,
                "new_physics_steps": 0,
                "newly_selected_commands": 0,
                "automatic_retry": False,
            },
        ),
        flush=True,
    )


def _check_qa(inputs):
    qa, dry = prefix.read(prefix.reference(QA)), prefix.read(prefix.reference(DRY))
    prefix.require(
        qa["complete"] is qa["passed"] is qa["all_tests_passed"] is qa["source_unchanged"] is True
        and qa["actual_wait"] is qa["reaped"] is True
        and qa["signal_sent"] is False
        and qa["skipped_count"] == qa["xfail_count"] == qa["xpass_count"] == 0
        and not qa["collection"]["collection_errors"]
        and qa["configured_parallelism"] == 32,
        "Complete 32-worker diagnostic CPU QA is required",
    )
    before = snapshot(ROOT)["sources"]
    prefix.require(before == prefix.read(qa["source_before_ref"])["sources"], "Diagnostic QA source identity changed")
    prefix.require(
        dry["complete"] is dry["passed"] is True
        and dry["inputs"] == admission.input_receipt(inputs)
        and dry["source_refs"] == source_refs()
        and dry["new_model_requests"] == dry["new_denoise_calls"] == dry["new_physics_steps"] == 0,
        "Matching zero-work diagnostic admission required",
    )
    return before


def worker_inputs():
    prefix.environment()
    registration = prefix.read(prefix.reference(OUTPUT / "registration.json"))
    inputs = admission.load_inputs()
    prefix.require(
        registration["inputs"] == admission.input_receipt(inputs) and registration["source_refs"] == source_refs(),
        "Diagnostic worker sources and inputs must match its owner",
    )
    return inputs


def compare_replay(runtime, expected_stream, events, *, reproduced):
    replay = runtime["streaming_replay_partial"] if reproduced else runtime["continuation"]
    actual_segments, expected_segments = replay["segments"], expected_stream["segments"]
    actual = [period for row in actual_segments for period in row["continuation"]["periods"]]
    expected = [period for row in expected_segments for period in row["continuation"]["periods"]]
    prefix.require(
        runtime["cleanup_complete"] is True
        and runtime["control_steps_attempted"] == runtime["control_steps_completed"] == 120
        and runtime["newly_selected_commands"] == 0
        and runtime["limits"]["mj_step"] == 3000
        and runtime["counters"]["mj_step"]["attempted"] == runtime["counters"]["mj_step"]["completed"] == 3000
        and len(actual_segments) == len(expected_segments) == 5
        and len(actual) == len(expected) == 40
        and replay["control_steps_attempted"] == replay["control_steps_completed"] == 40
        and len(events) == 2
        and all(event["step"] == 120 and event["completed"] is True for event in events),
        "The diagnostic must replay 40 completed actions and capture both step120 endpoint calls",
    )
    maxima = {"state": 0.0, "q": 0.0, "qvel": 0.0, "time": 0.0}
    for actual_period, expected_period in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(actual_period["command"], expected_period["command"])
        maxima["state"] = max(
            maxima["state"],
            float(
                np.max(np.abs(np.asarray(actual_period["actual_state"]) - np.asarray(expected_period["actual_state"])))
            ),
        )
        for name in ("q", "qvel", "time"):
            left = np.asarray([item[name] for item in actual_period["actual_inner"]])
            right = np.asarray([item[name] for item in expected_period["actual_inner"]])
            maxima[name] = max(maxima[name], float(np.max(np.abs(left - right))))
    prefix.require(all(value <= 1e-10 for value in maxima.values()), "Independent replay samples differ")
    if reproduced:
        prefix.require(
            runtime["failure_stage"] == "continuation"
            and runtime["error"]["message"] == "Replay endpoint observation changed the physical state"
            and len(events[-1]["changed_indices"]) > 0,
            "The captured extra endpoint must reproduce the original state change",
        )
    else:
        prefix.require(
            runtime["complete"] is runtime["passed"] is True and replay["complete"] is True, "Replay closure"
        )
    return {"reproduced": reproduced, "completed_actions": 40, "native_steps": 3000, "max_difference": maxima}


def worker():
    inputs = worker_inputs()
    directory = OUTPUT / "simulation"
    old, base = inputs["live"]["old"]["old"]["base"]["previous"], inputs["live"]["old"]["old"]["base"]
    result = {
        "complete": False,
        "passed": False,
        "control_qualified": False,
        "new_model_requests": 0,
        "new_denoise_calls": 0,
        "newly_selected_commands": 0,
        "automatic_retry": False,
        "endpoint_events": [],
    }
    try:
        prior.checker_isolation()
        cv2 = old["cv2_admission"]
        result["cv2_admission"] = restore_cv2_environment(cv2["cv2_probe_ref"], cv2["cv2_runtime_source_refs"])
        from s2a.research.native_runtime.development_prefix_runtime_48 import PrefixReplayFailure
        from s2a.research.native_runtime.endpoint_observation_probe import EndpointObservationProbe
        from s2a.research.native_runtime.guarded_endpoint_prefix import GuardedEndpointPrefixRuntime
        from s2a.research.native_runtime.streaming_replay_48 import replay_streaming_continuation

        def continuation(env, terminal, observers, receipt, *, progress=None):
            prior.checker_isolation()
            live_physical._check_prefix(receipt, inputs["branch"]["runtime"]["prefix_inner_periods"])
            probe = EndpointObservationProbe(observers.endpoint, step=120)
            result["endpoint_events"] = probe.events
            wrapped = SimpleNamespace(
                endpoint=probe, current=observers.current, render_cameras=observers.render_cameras
            )
            replay = replay_streaming_continuation(
                env,
                terminal,
                wrapped,
                receipt,
                producer_stream=inputs["stream"],
                target_chunks=inputs["chunks"],
                source_observations=inputs["source_observations"],
                max_windows=6,
                previous_command=old["commands"][-1],
                tool=old["tool"],
                target_object=base["config"]["target_object"],
                progress=progress,
            )
            receipt.pop("streaming_replay_partial", None)
            return replay

        reproduced = False
        try:
            runtime = GuardedEndpointPrefixRuntime().run(
                old["runtime_config"],
                old["record"],
                old["record_xml"],
                old["commands"],
                old["saved"],
                old["receipts"],
                directory / "runtime",
                old["asset_refs"],
                old["controller"],
                continuation=continuation,
                continuation_budget=40,
                continuation_limit=48,
                registered_seed=42,
                capture_final_prefix_inner=True,
            )
        except PrefixReplayFailure as error:
            runtime = error.receipt
            result["runtime"] = runtime
            if runtime.get("error", {}).get("message") != "Replay endpoint observation changed the physical state":
                raise
            reproduced = True
        result["runtime"] = runtime
        result["comparison"] = compare_replay(
            runtime, inputs["stream"], result["endpoint_events"], reproduced=reproduced
        )
        prior.checker_isolation()
        result.update(complete=True, passed=True, candidate_modules_unloaded=True, source_stop_was_derived=True)
    except BaseException as error:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        print(publish(directory / "result.json", result), flush=True)


def execute():
    prefix.environment()
    inputs = admission.load_inputs()
    before = _check_qa(inputs)
    sources = source_refs()
    prefix.require(not OUTPUT.exists(), "One fresh diagnostic output is required")
    OUTPUT.mkdir()
    directory = OUTPUT / "simulation"
    directory.mkdir()
    registration = publish(
        OUTPUT / "registration.json", {"inputs": admission.input_receipt(inputs), "source_refs": sources}
    )
    result = {
        "complete": False,
        "passed": False,
        "control_qualified": False,
        "registration_ref": registration,
        "new_model_requests": 0,
        "new_denoise_calls": 0,
        "automatic_retry": False,
    }
    try:
        env = physical._cpu_environment(directory)
        spec = {
            "key": "endpoint_diagnostic",
            "argv": [sys.executable, "-m", MODULE, "--worker"],
            "log": OUTPUT / "worker.log",
        }
        with physical._temporary_environment(env):
            lifecycle = run_owned_workers([spec], cwd=ROOT, timeout_seconds=inputs["config"]["worker_timeout_seconds"])
        result["workers_ref"] = publish(OUTPUT / "workers.json", lifecycle)
        physical.historical.registered_feedback_phase.require_closed_workers(lifecycle, ["endpoint_diagnostic"])
        branch_ref = prefix.reference(directory / "result.json")
        branch = prefix.read(branch_ref)
        prefix.require(
            branch["complete"] is branch["passed"] is branch["candidate_modules_unloaded"] is True
            and branch["new_model_requests"] == branch["new_denoise_calls"] == branch["newly_selected_commands"] == 0,
            "A closed, policy-isolated CPU diagnostic is required",
        )
        for path, digest in inputs["source_sha256"].items():
            prefix.checked({"path": path, "sha256": digest})
        prefix.require(snapshot(ROOT)["sources"] == before and source_refs() == sources, "Diagnostic sources changed")
        result.update(
            complete=True,
            passed=True,
            sources_unchanged=True,
            branch_ref=branch_ref,
            comparison=branch["comparison"],
        )
    except BaseException as error:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        branch_path = directory / "result.json"
        if branch_path.is_file():
            result["available_branch_ref"] = prefix.reference(branch_path)
        print(publish(OUTPUT / "result.json", result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true")
    group.add_argument("--execute", action="store_true")
    group.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        dry_run()
    elif args.execute:
        execute()
    else:
        worker()


if __name__ == "__main__":
    main()
