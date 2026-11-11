"""Replay the saved 40 actions using an exact-state endpoint cache."""

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from s2a.infrastructure.native_environment import restore_cv2_environment
from s2a.infrastructure.owned_processes import run_owned_workers
from s2a.infrastructure.publication import publish_result as publish
from scripts.maintenance.run_cpu_tests import snapshot
from scripts.simulation.runtime import causal_endpoint_diagnostic_admission as diagnostic_admission
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_causal_endpoint_diagnostic as diagnostic
from scripts.simulation.runtime import run_causal_stream_live_48 as live_physical
from scripts.simulation.runtime import run_causal_stream_recorded_physical as physical
from scripts.simulation.runtime import run_ordered_correct_stream as prior

ROOT = prefix.REPO
CONFIG = ROOT / "configs/tracking/references/causal_stream_endpoint_cache_replay_20261111.json"
PLAN = ROOT / "plans/v0.0.1/tracking/references/causal_stream_endpoint_cache_replay_20261111.md"
OUTPUT = ROOT / "outputs/causal_stream_endpoint_cache_replay_20261111"
QA = ROOT / "outputs/qa/causal_stream_endpoint_cache_replay_20261111/result.json"
DRY = ROOT / "outputs/causal_stream_endpoint_cache_replay_dry_20261111/result.json"
MODULE = "scripts.simulation.runtime.run_causal_endpoint_cache_replay"
FIXED = {
    "schema": "s2a.causal-stream-endpoint-cache-replay.v1",
    "output": "outputs/causal_stream_endpoint_cache_replay_20261111",
    "diagnostic_audit": "outputs/causal_stream_endpoint_audit_20261111/result.json",
    "diagnostic_audit_sha256": "3e425606d17ef7023a4c5203a348c1cfa1047ef71a94b1e9a9a98e3db17758fa",
    "case_index": 8,
    "seed": 42,
    "prefix_steps": 80,
    "replayed_steps": 40,
    "max_native_steps": 3000,
    "capture_step": 120,
    "new_model_initializations": 0,
    "new_model_requests": 0,
    "new_denoise_calls": 0,
    "newly_selected_commands": 0,
    "fresh_cases": 0,
    "worker_timeout_seconds": 1200,
    "automatic_retry": False,
}


def validate_config(config):
    prefix.require(
        type(config) is dict
        and set(config) == set(FIXED)
        and all(type(config[key]) is type(value) and config[key] == value for key, value in FIXED.items()),
        "The exact cached-endpoint physical budget is required",
    )


def load_inputs():
    prefix.environment()
    config_ref = prefix.reference(CONFIG)
    config = prefix.read(config_ref)
    validate_config(config)
    inputs = diagnostic_admission.load_inputs()
    consumed = dict(inputs["source_sha256"])
    audit_ref = {"path": str(ROOT / config["diagnostic_audit"]), "sha256": config["diagnostic_audit_sha256"]}
    audit = prefix.read(audit_ref, consumed)
    prefix.require(
        audit["complete"] is audit["passed"] is audit["reproduced_original_failure"] is True
        and audit["control_qualified"] is False
        and audit["replayed_actions"] == 40
        and audit["replayed_native_steps"] == 3000
        and audit["task_success_tail_periods"] == 33
        and audit["changed_joint"]["name"] == "cream_cheese_1_joint0"
        and all(value == 0 for value in audit["replay_max_difference"].values()),
        "The independently reproduced endpoint failure and all 40 actions are required",
    )
    prefix.checked(audit["owner_ref"], consumed)
    prefix.checked(audit["branch_ref"], consumed)
    prefix.checked(config_ref, consumed)
    prefix.reference(PLAN)
    return {
        "config": config,
        "config_ref": config_ref,
        "audit_ref": audit_ref,
        "base": inputs,
        "source_sha256": consumed,
    }


def input_receipt(inputs):
    return {
        "config_ref": inputs["config_ref"],
        "audit_ref": inputs["audit_ref"],
        "diagnostic_inputs": diagnostic_admission.input_receipt(inputs["base"]),
        "source_sha256": inputs["source_sha256"],
    }


def source_refs():
    names = (
        CONFIG,
        PLAN,
        Path(__file__).resolve(),
        ROOT / "src/s2a/research/native_runtime/endpoint_cache.py",
        ROOT / "tests/research/native_runtime/test_endpoint_cache.py",
        ROOT / "tests/research/native_runtime/test_endpoint_cache_stream.py",
        ROOT / "tests/research/native_runtime/test_causal_endpoint_cache_replay.py",
    )
    return diagnostic.source_refs() + [prefix.reference(path) for path in names]


def dry_run():
    inputs = load_inputs()
    print(
        publish(
            DRY,
            {
                "complete": True,
                "passed": True,
                "inputs": input_receipt(inputs),
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
        "Complete 32-worker cached-endpoint CPU QA is required",
    )
    before = snapshot(ROOT)["sources"]
    prefix.require(before == prefix.read(qa["source_before_ref"])["sources"], "Cached endpoint QA sources changed")
    prefix.require(
        dry["complete"] is dry["passed"] is True
        and dry["inputs"] == input_receipt(inputs)
        and dry["source_refs"] == source_refs()
        and dry["new_model_requests"] == dry["new_denoise_calls"] == dry["new_physics_steps"] == 0,
        "Matching zero-work cached-endpoint admission required",
    )
    return before


def worker_inputs():
    prefix.environment()
    registration = prefix.read(prefix.reference(OUTPUT / "registration.json"))
    inputs = load_inputs()
    prefix.require(
        registration["inputs"] == input_receipt(inputs) and registration["source_refs"] == source_refs(),
        "Cached replay worker inputs and sources must match its owner",
    )
    return inputs


def compare_replay(runtime, expected_stream, events, capture):
    replay = runtime["continuation"]
    actual = [period for row in replay["segments"] for period in row["continuation"]["periods"]]
    expected = [period for row in expected_stream["segments"] for period in row["continuation"]["periods"]]
    observed = [event["step"] for event in events if event["kind"] == "observed"]
    reused = [event["step"] for event in events if event["kind"] == "reused"]
    prefix.require(
        runtime["complete"] is runtime["passed"] is runtime["cleanup_complete"] is replay["complete"] is True
        and runtime["control_steps_attempted"] == runtime["control_steps_completed"] == 120
        and runtime["newly_selected_commands"] == 0
        and runtime["limits"]["mj_step"] == 3000
        and runtime["counters"]["mj_step"]["attempted"] == runtime["counters"]["mj_step"]["completed"] == 3000
        and replay["control_steps_attempted"] == replay["control_steps_completed"] == 40
        and len(actual) == len(expected) == 40
        and observed == list(range(81, 121))
        and reused == [88, 96, 104, 112, 120, 120]
        and capture["frame_index"] == 120
        and capture["current_endpoint"]["sim_state_sha256"] == expected[-1]["endpoint_receipt"]["sim_state_sha256"]
        and capture["render_before"] == capture["render_after"]
        and capture["future_rgb_frames"] == 0,
        "The cached replay must complete the saved actions and one unchanged current image capture",
    )
    maxima = {"state": 0.0, "q": 0.0, "qvel": 0.0, "time": 0.0}
    for left, right in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(left["command"], right["command"])
        maxima["state"] = max(
            maxima["state"], float(np.max(np.abs(np.asarray(left["actual_state"]) - right["actual_state"])))
        )
        for name in ("q", "qvel", "time"):
            observed_inner = np.asarray([row[name] for row in left["actual_inner"]])
            expected_inner = np.asarray([row[name] for row in right["actual_inner"]])
            maxima[name] = max(maxima[name], float(np.max(np.abs(observed_inner - expected_inner))))
    prefix.require(all(value <= 1e-10 for value in maxima.values()), "Cached replay differs from actual saved actions")
    return {"completed_actions": 40, "native_steps": 3000, "max_difference": maxima}


def worker():
    inputs = worker_inputs()
    base = inputs["base"]
    old, history = base["live"]["old"]["old"]["base"]["previous"], base["live"]["old"]["old"]["base"]
    directory = OUTPUT / "simulation"
    result = {
        "complete": False,
        "passed": False,
        "control_qualified": False,
        "new_model_requests": 0,
        "new_denoise_calls": 0,
        "newly_selected_commands": 0,
        "automatic_retry": False,
        "endpoint_cache_events": [],
    }
    try:
        prior.checker_isolation()
        cv2 = old["cv2_admission"]
        result["cv2_admission"] = restore_cv2_environment(cv2["cv2_probe_ref"], cv2["cv2_runtime_source_refs"])
        from s2a.research.native_runtime.development_prefix_runtime_48 import PrefixReplayFailure
        from s2a.research.native_runtime.endpoint_cache import SameStateEndpointCache
        from s2a.research.native_runtime.guarded_endpoint_prefix import GuardedEndpointPrefixRuntime
        from s2a.research.native_runtime.streaming_replay_48 import replay_streaming_continuation

        def continuation(env, terminal, observers, receipt, *, progress=None):
            prior.checker_isolation()
            live_physical._check_prefix(receipt, base["branch"]["runtime"]["prefix_inner_periods"])
            cache = SameStateEndpointCache(observers.endpoint)
            result["endpoint_cache_events"] = cache.events
            wrapped = SimpleNamespace(
                endpoint=cache, current=observers.current, render_cameras=observers.render_cameras
            )
            replay = replay_streaming_continuation(
                env,
                terminal,
                wrapped,
                receipt,
                producer_stream=base["stream"],
                target_chunks=base["chunks"],
                source_observations=base["source_observations"],
                max_windows=6,
                previous_command=old["commands"][-1],
                tool=old["tool"],
                target_object=history["config"]["target_object"],
                progress=progress,
            )
            sample, endpoint = cache(env, 120)
            binding = {key: endpoint[key] for key in ("endpoint_step", "sim_time", "sim_state_sha256")}
            before = cache.current_snapshot(env, 120)
            capture = observers.render_anchor(
                env,
                120,
                {"source_pose": sample["source_pose"], "opening": float(sample["opening"])},
                binding,
                directory / "runtime",
            )
            prefix.require(
                cache.current_snapshot(env, 120) == before, "Current image capture must not alter physical state"
            )
            result["capture"] = capture
            receipt.pop("streaming_replay_partial", None)
            return replay

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
            result["runtime"] = error.receipt
            raise
        result["runtime"] = runtime
        result["comparison"] = compare_replay(
            runtime, base["stream"], result["endpoint_cache_events"], result["capture"]
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
    inputs = load_inputs()
    before = _check_qa(inputs)
    sources = source_refs()
    prefix.require(not OUTPUT.exists(), "One fresh cached-endpoint output is required")
    OUTPUT.mkdir()
    directory = OUTPUT / "simulation"
    directory.mkdir()
    registration = publish(OUTPUT / "registration.json", {"inputs": input_receipt(inputs), "source_refs": sources})
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
            "key": "endpoint_cache_replay",
            "argv": [sys.executable, "-m", MODULE, "--worker"],
            "log": OUTPUT / "worker.log",
        }
        with physical._temporary_environment(env):
            lifecycle = run_owned_workers([spec], cwd=ROOT, timeout_seconds=inputs["config"]["worker_timeout_seconds"])
        result["workers_ref"] = publish(OUTPUT / "workers.json", lifecycle)
        physical.historical.registered_feedback_phase.require_closed_workers(lifecycle, ["endpoint_cache_replay"])
        branch_ref = prefix.reference(directory / "result.json")
        branch = prefix.read(branch_ref)
        prefix.require(
            branch["complete"] is branch["passed"] is branch["candidate_modules_unloaded"] is True
            and branch["new_model_requests"] == branch["new_denoise_calls"] == branch["newly_selected_commands"] == 0,
            "A closed policy-isolated cached replay is required",
        )
        prefix.checked(branch["capture"]["canvas_png"])
        for path, digest in inputs["source_sha256"].items():
            prefix.checked({"path": path, "sha256": digest})
        prefix.require(
            snapshot(ROOT)["sources"] == before and source_refs() == sources, "Cached replay sources changed"
        )
        result.update(
            complete=True,
            passed=True,
            sources_unchanged=True,
            branch_ref=branch_ref,
            comparison=branch["comparison"],
            capture_ref=branch["capture"]["canvas_png"],
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
