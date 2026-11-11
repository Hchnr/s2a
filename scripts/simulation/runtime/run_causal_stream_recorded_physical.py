"""Reproduce two saved-reference windows through owned CPU stream workers."""

import argparse
import os
import socket
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from s2a.infrastructure.framed_channel import FramedJSONChannel
from s2a.infrastructure.lifecycle.owned_socket_pair import OwnedSocketPairInterrupted, run_owned_socket_pair
from s2a.infrastructure.native_environment import restore_cv2_environment
from s2a.infrastructure.owned_processes import run_owned_workers
from s2a.infrastructure.publication import publish_result as publish
from scripts.maintenance.run_cpu_tests import snapshot
from scripts.simulation.runtime import causal_stream_recorded_admission as admission
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_causal_event_feedback as common
from scripts.simulation.runtime import run_causal_wam_two_window_feedback as historical
from scripts.simulation.runtime import run_ordered_correct_stream as prior

ROOT, OUTPUT, QA, DRY = admission.ROOT, admission.OUTPUT, admission.QA, admission.DRY


def source_refs():
    names = (
        admission.CONFIG,
        admission.PLAN,
        Path(__file__).resolve(),
        ROOT / "scripts/simulation/runtime/causal_stream_recorded_admission.py",
        ROOT / "src/s2a/infrastructure/framed_channel.py",
        ROOT / "src/s2a/infrastructure/lifecycle/owned_socket_pair.py",
        ROOT / "src/s2a/research/native_runtime/streaming_continuation.py",
        ROOT / "src/s2a/research/native_runtime/streaming_replay.py",
        ROOT / "src/s2a/research/wam/tracking/runtime/causal_stream_protocol.py",
        ROOT / "src/s2a/research/wam/tracking/runtime/causal_stream_reference.py",
        ROOT / "src/s2a/research/wam/tracking/runtime/recorded_reference_relay.py",
        ROOT / "tests/research/native_runtime/test_causal_stream_recorded_admission.py",
        ROOT / "tests/research/native_runtime/test_causal_stream_recorded_physical.py",
        ROOT / "tests/research/wam/test_recorded_reference_relay.py",
    )
    return historical.source_refs() + [prefix.reference(path) for path in names]


def model_config(inputs, image_root):
    first = inputs["records"][0]
    return {
        "session": admission.SESSION,
        "initial": {"step": first["step"], "sim_time_s": first["sim_time_s"], "state_sha256": first["state_sha256"]},
        "image_root": str(image_root.relative_to(ROOT)),
        "task": inputs["config"]["task"],
        "seed": inputs["config"]["seed"],
        "model_sha256": inputs["model_sha256"],
        "max_requests": inputs["config"]["max_local_reference_requests"],
        "denoise_steps": inputs["config"]["historical_denoise_calls_per_reference"],
    }


def dry_run():
    inputs = admission.load_inputs()
    result = {
        "complete": True,
        "passed": True,
        "inputs": admission.input_receipt(inputs),
        "source_refs": source_refs(),
        "new_model_initializations": 0,
        "new_model_requests": 0,
        "new_denoise_calls": 0,
        "new_physics_steps": 0,
        "fresh_cases": 0,
        "automatic_retry": False,
    }
    print(publish(DRY, result), flush=True)


def check_admission(inputs):
    qa, dry = prefix.read(prefix.reference(QA)), prefix.read(prefix.reference(DRY))
    prefix.require(
        qa["complete"] is qa["passed"] is qa["all_tests_passed"] is qa["source_unchanged"] is True
        and qa["actual_wait"] is qa["reaped"] is True
        and qa["signal_sent"] is False
        and qa["skipped_count"] == qa["xfail_count"] == qa["xpass_count"] == 0
        and not qa["collection"]["collection_errors"]
        and qa["configured_parallelism"] == 64,
        "Complete 64-worker CPU QA without skipped or uncollected tests is required",
    )
    before = snapshot(ROOT)["sources"]
    prefix.require(before == prefix.read(qa["source_before_ref"])["sources"], "QA source identity changed")
    prefix.require(
        dry["complete"] is dry["passed"] is True
        and dry["inputs"] == admission.input_receipt(inputs)
        and dry["source_refs"] == source_refs()
        and dry["new_physics_steps"] == dry["new_model_requests"] == 0,
        "A matching zero-physics dry admission is required",
    )
    return before


def worker_inputs(phase):
    prefix.environment()
    registration = prefix.read(prefix.reference(OUTPUT / phase / "registration.json"))
    inputs = admission.load_inputs()
    prefix.require(
        registration["phase"] == phase
        and registration["inputs"] == admission.input_receipt(inputs)
        and registration["source_refs"] == source_refs(),
        "Worker inputs and source identity must match the owner registration",
    )
    return inputs, registration


def serve_recorded(channel, inputs, result, *, image_root):
    from s2a.research.wam.tracking.runtime.causal_stream_protocol import CausalModelAdmission
    from s2a.research.wam.tracking.runtime.recorded_reference_relay import RecordedReferenceRelay

    service = CausalModelAdmission(**model_config(inputs, image_root))
    relay = RecordedReferenceRelay(service, inputs["records"])
    for _index in range(inputs["config"]["max_local_reference_requests"]):
        request = channel.receive()
        result["local_reference_requests"] += 1
        response, visible = relay.respond(request)
        result["records"] = relay.used
        prefix.require(
            set(visible) == {"image_bytes", "task"} and visible["task"] == inputs["config"]["task"],
            "Only the verified current image and public task are exposed to the relay",
        )
        channel.send(response)
        result["responses_sent"] += 1
    prefix.require(service.phase == "done" and len(relay.used) == 2, "Two saved references must be consumed")


def relay_worker():
    inputs, _registration = worker_inputs("producer")
    output = OUTPUT / "producer/model"
    image_root = OUTPUT / "producer/simulation/runtime"
    result = {
        "complete": False,
        "passed": False,
        "kind": "saved_response_relay",
        "local_reference_requests": 0,
        "responses_sent": 0,
        "new_model_initializations": 0,
        "new_model_requests": 0,
        "new_denoise_calls": 0,
        "historical_denoise_calls_per_reference": 35,
        "records": [],
        "automatic_retry": False,
    }
    channel = None
    try:
        channel = FramedJSONChannel(
            socket.socket(fileno=int(os.environ["S2A_CHANNEL_FD"])), max_bytes=2 * 1024 * 1024, timeout_seconds=1750
        )
        serve_recorded(channel, inputs, result, image_root=image_root)
        result.update(complete=True, passed=True)
    except BaseException as error:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        if channel is not None:
            channel.close()
        print(publish(output / "result.json", result), flush=True)


def simulation_worker():
    inputs, _registration = worker_inputs("producer")
    directory = OUTPUT / "producer/simulation"
    old, base = inputs["old"]["old"]["base"]["previous"], inputs["old"]["old"]["base"]
    result = {"complete": False, "passed": False, "control_qualified": False, "automatic_retry": False}
    channel = None
    try:
        gripper = prior.reconstruct_gripper(base, result)
        cv2 = old["cv2_admission"]
        result["cv2_admission"] = restore_cv2_environment(cv2["cv2_probe_ref"], cv2["cv2_runtime_source_refs"])
        from s2a.research.native_runtime.development_prefix_runtime import PrefixReplayFailure
        from s2a.research.native_runtime.guarded_correct_prefix import GuardedCorrectTask2Seed42PrefixRuntime
        from s2a.research.native_runtime.streaming_continuation import run_streaming_continuation
        from s2a.research.wam.tracking.runtime.causal_stream_protocol import CausalStreamLedger
        from s2a.research.wam.tracking.runtime.causal_stream_reference import CausalStreamingReference

        channel = FramedJSONChannel(
            socket.socket(fileno=int(os.environ["S2A_CHANNEL_FD"])), max_bytes=2 * 1024 * 1024, timeout_seconds=90
        )

        def continuation(env, terminal, observers, receipt, *, progress=None):
            prefix_periods = inputs["producer"]["runtime"]["prefix_inner_periods"]
            prefix.require(
                len(receipt["prefix_inner_periods"]) == len(prefix_periods) == 80, "All prefix samples required"
            )
            for actual, expected in zip(receipt["prefix_inner_periods"], prefix_periods, strict=True):
                common.check_prefix_inner(actual, expected)
            runtime_output = directory / "runtime"
            ledger = CausalStreamLedger(**model_config(inputs, runtime_output))

            def first_policy(poses, openings, anchor, step):
                prefix.require(step == 80, "The initial policy must start at the registered endpoint")
                np.testing.assert_allclose(anchor, inputs["old"]["anchors"][0], atol=1e-10, rtol=0)
                np.testing.assert_allclose(poses, inputs["old"]["targets"][0], atol=1e-12, rtol=0)
                np.testing.assert_array_equal(openings, inputs["old"]["openings"][0])
                data = dict(base, first_anchor=anchor, targets=[poses], openings=[openings])
                return prior.make_initial_policy(env, data, gripper, receipt)

            adapter = CausalStreamingReference(
                env=env,
                observers=observers,
                ledger=ledger,
                channel=channel,
                runtime_output=str(runtime_output.relative_to(ROOT)),
                tool=old["tool"],
                receipt=receipt,
                first_policy=first_policy,
                response_gate=lambda request, response: admission.gate_response(inputs["records"], request, response),
            )
            stream = run_streaming_continuation(
                env,
                terminal,
                observers,
                receipt,
                ledger=ledger,
                max_windows=2,
                previous_command=old["commands"][-1],
                tool=old["tool"],
                target_object=base["config"]["target_object"],
                open_segment=adapter,
                completed_guard=lambda period: admission.guard_period(inputs["periods"], period),
                progress=progress,
            )
            receipt.pop("streaming_partial", None)
            return stream

        try:
            runtime = GuardedCorrectTask2Seed42PrefixRuntime().run(
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
                continuation_budget=16,
                continuation_limit=16,
                registered_seed=42,
                capture_final_prefix_inner=True,
            )
        except PrefixReplayFailure as error:
            result["runtime"] = error.receipt
            raise
        result.update(complete=True, passed=True, runtime=runtime)
    except BaseException as error:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        if channel is not None:
            channel.close()
        print(publish(directory / "result.json", result), flush=True)


def checker_worker():
    inputs, registration = worker_inputs("checker")
    directory = OUTPUT / "checker/simulation"
    old, base = inputs["old"]["old"]["base"]["previous"], inputs["old"]["old"]["base"]
    result = {"complete": False, "passed": False, "control_qualified": False, "automatic_retry": False}
    try:
        prior.checker_isolation()
        producer = prefix.read(registration["producer_branch_ref"])
        prefix.require(producer["complete"] is producer["passed"] is True, "A closed stream producer is required")
        cv2 = old["cv2_admission"]
        result["cv2_admission"] = restore_cv2_environment(cv2["cv2_probe_ref"], cv2["cv2_runtime_source_refs"])
        from s2a.research.native_runtime.development_prefix_runtime import PrefixReplayFailure
        from s2a.research.native_runtime.guarded_correct_prefix import GuardedCorrectTask2Seed42PrefixRuntime
        from s2a.research.native_runtime.streaming_replay import replay_streaming_continuation

        def continuation(env, terminal, observers, receipt, *, progress=None):
            prior.checker_isolation()
            expected_prefix = producer["runtime"]["prefix_inner_periods"]
            prefix.require(
                len(receipt["prefix_inner_periods"]) == len(expected_prefix) == 80, "All prefix samples required"
            )
            for actual, expected in zip(receipt["prefix_inner_periods"], expected_prefix, strict=True):
                common.check_prefix_inner(actual, expected)
            targets = [np.asarray(value[:8]) for value in inputs["old"]["targets"]]
            replay = replay_streaming_continuation(
                env,
                terminal,
                observers,
                receipt,
                producer_stream=producer["runtime"]["continuation"],
                target_chunks=targets,
                max_windows=2,
                previous_command=old["commands"][-1],
                tool=old["tool"],
                target_object=base["config"]["target_object"],
                progress=progress,
            )
            receipt.pop("streaming_replay_partial", None)
            return replay

        try:
            runtime = GuardedCorrectTask2Seed42PrefixRuntime().run(
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
                continuation_budget=16,
                continuation_limit=16,
                registered_seed=42,
                capture_final_prefix_inner=True,
            )
        except PrefixReplayFailure as error:
            result["runtime"] = error.receipt
            raise
        prior.checker_isolation()
        result.update(complete=True, passed=True, runtime=runtime, candidate_modules_unloaded=True)
    except BaseException as error:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        print(publish(directory / "result.json", result), flush=True)


def summarize_branch(branch, *, phase):
    runtime = branch["runtime"]
    stream = runtime["continuation"]
    periods = [period for row in stream["segments"] for period in row["continuation"]["periods"]]
    prefix.require(
        branch["complete"] is branch["passed"] is runtime["complete"] is runtime["passed"] is True
        and runtime["cleanup_complete"] is stream["complete"] is True
        and runtime["control_steps_completed"] == 96
        and runtime["counters"]["mj_step"]["completed"] == 2400
        and runtime["prefix_control_steps"] == 80
        and len(periods) == 16
        and sum(len(row["samples"]) for row in runtime["prefix_inner_periods"]) == 2000
        and [row["step"] for row in periods] == list(range(80, 96))
        and all(len(row["actual_inner"]) == 25 and row["inner_below_buffer_samples"] == 0 for row in periods),
        "A fully closed, safe 80+16-period stream is required",
    )
    if phase == "producer":
        exchanges = runtime["streaming_model_exchanges"]
        segments = runtime["segment_receipts"]
        quality = sum(item["quality_refiner"]["quality_response_evaluations"] for item in segments)
        baseline = sum(item["quality_refiner"]["baseline_response_evaluations"] for item in segments)
        gripper = sum(item["gripper_proposals"]["candidate_sequences"] for item in segments)
        prefix.require(
            stream["requests_issued"] == stream["responses_accepted"] == 2
            and stream["feedback_periods_committed"] == 16
            and [item["frame_index"] for item in runtime["streaming_captures"]] == [80, 88]
            and len(exchanges) == 2
            and all(
                item["request_sent"]
                and item["response_received"]
                and item["response_accepted"]
                and item["policy_ready"]
                for item in exchanges
            )
            and runtime["newly_selected_commands"] == 16
            and quality <= 16 * 160
            and baseline <= 16 * 480
            and gripper <= 16 * 6561,
            "Both actual observations and completed feedback windows are required",
        )
    else:
        prefix.require(
            branch["candidate_modules_unloaded"] is True and runtime["newly_selected_commands"] == 0,
            "The checker must replay without loading candidate policies",
        )
    return {
        "control_steps": 96,
        "native_steps": 2400,
        "continuation_steps": 16,
        "prefix_inner_samples": 2000,
        "continuation_inner_samples": 400,
    }


def _cpu_environment(directory):
    temp, cache = directory / "tmp", directory / "cache"
    temp.mkdir()
    cache.mkdir()
    return {
        "CUDA_VISIBLE_DEVICES": "",
        "LD_LIBRARY_PATH": "",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": "src:.",
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "LP_NUM_THREADS": "1",
        "TMPDIR": str(temp),
        "XDG_CACHE_HOME": str(cache),
    }


@contextmanager
def _temporary_environment(values):
    previous = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def execute(phase):
    prefix.environment()
    prefix.require(phase in ("producer", "checker"), "Known registered phase required")
    inputs = admission.load_inputs()
    before = check_admission(inputs)
    sources = source_refs()
    producer_ref = None
    if phase == "checker":
        prior_result = prefix.read(prefix.reference(OUTPUT / "producer/result.json"))
        prefix.require(
            prior_result["complete"] is prior_result["passed"] is prior_result["sources_unchanged"] is True,
            "A completed producer and actual worker lifecycle are required",
        )
        producer_ref = prior_result["branch_ref"]
    directory = OUTPUT / phase
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "simulation").mkdir()
    registration = publish(
        directory / "registration.json",
        {
            "phase": phase,
            "inputs": admission.input_receipt(inputs),
            "source_refs": sources,
            "producer_branch_ref": producer_ref,
        },
    )
    result = {
        "complete": False,
        "passed": False,
        "phase": phase,
        "control_qualified": False,
        "registration_ref": registration,
        "new_model_initializations": 0,
        "new_model_requests": 0,
        "new_denoise_calls": 0,
        "fresh_cases": 0,
        "automatic_retry": False,
    }
    try:
        simulation_env = _cpu_environment(directory / "simulation")
        command = [sys.executable, "-m", "scripts.simulation.runtime.run_causal_stream_recorded_physical"]
        if phase == "producer":
            (directory / "model").mkdir()
            model_env = _cpu_environment(directory / "model")
            specs = {
                "simulation": {
                    "argv": command + ["--worker", "simulation"],
                    "env": simulation_env,
                    "log": str((directory / "simulation.log").relative_to(ROOT)),
                },
                "model": {
                    "argv": command + ["--worker", "relay"],
                    "env": model_env,
                    "log": str((directory / "model.log").relative_to(ROOT)),
                },
            }
            try:
                lifecycle = run_owned_socket_pair(specs, timeout_seconds=inputs["config"]["worker_timeout_seconds"])
            except OwnedSocketPairInterrupted as error:
                result["workers_ref"] = publish(directory / "workers.json", error.result)
                result["interrupted"] = True
                raise
            lifecycle_ref = publish(directory / "workers.json", lifecycle)
            result["workers_ref"] = lifecycle_ref
            prefix.require(lifecycle["passed"] is True, "Both owned stream children must exit and be reaped")
            relay_ref = prefix.reference(directory / "model/result.json")
            relay = prefix.read(relay_ref)
            prefix.require(
                relay["complete"] is relay["passed"] is True
                and relay["local_reference_requests"] == 2
                and relay["responses_sent"] == 2
                and relay["new_model_requests"] == relay["new_denoise_calls"] == 0,
                "Exactly two saved responses and no new model work required",
            )
            prefix.require(
                [row["historical_response_sha256"] for row in relay["records"]]
                == [row["response_sha256"] for row in inputs["records"]]
                and all(
                    row["historical_denoise_calls"] == 35 and row["new_model_requests"] == row["new_denoise_calls"] == 0
                    for row in relay["records"]
                ),
                "The relay must preserve both verified historical response identities",
            )
            result["relay_ref"] = relay_ref
        else:
            spec = {"key": "checker", "argv": command + ["--worker", "checker"], "log": directory / "checker.log"}
            with _temporary_environment(simulation_env):
                lifecycle = run_owned_workers(
                    [spec], cwd=ROOT, timeout_seconds=inputs["config"]["worker_timeout_seconds"]
                )
            result["workers_ref"] = publish(directory / "workers.json", lifecycle)
            historical.registered_feedback_phase.require_closed_workers(lifecycle, ["checker"])
        branch_ref = prefix.reference(directory / "simulation/result.json")
        branch = prefix.read(branch_ref)
        counts = summarize_branch(branch, phase=phase)
        for path, digest in inputs["source_sha256"].items():
            prefix.checked({"path": path, "sha256": digest})
        prefix.require(
            source_refs() == sources and snapshot(ROOT)["sources"] == before, "Sources changed during execution"
        )
        result.update(complete=True, passed=True, sources_unchanged=True, branch_ref=branch_ref, counts=counts)
    finally:
        print(publish(directory / "result.json", result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true")
    group.add_argument("--producer", action="store_true")
    group.add_argument("--checker", action="store_true")
    group.add_argument("--worker", choices=("simulation", "relay", "checker"))
    args = parser.parse_args()
    if args.dry_run:
        dry_run()
    elif args.producer:
        execute("producer")
    elif args.checker:
        execute("checker")
    else:
        {"simulation": simulation_worker, "relay": relay_worker, "checker": checker_worker}[args.worker]()


if __name__ == "__main__":
    main()
