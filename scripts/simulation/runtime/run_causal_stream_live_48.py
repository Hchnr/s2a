"""Run and independently replay the registered long live stream."""

import os
import socket
from pathlib import Path

import numpy as np

from s2a.infrastructure.framed_channel import FramedJSONChannel
from s2a.infrastructure.native_environment import restore_cv2_environment
from s2a.infrastructure.publication import publish_result as publish
from scripts.simulation.runtime import causal_stream_live_48_replay as references
from scripts.simulation.runtime import causal_stream_recorded_admission as historical
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_causal_event_feedback as common
from scripts.simulation.runtime import run_causal_stream_recorded_physical as physical
from scripts.simulation.runtime import run_ordered_correct_stream as prior


def gate_response(records, request, response):
    index = request.get("request_id")
    prefix.require(type(index) is int and 0 <= index < 6, "One registered current response is required")
    if index < 2:
        return historical.gate_response(records, request, response)
    return True


def guard_period(periods, period):
    step = period.get("step")
    prefix.require(type(step) is int and 80 <= step < 128, "One registered completed period is required")
    return historical.guard_period(periods, period) if step < 96 else True


def close_model_stream(channel, ledger, stream, receipt):
    """Send only a generic stop, and only while the model expects requests."""
    from s2a.research.wam.tracking.runtime.causal_stream_control import make_stop

    stop = stream["stop"]
    if stream["complete"]:
        prefix.require(stop is None and ledger.issued_requests == 6, "A complete stream must exhaust its budget")
        return
    prefix.require(
        type(stop) is dict
        and stop.get("next_action_dispatched") is False
        and stop.get("at_step") == ledger.current["step"]
        and stream["requests_issued"] == ledger.issued_requests,
        "A stop must follow the actually completed endpoint",
    )
    if ledger.issued_requests == 6:
        receipt["stream_stop_frame"] = {"required": False, "reason": "model_request_budget_exhausted"}
        return
    frame = make_stop(session=ledger.session, next_step=ledger.current["step"])
    receipt["stream_stop_frame"] = {"required": True, "frame": frame, "send_attempted": True, "sent": False}
    channel.send(frame)
    receipt["stream_stop_frame"]["sent"] = True


def _run_prefix(old, directory, continuation):
    from s2a.research.native_runtime.guarded_stream_prefix import GuardedStreamPrefixRuntime

    return GuardedStreamPrefixRuntime().run(
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
        continuation_budget=48,
        continuation_limit=48,
        registered_seed=42,
        capture_final_prefix_inner=True,
    )


def _check_prefix(receipt, expected):
    actual = receipt["prefix_inner_periods"]
    prefix.require(len(actual) == len(expected) == 80, "All 80 actual prefix periods are required")
    for left, right in zip(actual, expected, strict=True):
        common.check_prefix_inner(left, right)


def simulation_worker(*, output, loader):
    """Dispatch one bounded physical producer using actual live references."""
    inputs, _registration = loader("producer")
    directory = Path(output) / "producer/simulation"
    old, base = inputs["old"]["old"]["base"]["previous"], inputs["old"]["old"]["base"]
    result = {"complete": False, "passed": False, "control_qualified": False, "automatic_retry": False}
    channel = None
    try:
        gripper = prior.reconstruct_gripper(base, result)
        cv2 = old["cv2_admission"]
        result["cv2_admission"] = restore_cv2_environment(cv2["cv2_probe_ref"], cv2["cv2_runtime_source_refs"])
        from s2a.research.native_runtime.development_prefix_runtime_48 import PrefixReplayFailure
        from s2a.research.native_runtime.streaming_continuation import run_streaming_continuation
        from s2a.research.wam.tracking.runtime.causal_stream_protocol import CausalStreamLedger
        from s2a.research.wam.tracking.runtime.causal_stream_reference import CausalStreamingReference

        channel = FramedJSONChannel(
            socket.socket(fileno=int(os.environ["S2A_CHANNEL_FD"])),
            max_bytes=2 * 1024 * 1024,
            timeout_seconds=900,
        )

        def continuation(env, terminal, observers, receipt, *, progress=None):
            _check_prefix(receipt, inputs["producer"]["runtime"]["prefix_inner_periods"])
            runtime_output = directory / "runtime"
            registration = physical.model_config(inputs, runtime_output)
            registration["max_requests"] = 6
            ledger = CausalStreamLedger(**registration)

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
                runtime_output=str(runtime_output.relative_to(prefix.REPO)),
                tool=old["tool"],
                receipt=receipt,
                first_policy=first_policy,
                response_gate=lambda request, response: gate_response(inputs["records"], request, response),
            )
            stream = run_streaming_continuation(
                env,
                terminal,
                observers,
                receipt,
                ledger=ledger,
                max_windows=6,
                previous_command=old["commands"][-1],
                tool=old["tool"],
                target_object=base["config"]["target_object"],
                open_segment=adapter,
                completed_guard=lambda period: guard_period(inputs["periods"], period),
                progress=progress,
            )
            close_model_stream(channel, ledger, stream, receipt)
            receipt.pop("streaming_partial", None)
            return stream

        try:
            runtime = _run_prefix(old, directory, continuation)
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


def checker_worker(*, output, loader):
    """Decode saved responses and independently replay only completed actions."""
    inputs, registration = loader("checker")
    output = Path(output)
    directory = output / "checker/simulation"
    old, base = inputs["old"]["old"]["base"]["previous"], inputs["old"]["old"]["base"]
    result = {"complete": False, "passed": False, "control_qualified": False, "automatic_retry": False}
    try:
        prior.checker_isolation()
        producer = prefix.read(registration["producer_branch_ref"])
        prefix.require(producer["complete"] is producer["passed"] is True, "A closed producer is required")
        chunks, sources = references.decode_actual_references(
            inputs, producer, image_root=output / "producer/simulation/runtime", max_windows=6
        )
        cv2 = old["cv2_admission"]
        result["cv2_admission"] = restore_cv2_environment(cv2["cv2_probe_ref"], cv2["cv2_runtime_source_refs"])
        from s2a.research.native_runtime.development_prefix_runtime_48 import PrefixReplayFailure
        from s2a.research.native_runtime.streaming_replay_48 import replay_streaming_continuation

        def continuation(env, terminal, observers, receipt, *, progress=None):
            prior.checker_isolation()
            _check_prefix(receipt, producer["runtime"]["prefix_inner_periods"])
            replay = replay_streaming_continuation(
                env,
                terminal,
                observers,
                receipt,
                producer_stream=producer["runtime"]["continuation"],
                target_chunks=chunks,
                source_observations=sources,
                max_windows=6,
                previous_command=old["commands"][-1],
                tool=old["tool"],
                target_object=base["config"]["target_object"],
                progress=progress,
            )
            receipt.pop("streaming_replay_partial", None)
            return replay

        try:
            runtime = _run_prefix(old, directory, continuation)
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
