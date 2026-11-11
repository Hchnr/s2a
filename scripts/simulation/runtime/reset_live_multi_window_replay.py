"""Independently replay actual commands across current-image WAM windows."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from s2a.references.action_reference import decode_action_targets
from s2a.research.native_runtime.development_continuation import run_continuation
from s2a.research.native_runtime.endpoint_cache import SameStateEndpointCache
from s2a.research.native_runtime.reset_canvas_threads import render_with_two_torch_threads
from s2a.research.native_runtime.streaming_continuation import measured_snapshot
from s2a.research.wam.tracking.runtime.causal_stream_protocol import CausalStreamLedger
from s2a.research.wam.training.native.native_training import state_sha256
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import reset_live_two_window_replay as previous
from scripts.simulation.runtime import run_reset_start_feedback as original
from scripts.simulation.runtime import run_segmented_reset_equivalence as isolation
from scripts.simulation.runtime.reset_live_two_window_replay_v2 import minimal_renderer_endpoint
from scripts.wam.tracking.preview import gpu_resume_repro_20261111 as repro


def decode_current_response(ledger, env, sample, capture, exchange, model_response, tool):
    """Bind the saved response to the checker's current state, image and model output."""
    step = ledger.current["step"]
    current = measured_snapshot(env, step)
    binding = {"sim_time": current["sim_time_s"], "sim_state_sha256": current["state_sha256"]}
    request, response = prefix.read(exchange["request_ref"]), prefix.read(exchange["response_ref"])
    prefix.require(
        current == ledger.current
        and capture["frame_index"] == exchange["step"] == step
        and capture["render_before"] == capture["render_after"] == binding
        and capture["canvas_png"]["sha256"] == request["image"]["sha256"]
        and all(
            exchange.get(key) is True
            for key in ("request_sent", "response_received", "response_accepted", "policy_ready")
        ),
        "The saved response must follow the independently observed current image and state",
    )
    image_path = prefix.REPO / ledger.image_root / request["image"]["path"]
    prefix.require(
        str(image_path) == capture["canvas_png"]["path"], "The request must use the producer's recorded image"
    )
    prefix.require(
        ledger.issue(before_render=current, after_render=current, image=request["image"]) == request,
        "The request differs from the independent causal ledger",
    )
    action = ledger.accept(response, after_wait=measured_snapshot(env, step))
    arrays = repro.response_arrays(model_response)
    np.testing.assert_array_equal(action, arrays["action"])
    np.testing.assert_array_equal(response["normalized_action"], arrays["normalized_action"])
    anchor = np.asarray(sample["source_pose"], dtype=float) @ np.asarray(tool, dtype=float)
    return decode_action_targets(anchor, action, horizon=40)


def run_checker(
    env,
    terminal,
    observers,
    receipt,
    *,
    case,
    producer,
    registration,
    model_responses,
    runtime_output,
    progress=None,
    state_reader=None,
    task_observer=None,
    gripper_score=None,
    render_scope=render_with_two_torch_threads,
):
    """Replay actual dispatches from reset; never load the candidate or a model."""
    isolation.checker_isolation()
    gripper_score = original.measured_gripper if gripper_score is None else gripper_score
    runtime = producer["runtime"]
    old = runtime["continuation"]
    first, stream = old["first_window"]["periods"], old["stream"]
    segments, captures, exchanges = (
        stream["segments"],
        runtime["streaming_captures"],
        runtime["streaming_model_exchanges"],
    )
    prefix.require(
        registration["max_requests"] == 5
        and len(first) == 8
        and 0 <= len(segments) <= 5
        and len(segments) == len(captures) == len(exchanges) == len(model_responses)
        and stream["requests_issued"] == stream["responses_accepted"] == len(segments)
        and stream["automatic_retry"] is False,
        "One first window and at most five complete exchanges are required",
    )
    prefix.require(
        (stream["complete"] is True and len(segments) == 5 and stream["stop"] is None)
        or (stream["complete"] is False and stream["stop"] is not None),
        "The producer must record either completion or an explicit stop",
    )
    cache = SameStateEndpointCache(observers.endpoint)
    receipt["endpoint_cache_events"] = cache.events
    wrapped = SimpleNamespace(
        endpoint=cache,
        current=observers.current,
        render_anchor=observers.render_anchor,
        render_cameras=observers.render_cameras,
    )

    def score_expected(expected, openings, ledger=None):
        def completed(actual_env, _command, offset):
            period = receipt["continuation_partial"]["periods"][-1]
            endpoint = period["endpoint_receipt"]
            step = period["step"] + 1
            snapshot = measured_snapshot(actual_env, step)
            prefix.require(
                endpoint == expected[offset]["endpoint_receipt"]
                and endpoint["endpoint_step"] == step
                and endpoint["sim_time"] == snapshot["sim_time_s"]
                and endpoint["sim_state_sha256"] == snapshot["state_sha256"]
                and state_sha256(period["actual_state"]) == snapshot["state_sha256"],
                "The independently observed endpoint differs from the dispatched record",
            )
            score = gripper_score(actual_env, openings[offset])
            original.compare_score(score, expected[offset]["completed_feedback"]["score"])
            if ledger is not None:
                ledger.complete_period(before=ledger.current, after=snapshot, inner_samples=len(period["actual_inner"]))
            return dict(score=score, policy_history=None, policy_history_recomputed=False)

        return completed

    first_replay = run_continuation(
        env,
        terminal,
        wrapped,
        receipt,
        start_step=0,
        previous_command=np.zeros(7),
        targets=np.asarray(case["chunk"]["effective_targets_tcp"][:8]),
        tool=case["tool"],
        target_object=case["target_object"],
        choose_action=original.checker_action(first),
        stop_on_outside=True,
        expected_periods=first,
        on_completed=score_expected(first, case["chunk"]["openings"]),
        state_reader=state_reader,
        task_observer=task_observer,
        progress=progress,
    )
    receipt["reset_first_window"] = first_replay
    ledger = CausalStreamLedger(**registration)
    prefix.require(measured_snapshot(env, 8) == ledger.current, "The replayed reset must reach the registered endpoint")
    replays, current_captures, scopes = [], [], []
    receipt["checker_stream_captures"] = current_captures
    receipt["checker_stream_canvas_scopes"] = scopes
    previous_command = first[-1]["command"]
    for index, (segment, capture, exchange, model_response) in enumerate(
        zip(segments, captures, exchanges, model_responses, strict=True)
    ):
        step = 8 + 8 * index
        expected = segment["continuation"]["periods"]
        prefix.require(
            ledger.current["step"] == segment["start_step"] == step
            and segment["inflight"] is None
            and 0 <= len(expected) <= 8
            and segment["continuation"]["steps_completed"] == len(expected)
            and (index == len(segments) - 1 or len(expected) == 8),
            "Only consecutive actually completed segment periods may be replayed",
        )
        sample, endpoint = cache(env, step)
        scope = {}
        actual_capture = render_scope(
            observers.render_anchor,
            scope,
            env,
            step,
            sample,
            minimal_renderer_endpoint(endpoint, step),
            Path(runtime_output),
        )
        scopes.append(scope.get("canvas_thread_scope"))
        current_captures.append(actual_capture)
        previous.compare_capture(capture, actual_capture)
        poses, openings = decode_current_response(ledger, env, sample, capture, exchange, model_response, case["tool"])
        replay = run_continuation(
            env,
            {"sample": sample, "endpoint_receipt": endpoint},
            wrapped,
            receipt,
            start_step=step,
            previous_command=previous_command,
            targets=poses[:8],
            tool=case["tool"],
            target_object=case["target_object"],
            choose_action=original.checker_action(expected),
            stop_on_outside=True,
            expected_periods=expected,
            on_completed=score_expected(expected, openings, ledger),
            state_reader=state_reader,
            task_observer=task_observer,
            progress=progress,
        )
        replays.append(replay)
        if expected:
            previous_command = expected[-1]["command"]
    prefix.require(
        ledger.completed_periods == stream["feedback_periods_committed"] == stream["control_steps_completed"]
        and stream["control_steps_attempted"] == stream["control_steps_completed"]
        and (not stream["complete"] or ledger.phase == "done"),
        "The independently replayed physical and model counts must match",
    )
    if stream["stop"] is not None:
        stop = stream["stop"]
        prefix.require(
            type(stop) is dict
            and type(stop.get("reason")) is str
            and bool(stop["reason"])
            and stop.get("next_action_dispatched") is False
            and (
                ("at_step" in stop and "after_step" not in stop and stop["at_step"] == ledger.current["step"])
                or ("after_step" in stop and "at_step" not in stop and stop["after_step"] + 1 == ledger.current["step"])
            ),
            "The reported stop must immediately follow the recorded dispatches",
        )
    isolation.checker_isolation()
    return dict(
        first_window=first_replay,
        replays=replays,
        reported_producer_stop=stream["stop"],
        stop_reason_independently_verified=False,
        actual_model_requests=len(segments),
        actual_stream_periods=ledger.completed_periods,
    )
