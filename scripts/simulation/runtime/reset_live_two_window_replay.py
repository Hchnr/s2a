"""Independently replay reset feedback and decode the actual current-image WAM response."""

from types import SimpleNamespace

import numpy as np

from s2a.references.action_reference import decode_action_targets
from s2a.research.native_runtime.development_continuation import run_continuation
from s2a.research.native_runtime.endpoint_cache import SameStateEndpointCache
from s2a.research.native_runtime.reset_canvas_threads import render_with_two_torch_threads
from s2a.research.wam.tracking.runtime.causal_stream_protocol import CausalStreamLedger
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_reset_start_feedback as original
from scripts.simulation.runtime import run_segmented_reset_equivalence as isolation
from scripts.wam.tracking.preview import gpu_resume_repro_20261111 as repro


def compare_capture(expected, actual):
    """Compare endpoint identity and pixels while permitting distinct output paths."""
    keys = (
        "frame_index",
        "current_endpoint",
        "render_before",
        "render_after",
        "future_rgb_frames",
        "camera_keys",
        "canvas_shape",
        "canvas_dtype",
        "canvas_rgb_sha256",
    )
    prefix.require(
        all(expected[key] == actual[key] for key in keys), "The independent current image or endpoint differs"
    )
    prefix.require(expected["canvas_png"]["sha256"] == actual["canvas_png"]["sha256"], "The independent PNG differs")
    for key in ("current_source_pose", "current_opening"):
        np.testing.assert_allclose(expected[key], actual[key], atol=1e-10, rtol=0)


def decode_response(registration, capture, exchange, model_response, tool):
    """Rebuild the ledger and targets from the actual saved request and response."""
    request, response = prefix.read(exchange["request_ref"]), prefix.read(exchange["response_ref"])
    current = registration["initial"]
    binding = {"sim_time": current["sim_time_s"], "sim_state_sha256": current["state_sha256"]}
    prefix.require(
        capture["frame_index"] == exchange["step"] == current["step"] == 8
        and capture["render_before"] == capture["render_after"] == binding
        and capture["canvas_png"]["sha256"] == request["image"]["sha256"]
        and all(
            exchange.get(key) is True
            for key in ("request_sent", "response_received", "response_accepted", "policy_ready")
        ),
        "The response must follow the actual current image and unchanged state",
    )
    root = prefix.REPO / registration["image_root"]
    prefix.require(str(root / request["image"]["path"]) == capture["canvas_png"]["path"], "Request image path differs")
    ledger = CausalStreamLedger(**registration)
    prefix.require(
        ledger.issue(before_render=current, after_render=current, image=request["image"]) == request,
        "Request ledger differs",
    )
    action = ledger.accept(response, after_wait=current)
    arrays = repro.response_arrays(model_response)
    np.testing.assert_array_equal(action, arrays["action"])
    np.testing.assert_array_equal(response["normalized_action"], arrays["normalized_action"])
    anchor = np.asarray(capture["current_source_pose"], dtype=float) @ np.asarray(tool, dtype=float)
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
    model_response,
    runtime_output,
    progress=None,
    state_reader=None,
    task_observer=None,
    gripper_score=None,
    render_scope=render_with_two_torch_threads,
):
    """Replay only actual dispatched commands, independently checking the new capture."""
    isolation.checker_isolation()
    gripper_score = original.measured_gripper if gripper_score is None else gripper_score
    old = producer["runtime"]["continuation"]
    first = old["first_window"]["periods"]
    stream = old["stream"]
    prefix.require(
        len(first) == 8 and len(stream["segments"]) <= 1,
        "One completed initial and at most one live window are required",
    )
    cache = SameStateEndpointCache(observers.endpoint)
    receipt["endpoint_cache_events"] = cache.events
    wrapped = SimpleNamespace(
        endpoint=cache,
        current=observers.current,
        render_anchor=observers.render_anchor,
        render_cameras=observers.render_cameras,
    )

    def score_expected(expected, opening):
        def completed(actual_env, _command, offset):
            score = gripper_score(actual_env, opening[offset])
            original.compare_score(score, expected[offset]["completed_feedback"]["score"])
            return dict(score=score, policy_history=None, policy_history_recomputed=False)

        return completed

    replay_first = run_continuation(
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
    receipt["reset_first_window"] = replay_first
    sample, endpoint = cache(env, 8)
    current_terminal = {"sample": sample, "endpoint_receipt": endpoint}
    if not stream["segments"]:
        prefix.require(
            stream["requests_issued"] == 0 and stream["stop"] is not None,
            "An empty stream requires the recorded pre-request stop",
        )
        return dict(
            first_window=replay_first,
            replay=None,
            reported_producer_stop=stream["stop"],
            stop_reason_independently_verified=False,
        )
    capture = producer["runtime"]["streaming_captures"][0]
    exchange = producer["runtime"]["streaming_model_exchanges"][0]
    scope = {}
    actual_capture = render_scope(observers.render_anchor, scope, env, 8, sample, endpoint, runtime_output)
    receipt["checker_stream_capture"] = actual_capture
    receipt["checker_stream_canvas_scope"] = scope.get("canvas_thread_scope")
    compare_capture(capture, actual_capture)
    poses, openings = decode_response(registration, capture, exchange, model_response, case["tool"])
    expected = stream["segments"][0]["continuation"]["periods"]
    prefix.require(len(expected) <= 8, "At most one new reference window can be replayed")
    replay = run_continuation(
        env,
        current_terminal,
        wrapped,
        receipt,
        start_step=8,
        previous_command=first[-1]["command"],
        targets=poses[:8],
        tool=case["tool"],
        target_object=case["target_object"],
        choose_action=original.checker_action(expected),
        stop_on_outside=True,
        expected_periods=expected,
        on_completed=score_expected(expected, openings),
        state_reader=state_reader,
        task_observer=task_observer,
        progress=progress,
    )
    isolation.checker_isolation()
    return dict(
        first_window=replay_first,
        replay=replay,
        reported_producer_stop=stream["stop"],
        stop_reason_independently_verified=False,
    )
