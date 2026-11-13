"""Run a reset-origin adaptive WAM stream and replay its actual actions."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from s2a.research.native_runtime.development_continuation import current_joint_site_state, run_continuation
from s2a.research.native_runtime.endpoint_cache import SameStateEndpointCache
from s2a.research.native_runtime.reset_canvas_threads import render_with_two_torch_threads
from s2a.research.native_runtime.reset_stream_boundary import first_window_inner
from s2a.research.native_runtime.streaming_continuation import measured_snapshot
from s2a.research.wam.tracking.runtime.adaptive_stream_protocol import AdaptiveStreamLedger
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import reset_live_two_window_replay as capture_check
from scripts.simulation.runtime import run_reset_start_feedback as feedback
from scripts.simulation.runtime import run_segmented_reset_equivalence as isolation
from scripts.simulation.runtime.run_segmented_reset_equivalence import selected_command


def run_producer(
    env,
    terminal,
    observers,
    receipt,
    *,
    case,
    baseline,
    registration,
    channel,
    runtime_output,
    progress=None,
    state_reader=None,
    task_observer=None,
    policy_factory=None,
    first_advance=None,
    next_advance=None,
    trigger=None,
    gripper_score=None,
    render_scope=render_with_two_torch_threads,
):
    """Rebuild eight actual reset periods, then use only new current-image references."""
    from s2a.research.joint_control.preview.causal_reference_replan import (
        advance_recorded_segment,
        completed_quality_margin,
    )
    from s2a.research.native_runtime.adaptive_streaming_continuation import run_adaptive_streaming_continuation
    from s2a.research.wam.tracking.runtime.adaptive_stream_control import close_adaptive_model_stream
    from s2a.research.wam.tracking.runtime.adaptive_stream_reference import AdaptiveStreamingReference

    if policy_factory is None:
        from s2a.research.joint_control.preview.reset_feedback_policy import policy_inputs
        from s2a.research.joint_control.preview.segmented_reset_policy import build_segmented_reset_policy

        def policy_factory(actual_env, actual_case, current, runtime_receipt):
            return build_segmented_reset_policy(actual_env, policy_inputs(actual_case, current), runtime_receipt)

    if first_advance is None:
        from s2a.research.joint_control.preview.segmented_reset_policy import advance_segmented_reset_policy

        first_advance = advance_segmented_reset_policy
    next_advance = advance_recorded_segment if next_advance is None else next_advance
    trigger = completed_quality_margin if trigger is None else trigger
    state_reader = current_joint_site_state if state_reader is None else state_reader
    gripper_score = feedback.measured_gripper if gripper_score is None else gripper_score
    prefix.require(len(baseline) == 8, "The independently checked first eight periods are required")

    cache = SameStateEndpointCache(observers.endpoint)
    receipt["endpoint_cache_events"] = cache.events
    scopes = receipt["adaptive_canvas_thread_scopes"] = []

    def render_anchor(*args, **kwargs):
        scope = {}
        try:
            return render_scope(observers.render_anchor, scope, *args, **kwargs)
        finally:
            scopes.append(scope.get("canvas_thread_scope"))

    wrapped = SimpleNamespace(
        endpoint=cache, current=observers.current, render_anchor=render_anchor, render_cameras=observers.render_cameras
    )
    policy, owner, previous = policy_factory(env, case, terminal, receipt)

    def select(q, limits, jacobian, command, offset):
        return selected_command(policy.select(q, limits, jacobian, command, offset), baseline[offset])

    def complete(actual_env, command, offset):
        period = receipt["continuation_partial"]["periods"][-1]
        history = policy.complete(actual_env, command, offset, [item["q"] for item in period["actual_inner"]])
        score = gripper_score(actual_env, case["chunk"]["openings"][offset])
        feedback.compare_score(score, baseline[offset]["completed_feedback"]["score"])
        return dict(score=score, policy_history=history, policy_history_recomputed=True)

    first = run_continuation(
        env,
        terminal,
        wrapped,
        receipt,
        start_step=0,
        previous_command=previous,
        targets=np.asarray(case["chunk"]["effective_targets_tcp"][:8]),
        tool=case["tool"],
        target_object=case["target_object"],
        choose_action=select,
        stop_on_outside=True,
        expected_periods=baseline,
        on_completed=complete,
        state_reader=state_reader,
        task_observer=task_observer,
        progress=progress,
    )
    receipt["reset_first_window"] = first
    prefix.require(first["steps_completed"] == 8 and first["stop"] is None, "The reset prefix must complete")
    _q, limits, _jacobian = state_reader(env)
    receipt["final_prefix_inner"] = first_window_inner(first["periods"], limits)
    sample, endpoint = cache(env, 8)
    prefix.require(measured_snapshot(env, 8) == registration["initial"], "The actual step-8 state must match")
    ledger = AdaptiveStreamLedger(**registration)

    def first_live_policy(poses, openings, anchor, step):
        prefix.require(step == 8, "The first new reference must begin at the actual step-8 endpoint")
        new = first_advance(policy, owner, poses, openings, anchor, generated_step=step, runtime_receipt=receipt)
        return new, owner

    adapter = AdaptiveStreamingReference(
        env=env,
        observers=wrapped,
        ledger=ledger,
        channel=channel,
        runtime_output=str(Path(runtime_output).relative_to(prefix.REPO)),
        tool=case["tool"],
        receipt=receipt,
        first_policy=first_live_policy,
        response_gate=lambda _request, _response: True,
        advance_policy=next_advance,
        score_gripper=gripper_score,
    )
    stream = run_adaptive_streaming_continuation(
        env,
        {"sample": sample, "endpoint_receipt": endpoint},
        wrapped,
        receipt,
        ledger=ledger,
        previous_command=first["periods"][-1]["command"],
        tool=case["tool"],
        target_object=case["target_object"],
        open_segment=adapter,
        refresh_signal=trigger,
        state_reader=state_reader,
        task_observer=task_observer,
        progress=progress,
    )
    close_adaptive_model_stream(channel, ledger, stream, receipt)
    return dict(first_window=first, adaptive=stream)


def run_checker(
    env,
    terminal,
    observers,
    receipt,
    *,
    case,
    producer,
    responses,
    registration,
    runtime_output,
    progress=None,
    state_reader=None,
    task_observer=None,
    gripper_score=None,
    render_scope=render_with_two_torch_threads,
):
    """Replay only dispatched actions, independently observing each request image."""
    from s2a.research.native_runtime.adaptive_streaming_replay import replay_adaptive_streaming_continuation

    isolation.checker_isolation()
    gripper_score = feedback.measured_gripper if gripper_score is None else gripper_score
    original = producer["runtime"]["continuation"]
    first = original["first_window"]["periods"]
    prefix.require(len(first) == 8, "The completed producer reset prefix is required")
    cache = SameStateEndpointCache(observers.endpoint)
    receipt["endpoint_cache_events"] = cache.events
    wrapped = SimpleNamespace(
        endpoint=cache,
        current=observers.current,
        render_anchor=observers.render_anchor,
        render_cameras=observers.render_cameras,
    )

    def complete(actual_env, _command, offset):
        score = gripper_score(actual_env, case["chunk"]["openings"][offset])
        feedback.compare_score(score, first[offset]["completed_feedback"]["score"])
        return dict(score=score, policy_history=None, policy_history_recomputed=False)

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
        choose_action=feedback.checker_action(first),
        stop_on_outside=True,
        expected_periods=first,
        on_completed=complete,
        state_reader=state_reader,
        task_observer=task_observer,
        progress=progress,
    )
    receipt["reset_first_window"] = replay_first
    sample, endpoint = cache(env, 8)
    prefix.require(measured_snapshot(env, 8) == registration["initial"], "The replayed step-8 state must match")
    ledger = AdaptiveStreamLedger(**registration)
    captures = receipt["adaptive_checker_captures"] = []
    scopes = receipt["adaptive_checker_canvas_thread_scopes"] = []
    expected_captures = producer["runtime"]["adaptive_captures"]

    def capture_current(actual_env, observation, index):
        snapshot = observation["snapshot"]
        step = snapshot["step"]
        identity = dict(endpoint_step=step, sim_time=snapshot["sim_time_s"], sim_state_sha256=snapshot["state_sha256"])
        observed = {"source_pose": observation["source_pose"], "opening": observation["opening"]}
        scope = {}
        try:
            capture = render_scope(observers.render_anchor, scope, actual_env, step, observed, identity, runtime_output)
        finally:
            scopes.append(scope.get("canvas_thread_scope"))
        capture_check.compare_capture(expected_captures[index], capture)
        captures.append(capture)
        image = capture["canvas_png"]
        path = Path(image["path"])
        root = prefix.REPO / registration["image_root"]
        prefix.require(path.is_absolute() and path.is_relative_to(root), "The checker capture root must match")
        return dict(path=str(path.relative_to(root)), sha256=image["sha256"])

    result = replay_adaptive_streaming_continuation(
        env,
        {"sample": sample, "endpoint_receipt": endpoint},
        wrapped,
        receipt,
        ledger=ledger,
        producer_stream=original["adaptive"],
        responses=responses,
        capture_current=capture_current,
        previous_command=first[-1]["command"],
        tool=case["tool"],
        target_object=case["target_object"],
        score_gripper=gripper_score,
        compare_score=feedback.compare_score,
        state_reader=state_reader,
        task_observer=task_observer,
        progress=progress,
    )
    isolation.checker_isolation()
    return dict(first_window=replay_first, adaptive=result)
