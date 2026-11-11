"""Run bounded reset feedback with five current-image WAM reference windows."""

import time
from types import SimpleNamespace

import numpy as np

from s2a.research.native_runtime.development_continuation import current_joint_site_state, run_continuation
from s2a.research.native_runtime.endpoint_cache import SameStateEndpointCache
from s2a.research.native_runtime.reset_canvas_threads import render_with_two_torch_threads
from s2a.research.native_runtime.reset_stream_boundary import first_window_inner
from s2a.research.native_runtime.streaming_continuation import (
    StreamingSegment,
    measured_snapshot,
    run_streaming_continuation,
)
from s2a.research.wam.tracking.runtime.causal_stream_control import make_stop
from s2a.research.wam.tracking.runtime.causal_stream_protocol import CausalStreamLedger
from s2a.research.wam.tracking.runtime.causal_stream_reference import CausalStreamingReference
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_reset_start_feedback as original
from scripts.simulation.runtime.run_segmented_reset_equivalence import selected_command


def advance_registered_segment(
    previous, owner, poses, openings, anchor, *, generated_step, segment_receipt, runtime_receipt
):
    """Preserve shared solver trace boundaries as the adapter appends its receipt."""
    from s2a.research.joint_control.preview.streaming_feedback_boundary import advance_feedback

    segments, starts = runtime_receipt.get("segment_receipts"), runtime_receipt.get("segment_trace_starts")
    prefix.require(
        type(segments) is list
        and segments
        and segments[-1] is previous.receipt
        and type(starts) is list
        and len(starts) == len(segments),
        "The previous segment must own the latest history and receipt",
    )
    first = segments[0]
    shared, causal = first.get("shared_shift_traces"), first.get("causal_shift_fallback_traces")
    prefix.require(type(shared) is list and type(causal) is list, "Original shared solver trace lists are required")
    boundary = dict(shared=len(shared), causal=len(causal))
    policy = advance_feedback(
        previous, owner, poses, openings, anchor, generated_step=generated_step, receipt=segment_receipt
    )
    starts.append(boundary)
    return policy


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
    gripper_score=None,
    render_scope=render_with_two_torch_threads,
):
    """Use the saved first reference, then five references from actual current images."""
    if policy_factory is None:
        from s2a.research.joint_control.preview.reset_feedback_policy import policy_inputs
        from s2a.research.joint_control.preview.segmented_reset_policy import build_segmented_reset_policy

        def policy_factory(env, case, terminal, receipt):
            return build_segmented_reset_policy(env, policy_inputs(case, terminal), receipt)

    if first_advance is None:
        from s2a.research.joint_control.preview.segmented_reset_policy import advance_segmented_reset_policy

        first_advance = advance_segmented_reset_policy
    next_advance = advance_registered_segment if next_advance is None else next_advance
    state_reader = current_joint_site_state if state_reader is None else state_reader
    gripper_score = original.measured_gripper if gripper_score is None else gripper_score
    prefix.require(
        len(baseline) == 8 and registration["max_requests"] == 5,
        "Eight original periods and five new request slots are required",
    )
    cache = SameStateEndpointCache(observers.endpoint)
    receipt["endpoint_cache_events"] = cache.events
    scopes = receipt["stream_canvas_scopes"] = []

    def render_anchor(*args, **kwargs):
        current_scope = {}
        try:
            return render_scope(observers.render_anchor, current_scope, *args, **kwargs)
        finally:
            scopes.append(current_scope.get("canvas_thread_scope"))

    wrapped = SimpleNamespace(
        endpoint=cache, current=observers.current, render_anchor=render_anchor, render_cameras=observers.render_cameras
    )
    policy, owner, previous = policy_factory(env, case, terminal, receipt)

    def select(q, limits, jacobian, prior, offset):
        started = time.monotonic()
        decision = policy.select(q, limits, jacobian, prior, offset)
        decision["total_decision_seconds"] = time.monotonic() - started
        return selected_command(decision, baseline[offset])

    def complete(actual_env, command, offset):
        period = receipt["continuation_partial"]["periods"][-1]
        history = policy.complete(actual_env, command, offset, [item["q"] for item in period["actual_inner"]])
        score = gripper_score(actual_env, case["chunk"]["openings"][offset])
        original.compare_score(score, baseline[offset]["completed_feedback"]["score"])
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
    prefix.require(first["steps_completed"] == 8 and first["stop"] is None, "The first physical window must complete")
    _q, limits, _jacobian = state_reader(env)
    receipt["final_prefix_inner"] = first_window_inner(first["periods"], limits)
    sample, endpoint = cache(env, 8)
    next_terminal = {"sample": sample, "endpoint_receipt": endpoint}
    prefix.require(
        measured_snapshot(env, 8) == registration["initial"], "The current endpoint differs from model registration"
    )
    ledger = CausalStreamLedger(**registration)

    def first_live_policy(poses, openings, anchor, step):
        prefix.require(step == 8, "The first new reference must begin at step8")
        new = first_advance(policy, owner, poses, openings, anchor, generated_step=step, runtime_receipt=receipt)
        return new, owner

    def advance_live_policy(previous_policy, actual_owner, poses, openings, anchor, *, generated_step, receipt):
        return next_advance(
            previous_policy,
            actual_owner,
            poses,
            openings,
            anchor,
            generated_step=generated_step,
            segment_receipt=receipt,
            runtime_receipt=outer_receipt,
        )

    outer_receipt = receipt
    adapter = CausalStreamingReference(
        env=env,
        observers=wrapped,
        ledger=ledger,
        channel=channel,
        runtime_output=str(runtime_output.relative_to(prefix.REPO)),
        tool=case["tool"],
        receipt=receipt,
        first_policy=first_live_policy,
        response_gate=lambda _request, _response: True,
        advance_policy=advance_live_policy,
    )
    last_source = last_wrapped = None

    def open_segment(observation, previous_wrapped):
        nonlocal last_source, last_wrapped
        prefix.require(previous_wrapped is last_wrapped, "Segments must follow the last actually completed wrapper")
        source = adapter(observation, last_source)
        current_policy = adapter.policy

        def timed_select(*args):
            started = time.monotonic()
            decision = source.select(*args)
            decision["total_decision_seconds"] = time.monotonic() - started
            return decision

        def completed(actual_env, command, offset, inner):
            history = source.complete(actual_env, command, offset, inner)
            score = gripper_score(actual_env, current_policy.openings[offset])
            return dict(score=score, policy_history=history, policy_history_recomputed=True)

        last_source = source
        last_wrapped = StreamingSegment(source.targets, timed_select, completed)
        return last_wrapped

    stream = run_streaming_continuation(
        env,
        next_terminal,
        wrapped,
        receipt,
        ledger=ledger,
        max_windows=5,
        previous_command=first["periods"][-1]["command"],
        tool=case["tool"],
        target_object=case["target_object"],
        open_segment=open_segment,
        state_reader=state_reader,
        task_observer=task_observer,
        progress=progress,
    )
    if stream["requests_issued"] < registration["max_requests"]:
        prefix.require(
            stream["stop"] is not None and stream["stop"]["next_action_dispatched"] is False,
            "A waiting model requires an explicit pre-dispatch stop",
        )
        channel.send(make_stop(session=ledger.session, next_step=ledger.current["step"]))
        receipt["model_stop_sent"] = True
    else:
        receipt["model_stop_sent"] = False
    receipt.pop("streaming_partial", None)
    return dict(first_window=first, stream=stream)
