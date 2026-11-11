"""Run one current-image WAM window after eight actual reset feedback periods."""

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
    advance_policy=None,
    gripper_score=None,
    render_scope=render_with_two_torch_threads,
):
    """Recompute first-window feedback, then request one current-image reference."""
    if policy_factory is None:
        from s2a.research.joint_control.preview.reset_feedback_policy import policy_inputs
        from s2a.research.joint_control.preview.segmented_reset_policy import build_segmented_reset_policy

        def policy_factory(env, case, terminal, receipt):
            return build_segmented_reset_policy(env, policy_inputs(case, terminal), receipt)

    if advance_policy is None:
        from s2a.research.joint_control.preview.segmented_reset_policy import advance_segmented_reset_policy

        advance_policy = advance_segmented_reset_policy
    state_reader = current_joint_site_state if state_reader is None else state_reader
    gripper_score = original.measured_gripper if gripper_score is None else gripper_score
    prefix.require(
        len(baseline) == 8 and registration["max_requests"] == 1, "One exact first window and one live request required"
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
    current = measured_snapshot(env, 8)
    prefix.require(
        current == registration["initial"], "The actual current endpoint differs from the registered model start"
    )
    ledger = CausalStreamLedger(**registration)

    def first_live_policy(poses, openings, anchor, step):
        prefix.require(step == 8, "The new reference must begin at the actual step-8 endpoint")
        new = advance_policy(policy, owner, poses, openings, anchor, generated_step=step, runtime_receipt=receipt)
        return new, owner

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
    )

    def open_segment(observation, prior):
        source = adapter(observation, prior)
        current_policy = adapter.policy

        def completed(actual_env, command, offset, inner):
            history = source.complete(actual_env, command, offset, inner)
            score = gripper_score(actual_env, current_policy.openings[offset])
            return dict(score=score, policy_history=history, policy_history_recomputed=True)

        return StreamingSegment(source.targets, source.select, completed)

    stream = run_streaming_continuation(
        env,
        next_terminal,
        wrapped,
        receipt,
        ledger=ledger,
        max_windows=1,
        previous_command=first["periods"][-1]["command"],
        tool=case["tool"],
        target_object=case["target_object"],
        open_segment=open_segment,
        state_reader=state_reader,
        task_observer=task_observer,
        progress=progress,
    )
    if stream["requests_issued"] == 0:
        prefix.require(
            stream["stop"] is not None and stream["stop"]["next_action_dispatched"] is False,
            "Only an explicit pre-request stop may close a waiting model",
        )
        channel.send(make_stop(session=ledger.session, next_step=ledger.current["step"]))
        receipt["model_stop_sent"] = True
    else:
        receipt["model_stop_sent"] = False
    receipt.pop("streaming_partial", None)
    return dict(first_window=first, stream=stream)
