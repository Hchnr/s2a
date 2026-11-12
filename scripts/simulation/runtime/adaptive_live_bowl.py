"""Reconstruct a triggered reset prefix, then run or replay an adaptive stream."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from s2a.research.native_runtime.completed_period_boundary import completed_period_inner
from s2a.research.native_runtime.reset_canvas_threads import render_with_two_torch_threads
from s2a.research.native_runtime.streaming_continuation import measured_snapshot
from s2a.research.wam.tracking.runtime.adaptive_stream_protocol import AdaptiveStreamLedger
from scripts.simulation.runtime import audit_reset_multitask_live_recovery as comparison
from scripts.simulation.runtime import causal_replan_step107_execution as previous
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import reset_bowl_reference_tail as tail
from scripts.simulation.runtime import reset_live_two_window_replay as capture_check
from scripts.simulation.runtime import run_reset_start_feedback as feedback


def match_first_reference(request, response, expected):
    """Bind the first live response to the already audited same-image output."""
    if request["request_id"] != 0:
        return True
    prefix.require(
        request["step"] == expected["step"]
        and request["sim_time_s"] == expected["sim_time_s"]
        and request["state_sha256"] == expected["state_sha256"]
        and request["image"]["sha256"] == expected["image_sha256"]
        and response["image_sha256"] == expected["image_sha256"]
        and response["model_sha256"] == expected["model_sha256"]
        and response["seed"] == expected["seed"]
        and response["denoise_calls"] == expected["denoise_calls"],
        "The first live response must bind the previously audited image and sampling",
    )
    np.testing.assert_array_equal(response["action"], expected["action"])
    np.testing.assert_array_equal(response["normalized_action"], expected["normalized_action"])
    return True


def run_producer(
    env,
    terminal,
    observers,
    receipt,
    *,
    prefix_runner,
    prior_producer,
    expected_replan_capture,
    expected_response,
    registration,
    channel,
    runtime_output,
    case,
    reference_start=88,
    replan_step=107,
    policy_factory=None,
    advance_policy=None,
    partial_advance=None,
    trigger=None,
    gripper_score=None,
    state_reader=None,
    task_observer=None,
    progress=None,
    render_scope=render_with_two_torch_threads,
    **prefix_options,
):
    """Rebuild completed control history and run current-image adaptive references."""
    from s2a.research.joint_control.preview.causal_reference_replan import (
        advance_recorded_segment,
        completed_quality_margin,
    )
    from s2a.research.joint_control.preview.reset_feedback_policy import policy_inputs
    from s2a.research.joint_control.preview.segmented_reset_policy import (
        advance_segmented_reset_policy,
        build_segmented_reset_policy,
    )
    from s2a.research.native_runtime.adaptive_streaming_continuation import run_adaptive_streaming_continuation
    from s2a.research.native_runtime.development_continuation import current_joint_site_state
    from s2a.research.wam.tracking.runtime.adaptive_stream_control import close_adaptive_model_stream
    from s2a.research.wam.tracking.runtime.adaptive_stream_reference import AdaptiveStreamingReference

    if policy_factory is None:

        def policy_factory(actual_env, actual_case, current, runtime_receipt):
            return build_segmented_reset_policy(actual_env, policy_inputs(actual_case, current), runtime_receipt)

    advance_policy = advance_segmented_reset_policy if advance_policy is None else advance_policy
    partial_advance = advance_recorded_segment if partial_advance is None else partial_advance
    trigger = completed_quality_margin if trigger is None else trigger
    gripper_score = feedback.measured_gripper if gripper_score is None else gripper_score
    state_reader = current_joint_site_state if state_reader is None else state_reader
    state, saved = {"last_period": None, "signals": []}, []

    def guard(policy, owner, start_step):
        state.update(policy=policy, owner=owner)
        select = policy.select

        def guarded(q, limits, jacobian, command, offset):
            step = start_step + offset
            periods = receipt.get("continuation_partial", {}).get("periods", [])
            prior = periods[-1] if periods else state["last_period"]
            if prior is not None and prior["step"] >= reference_start and step == prior["step"] + 1:
                signal = trigger(prior, current_step=step)
                state["signals"].append(signal)
                if signal["replan"]:
                    return {"stop": {"reason": "requested_observation", "trigger": signal}}
            if step >= replan_step:
                return {"stop": {"reason": "expected_trigger_missing"}}
            return select(q, limits, jacobian, command, offset)

        policy.select = guarded
        return policy

    def start(*args):
        policy, owner, command = policy_factory(*args)
        return guard(policy, owner, 0), owner, command

    def advance(old, owner, *args, **options):
        periods = receipt.get("continuation_partial", {}).get("periods", [])
        if periods:
            state["last_period"] = periods[-1]
        policy = advance_policy(old, owner, *args, **options)
        return guard(policy, owner, options["generated_step"])

    wrapped = tail.remember_endpoint(observers, saved, step=replan_step)
    first = prefix_runner(
        env,
        terminal,
        wrapped,
        receipt,
        case=case,
        runtime_output=runtime_output,
        policy_factory=start,
        advance_policy=advance,
        gripper_score=gripper_score,
        state_reader=state_reader,
        task_observer=task_observer,
        progress=progress,
        render_scope=render_scope,
        **prefix_options,
    )
    actual = previous.prefix_periods(first)
    expected = previous.prefix_periods(prior_producer["runtime"]["continuation"])
    stop = first["new_reference"]["stop"]
    prefix.require(
        len(actual) == replan_step
        and len(expected) >= replan_step
        and len(saved) == 1
        and type(stop) is dict
        and stop.get("reason") == "requested_observation"
        and stop.get("at_step") == replan_step
        and stop.get("next_action_dispatched") is False
        and stop.get("trigger", {}).get("replan") is True,
        "The first registered causal trigger must precede the next action",
    )
    comparison.compare_periods(expected[:replan_step], actual)
    sample, endpoint = saved[0]
    prefix.require(
        endpoint == expected[replan_step - 1]["endpoint_receipt"]
        and measured_snapshot(env, replan_step) == registration["initial"],
        "The actual trigger endpoint must bind the registered state",
    )
    _q, limits, _jacobian = state_reader(env)
    if "final_prefix_inner" in receipt:
        receipt["pre_adaptive_final_prefix_inner"] = receipt["final_prefix_inner"]
    receipt["final_prefix_inner"] = completed_period_inner(actual[-1], limits)
    ledger = AdaptiveStreamLedger(**registration)
    scopes = receipt["adaptive_canvas_thread_scopes"] = []

    def render_anchor(*args, **kwargs):
        scope = {}
        try:
            capture = render_scope(observers.render_anchor, scope, *args, **kwargs)
            if args[1] == replan_step:
                capture_check.compare_capture(expected_replan_capture, capture)
            return capture
        finally:
            scopes.append(scope.get("canvas_thread_scope"))

    adaptive_observers = SimpleNamespace(
        endpoint=observers.endpoint,
        current=observers.current,
        render_anchor=render_anchor,
        render_cameras=observers.render_cameras,
    )

    def first_policy(poses, openings, anchor, step):
        policy = partial_advance(
            state["policy"], state["owner"], poses, openings, anchor, generated_step=step, runtime_receipt=receipt
        )
        return policy, state["owner"]

    adapter = AdaptiveStreamingReference(
        env=env,
        observers=adaptive_observers,
        ledger=ledger,
        channel=channel,
        runtime_output=str(Path(runtime_output).relative_to(prefix.REPO)),
        tool=case["tool"],
        receipt=receipt,
        first_policy=first_policy,
        response_gate=lambda request, response: match_first_reference(request, response, expected_response),
        advance_policy=partial_advance,
        score_gripper=gripper_score,
    )
    stream = run_adaptive_streaming_continuation(
        env,
        {"sample": sample, "endpoint_receipt": endpoint},
        adaptive_observers,
        receipt,
        ledger=ledger,
        previous_command=actual[-1]["command"],
        tool=case["tool"],
        target_object=case["target_object"],
        open_segment=adapter,
        refresh_signal=trigger,
        state_reader=state_reader,
        task_observer=task_observer,
        progress=progress,
    )
    close_adaptive_model_stream(channel, ledger, stream, receipt)
    return dict(prefix107=first, adaptive=stream, trigger_signals=state["signals"])


def run_checker(
    env,
    terminal,
    observers,
    receipt,
    *,
    prefix_runner,
    producer,
    expected_replan_capture,
    responses,
    registration,
    runtime_output,
    case,
    replan_step=107,
    gripper_score=None,
    compare_score=None,
    state_reader=None,
    task_observer=None,
    progress=None,
    render_scope=render_with_two_torch_threads,
    **prefix_options,
):
    """Replay the actual prefix and adaptive actions without loading a policy."""
    from s2a.research.native_runtime.adaptive_streaming_replay import replay_adaptive_streaming_continuation

    gripper_score = feedback.measured_gripper if gripper_score is None else gripper_score
    compare_score = feedback.compare_score if compare_score is None else compare_score
    recorded = producer["runtime"]["continuation"]
    prefix_producer = {"runtime": {**producer["runtime"], "continuation": recorded["prefix107"]}}
    saved = []
    wrapped = tail.remember_endpoint(observers, saved, step=replan_step)
    first = prefix_runner(
        env,
        terminal,
        wrapped,
        receipt,
        producer=prefix_producer,
        case=case,
        runtime_output=runtime_output,
        gripper_score=gripper_score,
        state_reader=state_reader,
        task_observer=task_observer,
        progress=progress,
        render_scope=render_scope,
        **prefix_options,
    )
    actual = previous.prefix_periods(first, replay=True)
    expected = previous.prefix_periods(recorded["prefix107"])
    prefix.require(len(actual) == len(expected) == replan_step and len(saved) == 1, "Complete recorded prefix required")
    comparison.compare_periods(expected, actual)
    sample, endpoint = saved[0]
    prefix.require(
        endpoint == expected[-1]["endpoint_receipt"] and measured_snapshot(env, replan_step) == registration["initial"],
        "The checker must reach the registered current endpoint",
    )
    ledger = AdaptiveStreamLedger(**registration)
    captures = receipt["adaptive_checker_captures"] = []
    scopes = receipt["adaptive_checker_canvas_thread_scopes"] = []
    expected_captures = producer["runtime"]["adaptive_captures"]

    def capture_current(actual_env, observation, index):
        snapshot = observation["snapshot"]
        step = snapshot["step"]
        identity = {
            "endpoint_step": step,
            "sim_time": snapshot["sim_time_s"],
            "sim_state_sha256": snapshot["state_sha256"],
        }
        observed = {"source_pose": observation["source_pose"], "opening": observation["opening"]}
        scope = {}
        try:
            capture = render_scope(observers.render_anchor, scope, actual_env, step, observed, identity, runtime_output)
        finally:
            scopes.append(scope.get("canvas_thread_scope"))
        capture_check.compare_capture(expected_captures[index], capture)
        if index == 0:
            capture_check.compare_capture(expected_replan_capture, capture)
        captures.append(capture)
        image = capture["canvas_png"]
        path = Path(image["path"])
        root = prefix.REPO / registration["image_root"]
        prefix.require(path.is_absolute() and path.is_relative_to(root), "Checker image must stay in its capture root")
        return dict(path=str(path.relative_to(root)), sha256=image["sha256"])

    result = replay_adaptive_streaming_continuation(
        env,
        {"sample": sample, "endpoint_receipt": endpoint},
        observers,
        receipt,
        ledger=ledger,
        producer_stream=recorded["adaptive"],
        responses=responses,
        capture_current=capture_current,
        previous_command=actual[-1]["command"],
        tool=case["tool"],
        target_object=case["target_object"],
        score_gripper=gripper_score,
        compare_score=compare_score,
        state_reader=state_reader,
        task_observer=task_observer,
        progress=progress,
    )
    return dict(prefix107=first, adaptive=result)
