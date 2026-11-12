"""Continue a verified reset H40 using one saved step-48 WAM response."""

import copy
import time
from types import SimpleNamespace

import numpy as np

from s2a.references.action_reference import decode_action_targets
from s2a.research.native_runtime.development_continuation import current_joint_site_state, run_continuation
from s2a.research.native_runtime.reset_canvas_threads import render_with_two_torch_threads
from s2a.research.native_runtime.streaming_continuation import measured_snapshot
from s2a.research.wam.training.native.native_training import state_sha256
from scripts.simulation.runtime import audit_reset_multitask_live_recovery as comparison
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import reset_live_two_window_replay as capture_check
from scripts.simulation.runtime import run_reset_start_feedback as original
from scripts.simulation.runtime.reset_live_two_window_replay_v2 import minimal_renderer_endpoint
from scripts.wam.tracking.preview import gpu_resume_repro_20261111 as repro


def periods(continuation):
    """Flatten only the physically dispatched first 48 periods."""
    return continuation["first_window"]["periods"] + [
        row for segment in continuation["fixed"]["segments"] for row in segment["continuation"]["periods"]
    ]


def current_capture(env, observers, saved, expected_capture, runtime_output, *, render_scope):
    """Reproduce the actual endpoint and current pixels before accepting a response."""
    prefix.require(len(saved) == 1, "One actual step-48 endpoint is required")
    sample, endpoint = saved[0]
    current = measured_snapshot(env, 48)
    identity = minimal_renderer_endpoint(endpoint, 48)
    binding = {"sim_time": current["sim_time_s"], "sim_state_sha256": current["state_sha256"]}
    prefix.require(
        identity == {"endpoint_step": 48, **binding} and state_sha256(sample["sim_states"]) == current["state_sha256"],
        "The observed endpoint must bind the actual current state",
    )
    pose, opening = np.asarray(sample["source_pose"]).copy(), sample["opening"]
    scope = {}
    capture = render_scope(observers.render_anchor, scope, env, 48, sample, copy.deepcopy(identity), runtime_output)
    capture_check.compare_capture(expected_capture, capture)
    prefix.require(
        measured_snapshot(env, 48) == current
        and capture["current_endpoint"] == {"endpoint_step": 48, **binding}
        and capture["render_before"] == capture["render_after"] == binding
        and capture["future_rgb_frames"] == 0,
        "Reobservation must preserve the actual state and contain no future image",
    )
    np.testing.assert_array_equal(capture["current_source_pose"], pose)
    prefix.require(capture["current_opening"] == opening, "The observed gripper opening must match")
    return sample, endpoint, current, capture, scope.get("canvas_thread_scope")


def decode_saved_reference(current, capture, sample, tool, load_second_reference):
    """Bind both model representations to the newly reproduced current observation."""
    response, envelope = load_second_reference()
    arrays = repro.response_arrays(response)
    np.testing.assert_array_equal(arrays["action"], envelope["action"])
    np.testing.assert_array_equal(arrays["normalized_action"], envelope["normalized_action"])
    prefix.require(
        envelope["step"] == current["step"] == 48
        and envelope["sim_time_s"] == current["sim_time_s"]
        and envelope["state_sha256"] == current["state_sha256"]
        and envelope["image_sha256"] == capture["canvas_png"]["sha256"]
        and envelope["denoise_calls"] == 35
        and envelope["seed"] == 42,
        "The saved WAM response must match the actual current image and state",
    )
    anchor = np.asarray(sample["source_pose"], dtype=float) @ np.asarray(tool, dtype=float)
    poses, openings = decode_action_targets(anchor, arrays["action"], horizon=40)
    return anchor, poses, openings


def _remember_endpoint(observers, saved, receipt):
    def endpoint(env, step):
        sample, evidence = observers.endpoint(env, step)
        if step == 48:
            saved.append((copy.deepcopy(sample), copy.deepcopy(evidence)))
        return sample, evidence

    def render_cameras(env, size):
        step = receipt["control_steps_completed"]
        before = measured_snapshot(env, step)
        result = observers.render_cameras(env, size)
        prefix.require(measured_snapshot(env, step) == before, "Rendering cannot advance physical state")
        return result

    return SimpleNamespace(
        endpoint=endpoint,
        current=observers.current,
        render_anchor=observers.render_anchor,
        render_cameras=render_cameras,
    )


def run_producer(
    env,
    terminal,
    observers,
    receipt,
    *,
    case,
    baseline,
    expected_capture,
    load_reference,
    runtime_output,
    previous_producer,
    expected_second_capture,
    load_second_reference,
    progress=None,
    state_reader=None,
    task_observer=None,
    policy_factory=None,
    advance_policy=None,
    gripper_score=None,
    render_scope=render_with_two_torch_threads,
):
    """Continue only after reproducing the actual 48-period history and image."""
    from scripts.simulation.runtime.reset_bowl_fixed_h40 import run_producer as first_producer

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
    state = {}

    def start(*args):
        policy, owner, previous = policy_factory(*args)
        state.update(policy=policy, owner=owner)
        return policy, owner, previous

    def advance(previous, owner, *args, **kwargs):
        policy = advance_policy(previous, owner, *args, **kwargs)
        state.update(policy=policy, owner=owner)
        return policy

    saved = []
    wrapped = _remember_endpoint(observers, saved, receipt)
    first = first_producer(
        env,
        terminal,
        wrapped,
        receipt,
        case=case,
        baseline=baseline,
        expected_capture=expected_capture,
        load_reference=load_reference,
        runtime_output=runtime_output,
        progress=progress,
        state_reader=state_reader,
        task_observer=task_observer,
        policy_factory=start,
        advance_policy=advance,
        gripper_score=gripper_score,
        render_scope=render_scope,
    )
    actual = periods(first)
    expected = periods(previous_producer["runtime"]["continuation"])
    comparison.compare_periods(expected[: len(actual)], actual)
    second = dict(attempted=False, periods=[], stop=None)
    first["second"] = second
    receipt["second_reference_partial"] = second
    if not first["fixed"]["complete"]:
        second["stop"] = first["fixed"]["stop"]
        return first
    prefix.require(len(actual) == len(expected) == 48, "All original 48 periods must be reproduced")
    sample, endpoint, current, capture, scope = current_capture(
        env, observers, saved, expected_second_capture, runtime_output, render_scope=render_scope
    )
    receipt["second_reference_capture"] = capture
    receipt["second_canvas_thread_scope"] = scope
    anchor, poses, openings = decode_saved_reference(current, capture, sample, case["tool"], load_second_reference)
    q, limits, _jacobian = state_reader(env)
    q, limits = np.asarray(q, dtype=float), np.asarray(limits, dtype=float)
    if actual[-1]["inner_below_buffer_samples"] or np.min(np.minimum(q - limits[:, 0], limits[:, 1] - q)) < 0.05:
        second["stop"] = dict(reason="observed_joint_buffer_loss", at_step=48, next_action_dispatched=False)
        return first
    policy = advance(
        state["policy"], state["owner"], poses, openings, anchor, generated_step=48, runtime_receipt=receipt
    )
    prefix.require(measured_snapshot(env, 48) == current, "Segment creation cannot advance physical state")

    def select(q, limits, jacobian, prior, offset):
        snapshot = measured_snapshot(env, 48 + offset)
        past = receipt["continuation_partial"]["periods"]
        if past and past[-1]["inner_below_buffer_samples"]:
            return {"stop": {"reason": "observed_joint_buffer_loss"}}
        started = time.monotonic()
        decision = policy.select(q, limits, jacobian, prior, offset)
        decision["total_decision_seconds"] = time.monotonic() - started
        prefix.require(measured_snapshot(env, 48 + offset) == snapshot, "Action selection cannot advance physics")
        return decision

    def complete(actual_env, command, offset):
        period = receipt["continuation_partial"]["periods"][-1]
        snapshot = measured_snapshot(actual_env, 49 + offset)
        history = policy.complete(actual_env, command, offset, [item["q"] for item in period["actual_inner"]])
        score = gripper_score(actual_env, openings[offset])
        prefix.require(measured_snapshot(actual_env, 49 + offset) == snapshot, "Feedback cannot advance physics")
        return dict(score=score, policy_history=history, policy_history_recomputed=True)

    second["attempted"] = True
    try:
        outcome = run_continuation(
            env,
            {"sample": sample, "endpoint_receipt": endpoint},
            wrapped,
            receipt,
            start_step=48,
            previous_command=actual[-1]["command"],
            targets=poses[:8],
            tool=case["tool"],
            target_object=case["target_object"],
            choose_action=select,
            stop_on_outside=True,
            on_completed=complete,
            state_reader=state_reader,
            task_observer=task_observer,
            progress=progress,
        )
    finally:
        second["continuation"] = receipt.get("continuation_partial")
    second.update(periods=outcome["periods"], stop=outcome["stop"])
    return first


def run_checker(
    env,
    terminal,
    observers,
    receipt,
    *,
    producer,
    case,
    expected_capture,
    load_reference,
    runtime_output,
    expected_second_capture,
    load_second_reference,
    progress=None,
    state_reader=None,
    task_observer=None,
    gripper_score=None,
    render_scope=render_with_two_torch_threads,
):
    """Reconstruct references and replay only actions actually dispatched."""
    from scripts.simulation.runtime.reset_bowl_fixed_h40_replay import run_checker as first_checker

    state_reader = current_joint_site_state if state_reader is None else state_reader
    gripper_score = original.measured_gripper if gripper_score is None else gripper_score
    saved = []
    wrapped = _remember_endpoint(observers, saved, receipt)
    first = first_checker(
        env,
        terminal,
        wrapped,
        receipt,
        case=case,
        producer=producer,
        expected_capture=expected_capture,
        load_reference=load_reference,
        runtime_output=runtime_output,
        progress=progress,
        state_reader=state_reader,
        task_observer=task_observer,
        gripper_score=gripper_score,
        render_scope=render_scope,
    )
    recorded = producer["runtime"]["continuation"]
    expected_second = recorded["second"]
    second = dict(attempted=False, periods=[], reported_producer_stop=expected_second["stop"], stop_recomputed=False)
    first["second"] = second
    if not recorded["fixed"]["complete"]:
        prefix.require(not expected_second["attempted"] and not expected_second["periods"], "No second window exists")
        return first
    sample, endpoint, current, capture, scope = current_capture(
        env, observers, saved, expected_second_capture, runtime_output, render_scope=render_scope
    )
    receipt["second_reference_capture"] = capture
    receipt["second_canvas_thread_scope"] = scope
    _anchor, poses, openings = decode_saved_reference(current, capture, sample, case["tool"], load_second_reference)
    expected = expected_second["periods"]
    if not expected_second["attempted"]:
        prefix.require(not expected and expected_second["stop"] is not None, "Only a recorded boundary stop is allowed")
        return first

    def score_completed(actual_env, _command, offset):
        snapshot = measured_snapshot(actual_env, 49 + offset)
        score = gripper_score(actual_env, openings[offset])
        original.compare_score(score, expected[offset]["completed_feedback"]["score"])
        prefix.require(measured_snapshot(actual_env, 49 + offset) == snapshot, "Scoring cannot advance physics")
        return dict(score=score, policy_history=None, policy_history_recomputed=False)

    second["attempted"] = True
    outcome = run_continuation(
        env,
        {"sample": sample, "endpoint_receipt": endpoint},
        wrapped,
        receipt,
        start_step=48,
        previous_command=periods(recorded)[-1]["command"],
        targets=poses[:8],
        tool=case["tool"],
        target_object=case["target_object"],
        choose_action=original.checker_action(expected),
        stop_on_outside=True,
        expected_periods=expected,
        on_completed=score_completed,
        state_reader=state_reader,
        task_observer=task_observer,
        progress=progress,
    )
    second["periods"] = outcome["periods"]
    second["physical_replay_stop"] = outcome["stop"]
    return first
