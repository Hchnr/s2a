"""Consume one already-generated H40 after an exact reset and first window."""

import time
from types import SimpleNamespace

import numpy as np

from s2a.references.action_reference import decode_action_targets
from s2a.references.finite_horizon_window import remaining_h40
from s2a.research.native_runtime.development_continuation import current_joint_site_state, run_continuation
from s2a.research.native_runtime.endpoint_cache import SameStateEndpointCache
from s2a.research.native_runtime.reset_canvas_threads import render_with_two_torch_threads
from s2a.research.native_runtime.reset_stream_boundary import first_window_inner
from s2a.research.native_runtime.streaming_continuation import _prefix_safe, measured_snapshot
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import reset_live_two_window_replay as capture_check
from scripts.simulation.runtime import run_reset_start_feedback as original
from scripts.simulation.runtime.reset_live_two_window_replay_v2 import minimal_renderer_endpoint
from scripts.simulation.runtime.run_segmented_reset_equivalence import selected_command
from scripts.wam.tracking.preview import gpu_resume_repro_20261111 as repro


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
    progress=None,
    state_reader=None,
    task_observer=None,
    policy_factory=None,
    advance_policy=None,
    gripper_score=None,
    render_scope=render_with_two_torch_threads,
):
    """Preserve actual feedback history while consuming a frozen generated reference."""
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
        len(baseline) == 8 and callable(load_reference),
        "Eight completed initial periods and a saved response are required",
    )
    cache = SameStateEndpointCache(observers.endpoint)
    receipt["endpoint_cache_events"] = cache.events

    def render_checked(actual_env, size):
        step = receipt["control_steps_completed"]
        before = measured_snapshot(actual_env, step)
        result = observers.render_cameras(actual_env, size)
        prefix.require(measured_snapshot(actual_env, step) == before, "Rendering cannot advance physical state")
        return result

    wrapped = SimpleNamespace(endpoint=cache, current=observers.current, render_cameras=render_checked)
    policy, owner, previous = policy_factory(env, case, terminal, receipt)

    def first_select(q, limits, jacobian, prior, offset):
        return selected_command(policy.select(q, limits, jacobian, prior, offset), baseline[offset])

    def first_complete(actual_env, command, offset):
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
        choose_action=first_select,
        stop_on_outside=True,
        expected_periods=baseline,
        on_completed=first_complete,
        state_reader=state_reader,
        task_observer=task_observer,
        progress=progress,
    )
    receipt["reset_first_window"] = first
    prefix.require(first["steps_completed"] == 8 and first["stop"] is None, "The initial reset window must complete")
    q, limits, _jacobian = state_reader(env)
    receipt["final_prefix_inner"] = first_window_inner(first["periods"], limits)
    current = measured_snapshot(env, 8)
    fixed = dict(segments=[], stop=None, complete=False, actual_model_requests=0, automatic_retry=False)
    receipt["fixed_partial"] = fixed
    if not _prefix_safe(receipt, np.asarray(q, dtype=float), np.asarray(limits, dtype=float), 8, current["sim_time_s"]):
        fixed["stop"] = dict(reason="observed_prefix_joint_buffer_loss", at_step=8, next_action_dispatched=False)
        return dict(first_window=first, fixed=fixed)
    sample, endpoint = cache(env, 8)
    scope = {}
    capture = render_scope(
        observers.render_anchor, scope, env, 8, sample, minimal_renderer_endpoint(endpoint, 8), runtime_output
    )
    receipt["fixed_reference_capture"] = capture
    receipt["fixed_canvas_thread_scope"] = scope.get("canvas_thread_scope")
    capture_check.compare_capture(expected_capture, capture)
    prefix.require(measured_snapshot(env, 8) == current, "Current-image validation must not advance physics")
    response, envelope = load_reference()
    arrays = repro.response_arrays(response)
    np.testing.assert_array_equal(arrays["action"], envelope["action"])
    np.testing.assert_array_equal(arrays["normalized_action"], envelope["normalized_action"])
    prefix.require(
        envelope["step"] == 8
        and envelope["state_sha256"] == current["state_sha256"]
        and envelope["sim_time_s"] == current["sim_time_s"]
        and envelope["image_sha256"] == capture["canvas_png"]["sha256"]
        and envelope["denoise_calls"] == 35,
        "The saved WAM response must belong to this actual current image and state",
    )
    anchor = np.asarray(sample["source_pose"], dtype=float) @ np.asarray(case["tool"], dtype=float)
    poses, openings = decode_action_targets(anchor, arrays["action"], horizon=40)
    previous_command = np.asarray(first["periods"][-1]["command"], dtype=float)
    last_period = first["periods"][-1]
    for offset in range(0, 40, 8):
        step = 8 + offset
        snapshot = measured_snapshot(env, step)
        q, limits, _jacobian = state_reader(env)
        q, limits = np.asarray(q, dtype=float), np.asarray(limits, dtype=float)
        if last_period["inner_below_buffer_samples"] or np.min(np.minimum(q - limits[:, 0], limits[:, 1] - q)) < 0.05:
            fixed["stop"] = dict(reason="observed_joint_buffer_loss", at_step=step, next_action_dispatched=False)
            break
        sample, endpoint = cache(env, step)
        current_anchor = np.asarray(sample["source_pose"], dtype=float) @ np.asarray(case["tool"], dtype=float)
        window_poses, window_openings, window_receipt = remaining_h40(poses, openings, consumed=offset)
        policy = advance_policy(
            policy,
            owner,
            window_poses,
            window_openings,
            current_anchor,
            generated_step=step,
            runtime_receipt=receipt,
        )
        prefix.require(measured_snapshot(env, step) == snapshot, "Segment creation cannot advance physics")
        row = dict(start_step=step, reference_offset=offset, **window_receipt, continuation=None)
        fixed["segments"].append(row)

        def select(q, limits, jacobian, prior, local_offset):
            actual_step = step + local_offset
            before = measured_snapshot(env, actual_step)
            past = receipt["continuation_partial"]["periods"]
            if past and past[-1]["inner_below_buffer_samples"]:
                return {"stop": {"reason": "observed_joint_buffer_loss"}}
            started = time.monotonic()
            decision = policy.select(q, limits, jacobian, prior, local_offset)
            decision["total_decision_seconds"] = time.monotonic() - started
            prefix.require(measured_snapshot(env, actual_step) == before, "Action selection cannot advance physics")
            return decision

        def complete(actual_env, command, local_offset):
            period = receipt["continuation_partial"]["periods"][-1]
            after = measured_snapshot(actual_env, step + local_offset + 1)
            history = policy.complete(actual_env, command, local_offset, [item["q"] for item in period["actual_inner"]])
            score = gripper_score(actual_env, window_openings[local_offset])
            prefix.require(
                measured_snapshot(actual_env, step + local_offset + 1) == after, "Feedback cannot advance physics"
            )
            return dict(score=score, policy_history=history, policy_history_recomputed=True)

        try:
            outcome = run_continuation(
                env,
                {"sample": sample, "endpoint_receipt": endpoint},
                wrapped,
                receipt,
                start_step=step,
                previous_command=previous_command,
                targets=window_poses[:8],
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
            row["continuation"] = receipt.get("continuation_partial")
        if outcome["stop"] is not None:
            fixed["stop"] = outcome["stop"]
            break
        prefix.require(outcome["steps_completed"] == 8, "Each nonterminal segment must complete eight periods")
        last_period = outcome["periods"][-1]
        previous_command = np.asarray(last_period["command"], dtype=float)
    else:
        fixed["complete"] = True
    return dict(first_window=first, fixed=fixed)
