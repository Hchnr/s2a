"""Rebuild one reset history and compare two already-generated reference tails."""

import copy

import numpy as np

from s2a.references.action_reference import decode_action_targets
from s2a.research.native_runtime.development_continuation import run_continuation
from s2a.research.native_runtime.endpoint_cache import SameStateEndpointCache
from s2a.research.native_runtime.finite_reference_tail import run_finite_reference_tail
from s2a.research.native_runtime.reference_choice_comparison import paired_errors, quality_stop
from s2a.research.native_runtime.reset_canvas_threads import render_with_two_torch_threads
from s2a.research.native_runtime.streaming_continuation import measured_snapshot
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import reset_live_two_window_replay as captures
from scripts.simulation.runtime import run_reset_start_feedback as original
from scripts.simulation.runtime.run_segmented_reset_equivalence import selected_command
from scripts.wam.tracking.preview import gpu_resume_repro_20261111 as repro


def saved_reference(capture, exchange, response, tool, expected_step):
    """Bind one stored H40 to its actual original current image and state."""
    request, envelope, raw = (
        prefix.read(exchange["request_ref"]),
        prefix.read(exchange["response_ref"]),
        prefix.read(response["actual_response_ref"]),
    )
    endpoint = capture["current_endpoint"]
    prefix.require(
        expected_step in (8, 16)
        and capture["frame_index"] == exchange["step"] == request["step"] == envelope["step"] == expected_step
        and response["step"] == expected_step
        and request["sim_time_s"] == envelope["sim_time_s"] == endpoint["sim_time"]
        and request["state_sha256"] == envelope["state_sha256"] == endpoint["sim_state_sha256"]
        and request["image"]["sha256"] == envelope["image_sha256"] == capture["canvas_png"]["sha256"]
        and request["seed"] == envelope["seed"] == 43
        and envelope["denoise_calls"] == 35
        and all(exchange.get(name) is True for name in ("request_sent", "response_received", "response_accepted")),
        "The saved WAM response must bind the original current image, state and seed",
    )
    arrays = repro.response_arrays(raw)
    np.testing.assert_array_equal(arrays["action"], envelope["action"])
    np.testing.assert_array_equal(arrays["normalized_action"], envelope["normalized_action"])
    anchor = np.asarray(capture["current_source_pose"], dtype=float) @ np.asarray(tool, dtype=float)
    return decode_action_targets(anchor, arrays["action"], horizon=40)


def run_comparison(
    env,
    terminal,
    observers,
    receipt,
    *,
    phase,
    mode,
    case,
    history,
    original_captures,
    exchanges,
    responses,
    runtime_output,
    producer=None,
    progress=None,
    policy_factory=None,
    after_history=None,
):
    """Execute one branch or replay its actions, without future model requests."""
    if phase not in ("producer", "checker") or mode not in ("old", "new"):
        raise ValueError("A registered phase and reference branch are required")
    if len(history) != 16 or len(original_captures) != 2 or len(exchanges) != 2 or len(responses) != 2:
        raise ValueError("Exactly the completed prefix and its two generated references are required")
    if (phase == "checker") != (producer is not None):
        raise ValueError("Only the independent checker may receive a completed producer")
    if (policy_factory is None) != (after_history is None) or (
        policy_factory is not None
        and (phase != "producer" or not callable(policy_factory) or not callable(after_history))
    ):
        raise ValueError("Only a producer may use an explicit paired policy factory and activation hook")
    if phase == "producer":
        from s2a.research.joint_control.preview.causal_reference_replan import completed_quality_margin
        from s2a.research.joint_control.preview.reset_feedback_policy import policy_inputs
        from s2a.research.joint_control.preview.segmented_reset_policy import (
            advance_segmented_reset_policy,
            build_segmented_reset_policy,
        )

        if policy_factory is None:
            policy, owner, previous = build_segmented_reset_policy(env, policy_inputs(case, terminal), receipt)
        else:
            policy, owner, previous = policy_factory(env, case, terminal, receipt)
    else:
        policy = owner = None
        previous = np.zeros(7)
    cache = SameStateEndpointCache(observers.endpoint)
    observed = []

    def capture_at(step):
        sample, endpoint = cache(env, step)
        before = measured_snapshot(env, step)
        identity = {
            "endpoint_step": step,
            "sim_time": endpoint["sim_time"],
            "sim_state_sha256": endpoint["sim_state_sha256"],
        }
        scope = {}
        actual = render_with_two_torch_threads(
            observers.render_anchor, scope, env, step, sample, identity, runtime_output
        )
        prefix.require(measured_snapshot(env, step) == before, "Reobservation may not advance the physics")
        captures.compare_capture(original_captures[len(observed)], actual)
        observed.append(actual)
        return {"sample": sample, "endpoint_receipt": endpoint}

    def history_window(start, current_terminal, targets, openings, expected):
        nonlocal previous

        def choose(q, limits, jacobian, prior, offset):
            if phase == "checker":
                return original.checker_action(expected)(q, limits, jacobian, prior, offset)
            return selected_command(policy.select(q, limits, jacobian, prior, offset), expected[offset])

        def complete(actual_env, command, offset):
            period = receipt["continuation_partial"]["periods"][-1]
            policy_history = None
            if phase == "producer":
                policy_history = policy.complete(
                    actual_env, command, offset, [sample["q"] for sample in period["actual_inner"]]
                )
            score = original.measured_gripper(actual_env, openings[offset])
            original.compare_score(score, expected[offset]["completed_feedback"]["score"])
            return dict(score=score, policy_history=policy_history, policy_history_recomputed=phase == "producer")

        result = run_continuation(
            env,
            current_terminal,
            observers,
            receipt,
            start_step=start,
            previous_command=previous,
            targets=targets,
            tool=case["tool"],
            target_object=case["target_object"],
            choose_action=choose,
            stop_on_outside=True,
            expected_periods=expected,
            on_completed=complete,
            progress=progress,
        )
        prefix.require(result["steps_completed"] == 8 and result["stop"] is None, "The verified prefix must complete")
        previous = np.asarray(result["periods"][-1]["command"], dtype=float)
        return result

    first = history_window(
        0, terminal, np.asarray(case["chunk"]["effective_targets_tcp"][:8]), case["chunk"]["openings"][:8], history[:8]
    )
    current = capture_at(8)
    old_poses, old_openings = saved_reference(observed[0], exchanges[0], responses[0], case["tool"], 8)
    if phase == "producer":
        anchor = np.asarray(current["sample"]["source_pose"]) @ np.asarray(case["tool"])
        policy = advance_segmented_reset_policy(
            policy, owner, old_poses, old_openings, anchor, generated_step=8, runtime_receipt=receipt
        )
    second = history_window(8, current, old_poses[:8], old_openings[:8], history[8:16])
    current = capture_at(16)
    new_poses, new_openings = saved_reference(observed[1], exchanges[1], responses[1], case["tool"], 16)
    if after_history is not None:
        before = measured_snapshot(env, 16)
        prefix.require(
            receipt["control_steps_completed"] == 16, "The full physical history is required before activation"
        )
        after_history(policy, owner, receipt)
        prefix.require(
            measured_snapshot(env, 16) == before, "Activating the candidate cannot change the physical state"
        )
    last = [second["periods"][-1]]
    reference_poses, reference_openings, start, consumed = (
        (old_poses, old_openings, 8, 8) if mode == "old" else (new_poses, new_openings, 16, 0)
    )

    def open_segment(window, openings, anchor, step, expected):
        nonlocal policy
        if phase == "producer":
            prefix.require(expected is None, "A producer cannot receive future recorded actions")
            policy = advance_segmented_reset_policy(
                policy, owner, window, openings, anchor, generated_step=step, runtime_receipt=receipt
            )
        else:
            prefix.require(expected is not None, "The checker must replay actual producer actions")

        def choose(q, limits, jacobian, prior, offset):
            if phase == "checker":
                return original.checker_action(expected)(q, limits, jacobian, prior, offset)
            if not last[0]["pose_caps_satisfied"]:
                return {"stop": {"reason": "observed_pose_cap_exceeded"}}
            stop = quality_stop(last[0], step + offset, completed_quality_margin)
            return stop if stop is not None else policy.select(q, limits, jacobian, prior, offset)

        def complete(actual_env, command, offset):
            period = receipt["continuation_partial"]["periods"][-1]
            policy_history = None
            if phase == "producer":
                policy_history = policy.complete(
                    actual_env, command, offset, [sample["q"] for sample in period["actual_inner"]]
                )
            score = original.measured_gripper(actual_env, openings[offset])
            sample, endpoint = cache(actual_env, step + offset + 1)
            prefix.require(endpoint == period["endpoint_receipt"], "Only the completed actual endpoint may be scored")
            actual_tcp = np.asarray(sample["source_pose"]) @ np.asarray(case["tool"])
            pair = paired_errors(actual_tcp, old_poses, new_poses, step + offset - 16)
            np.testing.assert_allclose(pair[mode], period["tcp_error"], atol=1e-10, rtol=0)
            if phase == "checker":
                original.compare_score(score, expected[offset]["completed_feedback"]["score"])
                for key in ("old", "new"):
                    np.testing.assert_allclose(
                        pair[key], expected[offset]["completed_feedback"]["paired_errors"][key], atol=1e-10, rtol=0
                    )
            last[0] = period
            return dict(
                score=score,
                paired_errors=pair,
                policy_history=policy_history,
                policy_history_recomputed=phase == "producer",
            )

        return choose, complete

    prior_tail = None if producer is None else producer["tail"]
    tail = run_finite_reference_tail(
        env,
        current,
        observers,
        receipt,
        reference_poses=reference_poses,
        reference_openings=reference_openings,
        reference_start_step=start,
        consumed=consumed,
        max_segments=4,
        previous_command=previous,
        previous_period=last[0],
        tool=case["tool"],
        target_object=case["target_object"],
        open_segment=open_segment,
        recorded_segments=None if prior_tail is None else prior_tail["segments"],
        recorded_stop=None if prior_tail is None else prior_tail["stop"],
        progress=progress,
    )
    return dict(first_window=first, second_window=second, captures=copy.deepcopy(observed), tail=tail)
