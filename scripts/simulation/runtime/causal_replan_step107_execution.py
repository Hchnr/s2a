"""Rebuild completed feedback and execute one causally triggered reference."""

import time

import numpy as np

from s2a.references.action_reference import decode_action_targets
from s2a.research.native_runtime.finite_reference_tail import run_finite_reference_tail
from s2a.research.native_runtime.verified_terminal_capture import capture_current_endpoint
from scripts.simulation.runtime import audit_reset_multitask_live_recovery as comparison
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import reset_bowl_next_reference as reference
from scripts.simulation.runtime import reset_bowl_reference_tail as tail
from scripts.simulation.runtime import reset_live_two_window_replay as capture_check
from scripts.simulation.runtime import run_reset_start_feedback as original
from scripts.wam.tracking.preview import gpu_resume_repro_20261111 as repro

MODEL_SHA = "48138bb8c95ce8cc1ca0b70f925e57c8f16bfb21be0a243495b391a06cf5b8ec"


def prefix_periods(continuation, *, replay=False):
    return reference.prefix_periods(continuation["prefix88"], replay=replay) + tail.tail_periods(
        continuation["new_reference"]
    )


def decode_reference(snapshot, capture, sample, tool, load_reference, *, step):
    """Bind an actual saved model response to this exact current observation."""
    response, envelope = load_reference()
    arrays = repro.response_arrays(response)
    np.testing.assert_array_equal(arrays["action"], envelope["action"])
    np.testing.assert_array_equal(arrays["normalized_action"], envelope["normalized_action"])
    prefix.require(
        snapshot["step"] == envelope["step"] == step
        and snapshot["sim_time_s"] == envelope["sim_time_s"]
        and snapshot["state_sha256"] == envelope["state_sha256"]
        and capture["canvas_png"]["sha256"] == envelope["image_sha256"]
        and envelope["seed"] == 42
        and envelope["model_sha256"] == MODEL_SHA
        and envelope["denoise_calls"] == 35
        and envelope["action_dim_mask"] == [True] * 10 + [False] * 118,
        "The new model response must bind the actual current image and state",
    )
    anchor = np.asarray(sample["source_pose"], dtype=float) @ np.asarray(tool, dtype=float)
    return decode_action_targets(anchor, arrays["action"], horizon=40)


def _current_capture(env, observers, saved, expected, step, runtime_output, render_scope):
    if len(saved) != 1:
        raise ValueError("One actual observation at the requested boundary is required")
    options = dict(step=step, expected_endpoint=expected["endpoint_receipt"], runtime_output=runtime_output)
    if render_scope is not None:
        options["render_scope"] = render_scope
    result = capture_current_endpoint(env, observers.render_anchor, saved[0][0], saved[0][1], **options)
    return result


def run_producer(
    env,
    terminal,
    observers,
    receipt,
    *,
    prior_producer,
    expected_replan_capture,
    load_new_reference,
    prefix_runner,
    policy_factory=None,
    advance_policy=None,
    partial_advance=None,
    trigger=None,
    gripper_score=None,
    reference_start=88,
    replan_step=107,
    **kwargs,
):
    """Rebuild actual history, stop at a causal signal and consume the new H40."""
    if policy_factory is None:
        from s2a.research.joint_control.preview.reset_feedback_policy import policy_inputs
        from s2a.research.joint_control.preview.segmented_reset_policy import build_segmented_reset_policy

        def policy_factory(actual_env, case, current, runtime_receipt):
            return build_segmented_reset_policy(actual_env, policy_inputs(case, current), runtime_receipt)

    if advance_policy is None:
        from s2a.research.joint_control.preview.segmented_reset_policy import advance_segmented_reset_policy

        advance_policy = advance_segmented_reset_policy
    if partial_advance is None or trigger is None:
        from s2a.research.joint_control.preview.causal_reference_replan import (
            advance_recorded_segment,
            completed_quality_margin,
        )

        partial_advance = advance_recorded_segment if partial_advance is None else partial_advance
        trigger = completed_quality_margin if trigger is None else trigger
    gripper_score = original.measured_gripper if gripper_score is None else gripper_score
    state, saved = {"last_period": None, "signals": []}, []

    def guard(policy, owner, start_step):
        state.update(policy=policy, owner=owner)
        select = policy.select

        def guarded(q, limits, jacobian, previous, offset):
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
            return select(q, limits, jacobian, previous, offset)

        policy.select = guarded
        return policy

    def start(*args):
        policy, owner, previous = policy_factory(*args)
        return guard(policy, owner, 0), owner, previous

    def advance(previous, owner, *args, **options):
        periods = receipt.get("continuation_partial", {}).get("periods", [])
        if periods:
            state["last_period"] = periods[-1]
        policy = advance_policy(previous, owner, *args, **options)
        return guard(policy, owner, options["generated_step"])

    wrapped = tail.remember_endpoint(observers, saved, step=replan_step)
    first = prefix_runner(
        env,
        terminal,
        wrapped,
        receipt,
        policy_factory=start,
        advance_policy=advance,
        gripper_score=gripper_score,
        **kwargs,
    )
    actual = prefix_periods(first)
    expected = prefix_periods(prior_producer["runtime"]["continuation"])
    stop = first["new_reference"]["stop"]
    prefix.require(
        len(actual) == replan_step
        and len(expected) >= replan_step
        and type(stop) is dict
        and stop.get("reason") == "requested_observation"
        and stop.get("at_step") == replan_step
        and stop.get("next_action_dispatched") is False
        and stop.get("trigger", {}).get("replan") is True,
        "The first causal trigger must occur before dispatch at the registered endpoint",
    )
    comparison.compare_periods(expected[:replan_step], actual)
    captured = _current_capture(
        env,
        observers,
        saved,
        expected[replan_step - 1],
        replan_step,
        kwargs["runtime_output"],
        kwargs.get("render_scope"),
    )
    capture_check.compare_capture(expected_replan_capture, captured["capture"])
    receipt["replan_capture"] = captured
    sample, _endpoint = saved[0]
    poses, openings = decode_reference(
        captured["snapshot"], captured["capture"], sample, kwargs["case"]["tool"], load_new_reference, step=replan_step
    )
    state["partial_advanced"] = False

    def open_segment(window, opening, anchor, step, expected_periods):
        if expected_periods is not None:
            raise ValueError("A producer cannot use recorded future actions")
        advance_fn = partial_advance if not state["partial_advanced"] else advance_policy
        policy = advance_fn(
            state["policy"], state["owner"], window, opening, anchor, generated_step=step, runtime_receipt=receipt
        )
        state.update(policy=policy, partial_advanced=True)

        def select(q, limits, jacobian, previous, offset):
            started = time.monotonic()
            decision = policy.select(q, limits, jacobian, previous, offset)
            decision["total_decision_seconds"] = time.monotonic() - started
            return decision

        def complete(actual_env, command, offset):
            period = receipt["continuation_partial"]["periods"][-1]
            history = policy.complete(actual_env, command, offset, [item["q"] for item in period["actual_inner"]])
            return dict(
                score=gripper_score(actual_env, opening[offset]), policy_history=history, policy_history_recomputed=True
            )

        return select, complete

    replanned = run_finite_reference_tail(
        env,
        {"sample": sample, "endpoint_receipt": saved[0][1]},
        observers,
        receipt,
        reference_poses=poses,
        reference_openings=openings,
        reference_start_step=replan_step,
        consumed=0,
        max_segments=5,
        previous_command=actual[-1]["command"],
        previous_period=actual[-1],
        tool=kwargs["case"]["tool"],
        target_object=kwargs["case"]["target_object"],
        open_segment=open_segment,
        state_reader=kwargs.get("state_reader"),
        task_observer=kwargs.get("task_observer"),
        progress=kwargs.get("progress"),
    )
    return dict(prefix107=first, replanned=replanned, trigger_signals=state["signals"])


def run_checker(
    env,
    terminal,
    observers,
    receipt,
    *,
    producer,
    expected_replan_capture,
    load_new_reference,
    prefix_runner,
    gripper_score=None,
    replan_step=107,
    **kwargs,
):
    """Replay only recorded actions without importing or selecting a policy."""
    gripper_score = original.measured_gripper if gripper_score is None else gripper_score
    recorded = producer["runtime"]["continuation"]
    prefix_producer = {"runtime": {**producer["runtime"], "continuation": recorded["prefix107"]}}
    saved = []
    wrapped = tail.remember_endpoint(observers, saved, step=replan_step)
    first = prefix_runner(
        env, terminal, wrapped, receipt, producer=prefix_producer, gripper_score=gripper_score, **kwargs
    )
    actual = prefix_periods(first, replay=True)
    expected = prefix_periods(recorded["prefix107"])
    prefix.require(len(actual) == len(expected) == replan_step, "The complete recorded prefix is required")
    comparison.compare_periods(expected, actual)
    captured = _current_capture(
        env, observers, saved, expected[-1], replan_step, kwargs["runtime_output"], kwargs.get("render_scope")
    )
    capture_check.compare_capture(expected_replan_capture, captured["capture"])
    receipt["replan_capture"] = captured
    sample, _endpoint = saved[0]
    poses, openings = decode_reference(
        captured["snapshot"], captured["capture"], sample, kwargs["case"]["tool"], load_new_reference, step=replan_step
    )

    def open_segment(_window, opening, _anchor, _step, expected_periods):
        if expected_periods is None:
            raise ValueError("A checker can replay only recorded actions")

        def complete(actual_env, _command, offset):
            score = gripper_score(actual_env, opening[offset])
            original.compare_score(score, expected_periods[offset]["completed_feedback"]["score"])
            return dict(score=score, policy_history=None, policy_history_recomputed=False)

        return original.checker_action(expected_periods), complete

    result = run_finite_reference_tail(
        env,
        {"sample": sample, "endpoint_receipt": saved[0][1]},
        observers,
        receipt,
        reference_poses=poses,
        reference_openings=openings,
        reference_start_step=replan_step,
        consumed=0,
        max_segments=5,
        previous_command=actual[-1]["command"],
        previous_period=actual[-1],
        tool=kwargs["case"]["tool"],
        target_object=kwargs["case"]["target_object"],
        open_segment=open_segment,
        recorded_segments=recorded["replanned"]["segments"],
        recorded_stop=recorded["replanned"]["stop"],
        state_reader=kwargs.get("state_reader"),
        task_observer=kwargs.get("task_observer"),
        progress=kwargs.get("progress"),
    )
    return dict(prefix107=first, replanned=result)
