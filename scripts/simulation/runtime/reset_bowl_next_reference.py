"""Continue a verified 88-period bowl run using its next current-image reference."""

import time

import numpy as np

from s2a.references.action_reference import decode_action_targets
from s2a.research.native_runtime.finite_reference_tail import run_finite_reference_tail
from s2a.research.native_runtime.reset_canvas_threads import render_with_two_torch_threads
from s2a.research.native_runtime.verified_terminal_capture import capture_current_endpoint
from scripts.simulation.runtime import audit_reset_multitask_live_recovery as comparison
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import reset_bowl_reference_tail as prior
from scripts.simulation.runtime import reset_live_two_window_replay as capture_check
from scripts.simulation.runtime import run_reset_start_feedback as original
from scripts.wam.tracking.preview import gpu_resume_repro_20261111 as repro

MODEL_SHA = "48138bb8c95ce8cc1ca0b70f925e57c8f16bfb21be0a243495b391a06cf5b8ec"


def prefix_periods(continuation, *, replay=False):
    return prior.prefix_periods(continuation["prefix"], replay=replay) + prior.tail_periods(continuation["tail"])


def decode_reference(snapshot, capture, sample, tool, load_reference):
    """Bind the saved response to the actual re-rendered step-88 observation."""
    response, envelope = load_reference()
    arrays = repro.response_arrays(response)
    np.testing.assert_array_equal(arrays["action"], envelope["action"])
    np.testing.assert_array_equal(arrays["normalized_action"], envelope["normalized_action"])
    prefix.require(
        snapshot["step"] == envelope["step"] == 88
        and snapshot["sim_time_s"] == envelope["sim_time_s"]
        and snapshot["state_sha256"] == envelope["state_sha256"]
        and capture["canvas_png"]["sha256"] == envelope["image_sha256"]
        and envelope["seed"] == 42
        and envelope["model_sha256"] == MODEL_SHA
        and envelope["denoise_calls"] == 35
        and envelope["action_dim_mask"] == [True] * 10 + [False] * 118,
        "The saved reference must match the current step-88 image and state",
    )
    anchor = np.asarray(sample["source_pose"], dtype=float) @ np.asarray(tool, dtype=float)
    return decode_action_targets(anchor, arrays["action"], horizon=40)


def run_producer(
    env,
    terminal,
    observers,
    receipt,
    *,
    previous_88_producer,
    expected_step88_capture,
    load_step88_reference,
    policy_factory=None,
    advance_policy=None,
    gripper_score=None,
    prefix_runner=None,
    **kwargs,
):
    """Rebuild the audited prefix and continue with the new generated H40."""
    if policy_factory is None:
        from s2a.research.joint_control.preview.reset_feedback_policy import policy_inputs
        from s2a.research.joint_control.preview.segmented_reset_policy import build_segmented_reset_policy

        def policy_factory(env, case, terminal, receipt):
            return build_segmented_reset_policy(env, policy_inputs(case, terminal), receipt)

    if advance_policy is None:
        from s2a.research.joint_control.preview.segmented_reset_policy import advance_segmented_reset_policy

        advance_policy = advance_segmented_reset_policy
    prefix_runner = prior.run_producer if prefix_runner is None else prefix_runner
    gripper_score = original.measured_gripper if gripper_score is None else gripper_score
    state, saved = {}, []

    def start(*args):
        policy, owner, previous = policy_factory(*args)
        state.update(policy=policy, owner=owner)
        return policy, owner, previous

    def advance(previous, owner, *args, **options):
        policy = advance_policy(previous, owner, *args, **options)
        state.update(policy=policy, owner=owner)
        return policy

    wrapped = prior.remember_endpoint(observers, saved, step=88)
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
    expected = prefix_periods(previous_88_producer["runtime"]["continuation"])
    comparison.compare_periods(expected[: len(actual)], actual)
    new = dict(segments=[], stop=first["tail"]["stop"], complete=False, generated_periods=0, stop_recomputed=False)
    result = dict(prefix88=first, new_reference=new)
    receipt["next_reference_partial"] = result
    if new["stop"] is not None:
        return result
    if len(actual) != 88 or len(expected) != 88 or len(saved) != 1:
        raise ValueError("The exact 88-period prefix and current endpoint are required")
    sample, endpoint = saved[0]
    render_scope = kwargs.get("render_scope", render_with_two_torch_threads)
    captured = capture_current_endpoint(
        env,
        observers.render_anchor,
        sample,
        endpoint,
        step=88,
        expected_endpoint=expected[-1]["endpoint_receipt"],
        runtime_output=kwargs["runtime_output"],
        render_scope=render_scope,
    )
    capture_check.compare_capture(expected_step88_capture, captured["capture"])
    receipt["step88_reference_capture"] = captured
    poses, openings = decode_reference(
        captured["snapshot"], captured["capture"], sample, kwargs["case"]["tool"], load_step88_reference
    )

    def open_segment(window, opening, anchor, step, expected_periods):
        if expected_periods is not None:
            raise ValueError("A producer cannot use recorded future commands")
        policy = advance(
            state["policy"], state["owner"], window, opening, anchor, generated_step=step, runtime_receipt=receipt
        )

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

    result["new_reference"] = run_finite_reference_tail(
        env,
        {"sample": sample, "endpoint_receipt": endpoint},
        wrapped,
        receipt,
        reference_poses=poses,
        reference_openings=openings,
        reference_start_step=88,
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
    return result


def run_checker(
    env,
    terminal,
    observers,
    receipt,
    *,
    producer,
    expected_step88_capture,
    load_step88_reference,
    gripper_score=None,
    prefix_runner=None,
    **kwargs,
):
    """Replay the reconstructed prefix and only the actually dispatched new reference."""
    prefix_runner = prior.run_checker if prefix_runner is None else prefix_runner
    gripper_score = original.measured_gripper if gripper_score is None else gripper_score
    recorded = producer["runtime"]["continuation"]
    prefix_producer = {"runtime": {**producer["runtime"], "continuation": recorded["prefix88"]}}
    saved = []
    wrapped = prior.remember_endpoint(observers, saved, step=88)
    first = prefix_runner(
        env, terminal, wrapped, receipt, producer=prefix_producer, gripper_score=gripper_score, **kwargs
    )
    actual = prefix_periods(first, replay=True)
    new = dict(
        segments=[], stop=recorded["new_reference"]["stop"], complete=False, generated_periods=0, stop_recomputed=False
    )
    result = dict(prefix88=first, new_reference=new)
    if len(actual) < 88:
        if recorded["new_reference"]["segments"] or new["stop"] is None:
            raise ValueError("An incomplete prefix cannot have dispatched new-reference actions")
        return result
    if len(actual) != 88 or len(saved) != 1:
        raise ValueError("The complete recorded prefix and current endpoint are required")
    sample, endpoint = saved[0]
    expected = prefix_periods(recorded["prefix88"])
    captured = capture_current_endpoint(
        env,
        observers.render_anchor,
        sample,
        endpoint,
        step=88,
        expected_endpoint=expected[-1]["endpoint_receipt"],
        runtime_output=kwargs["runtime_output"],
        render_scope=kwargs.get("render_scope", render_with_two_torch_threads),
    )
    capture_check.compare_capture(expected_step88_capture, captured["capture"])
    receipt["step88_reference_capture"] = captured
    poses, openings = decode_reference(
        captured["snapshot"], captured["capture"], sample, kwargs["case"]["tool"], load_step88_reference
    )

    def open_segment(_window, opening, _anchor, _step, expected_periods):
        if expected_periods is None:
            raise ValueError("A checker must use only recorded commands")

        def complete(actual_env, _command, offset):
            score = gripper_score(actual_env, opening[offset])
            original.compare_score(score, expected_periods[offset]["completed_feedback"]["score"])
            return dict(score=score, policy_history=None, policy_history_recomputed=False)

        return original.checker_action(expected_periods), complete

    result["new_reference"] = run_finite_reference_tail(
        env,
        {"sample": sample, "endpoint_receipt": endpoint},
        wrapped,
        receipt,
        reference_poses=poses,
        reference_openings=openings,
        reference_start_step=88,
        consumed=0,
        max_segments=5,
        previous_command=actual[-1]["command"],
        previous_period=actual[-1],
        tool=kwargs["case"]["tool"],
        target_object=kwargs["case"]["target_object"],
        open_segment=open_segment,
        recorded_segments=recorded["new_reference"]["segments"],
        recorded_stop=recorded["new_reference"]["stop"],
        state_reader=kwargs.get("state_reader"),
        task_observer=kwargs.get("task_observer"),
        progress=kwargs.get("progress"),
    )
    return result
