"""Continue an independently verified 56-period bowl prefix through an H40 tail."""

import copy
import time
from types import SimpleNamespace

from s2a.research.native_runtime.finite_reference_tail import run_finite_reference_tail
from scripts.simulation.runtime import audit_reset_multitask_live_recovery as comparison
from scripts.simulation.runtime import reset_bowl_h40_second_window as prior
from scripts.simulation.runtime import run_reset_start_feedback as original


def prefix_periods(continuation, *, replay=False):
    """Flatten the actual 56-period producer or checker prefix."""
    if replay:
        first = continuation["first_window"]["periods"] + [
            row for segment in continuation["replays"] for row in segment["periods"]
        ]
    else:
        first = prior.periods(continuation)
    return first + continuation["second"]["periods"]


def tail_periods(tail):
    return [row for segment in tail["segments"] for row in segment["continuation"]["periods"]]


def remember_endpoint(observers, saved, *, step):
    def endpoint(env, current_step):
        sample, evidence = observers.endpoint(env, current_step)
        if current_step == step:
            saved.append((copy.deepcopy(sample), copy.deepcopy(evidence)))
        return sample, evidence

    return SimpleNamespace(
        endpoint=endpoint,
        current=observers.current,
        render_anchor=observers.render_anchor,
        render_cameras=observers.render_cameras,
    )


def reference_targets(capture, tool, load_second_reference):
    """Decode the already verified step-48 reference without observing future state."""
    endpoint = capture["current_endpoint"]
    current = {
        "step": endpoint["endpoint_step"],
        "sim_time_s": endpoint["sim_time"],
        "state_sha256": endpoint["sim_state_sha256"],
    }
    sample = {"source_pose": capture["current_source_pose"]}
    _anchor, poses, openings = prior.decode_saved_reference(current, capture, sample, tool, load_second_reference)
    return poses, openings


def run_producer(
    env,
    terminal,
    observers,
    receipt,
    *,
    previous_56_producer,
    policy_factory=None,
    advance_policy=None,
    gripper_score=None,
    **kwargs,
):
    """Rebuild the audited history, then consume only remaining generated targets."""
    if policy_factory is None:
        from s2a.research.joint_control.preview.reset_feedback_policy import policy_inputs
        from s2a.research.joint_control.preview.segmented_reset_policy import build_segmented_reset_policy

        def policy_factory(env, case, terminal, receipt):
            return build_segmented_reset_policy(env, policy_inputs(case, terminal), receipt)

    if advance_policy is None:
        from s2a.research.joint_control.preview.segmented_reset_policy import advance_segmented_reset_policy

        advance_policy = advance_segmented_reset_policy
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

    wrapped = remember_endpoint(observers, saved, step=56)
    first = prior.run_producer(
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
    expected = prefix_periods(previous_56_producer["runtime"]["continuation"])
    comparison.compare_periods(expected[: len(actual)], actual)
    tail = dict(segments=[], stop=first["second"]["stop"], complete=False, generated_periods=0, stop_recomputed=False)
    result = dict(prefix=first, tail=tail)
    receipt["bowl_tail_partial"] = result
    if tail["stop"] is not None:
        return result
    if len(actual) != 56 or len(expected) != 56 or len(saved) != 1:
        raise ValueError("The exact 56-period prefix and current endpoint are required")
    sample, endpoint = saved[0]
    poses, openings = reference_targets(
        kwargs["expected_second_capture"], kwargs["case"]["tool"], kwargs["load_second_reference"]
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

    tail = run_finite_reference_tail(
        env,
        {"sample": sample, "endpoint_receipt": endpoint},
        wrapped,
        receipt,
        reference_poses=poses,
        reference_openings=openings,
        reference_start_step=48,
        consumed=8,
        max_segments=4,
        previous_command=actual[-1]["command"],
        previous_period=actual[-1],
        tool=kwargs["case"]["tool"],
        target_object=kwargs["case"]["target_object"],
        open_segment=open_segment,
        state_reader=kwargs.get("state_reader"),
        task_observer=kwargs.get("task_observer"),
        progress=kwargs.get("progress"),
    )
    result["tail"] = tail
    return result


def run_checker(env, terminal, observers, receipt, *, producer, gripper_score=None, **kwargs):
    """Replay the prefix and remaining actually dispatched reference targets."""
    gripper_score = original.measured_gripper if gripper_score is None else gripper_score
    recorded = producer["runtime"]["continuation"]
    prefix_producer = {"runtime": {**producer["runtime"], "continuation": recorded["prefix"]}}
    saved = []
    wrapped = remember_endpoint(observers, saved, step=56)
    first = prior.run_checker(
        env, terminal, wrapped, receipt, producer=prefix_producer, gripper_score=gripper_score, **kwargs
    )
    actual = prefix_periods(first, replay=True)
    tail = dict(segments=[], stop=recorded["tail"]["stop"], complete=False, generated_periods=0, stop_recomputed=False)
    result = dict(prefix=first, tail=tail)
    if len(actual) < 56:
        if recorded["tail"]["segments"] or tail["stop"] is None:
            raise ValueError("An incomplete prefix cannot have dispatched tail actions")
        return result
    if len(actual) != 56 or len(saved) != 1:
        raise ValueError("The complete recorded prefix and current endpoint are required")
    sample, endpoint = saved[0]
    poses, openings = reference_targets(
        kwargs["expected_second_capture"], kwargs["case"]["tool"], kwargs["load_second_reference"]
    )

    def open_segment(_window, opening, _anchor, _step, expected):
        if expected is None:
            raise ValueError("A checker must use only recorded commands")

        def complete(actual_env, _command, offset):
            score = gripper_score(actual_env, opening[offset])
            original.compare_score(score, expected[offset]["completed_feedback"]["score"])
            return dict(score=score, policy_history=None, policy_history_recomputed=False)

        return original.checker_action(expected), complete

    tail = run_finite_reference_tail(
        env,
        {"sample": sample, "endpoint_receipt": endpoint},
        wrapped,
        receipt,
        reference_poses=poses,
        reference_openings=openings,
        reference_start_step=48,
        consumed=8,
        max_segments=4,
        previous_command=actual[-1]["command"],
        previous_period=actual[-1],
        tool=kwargs["case"]["tool"],
        target_object=kwargs["case"]["target_object"],
        open_segment=open_segment,
        recorded_segments=recorded["tail"]["segments"],
        recorded_stop=recorded["tail"]["stop"],
        state_reader=kwargs.get("state_reader"),
        task_observer=kwargs.get("task_observer"),
        progress=kwargs.get("progress"),
    )
    result["tail"] = tail
    return result
