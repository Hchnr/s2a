"""Execute or independently replay a bounded reset-origin oracle timeline."""

from types import SimpleNamespace

import numpy as np

from s2a.references.oracle_timeline import OracleTimeline
from s2a.research.native_runtime.adaptive_streaming_continuation import quality_trigger_view
from s2a.research.native_runtime.development_continuation import run_continuation
from s2a.research.native_runtime.endpoint_cache import SameStateEndpointCache
from s2a.research.native_runtime.streaming_continuation import measured_snapshot


def boundary(outcome, *, start, total):
    """Distinguish a completed reference, an early refresh and a real stop."""
    count = outcome["steps_completed"]
    stop = outcome["stop"]
    if type(count) is not int or not 0 <= count <= 8 or start + count > total:
        raise ValueError("The segment cannot exceed its registered period budget")
    if stop is not None and stop.get("reason") == "requested_observation":
        if count == 0 or stop.get("at_step") != start + count or stop.get("next_action_dispatched") is not False:
            raise ValueError("An early refresh requires a completed current period")
        return "quality_trigger"
    if stop is not None and stop.get("reason") == "oracle_reference_complete":
        if start + count != total or stop.get("at_step") != total or stop.get("next_action_dispatched") is not False:
            raise ValueError("The oracle can end only after its registered final period")
        return "reference_complete"
    if stop is not None:
        return "stopped"
    if start + count == total:
        return "reference_complete"
    if count != 8:
        raise ValueError("An incomplete segment requires a recorded reason")
    return "window_end"


def summarize(periods, *, expected_periods):
    """Score actual completed endpoints without changing the original denominator."""
    if type(expected_periods) is not int or expected_periods < 40 or len(periods) > expected_periods:
        raise ValueError("A bounded timeline with a 40-period terminal window is required")
    if any(row.get("step") != index for index, row in enumerate(periods)):
        raise ValueError("Only consecutive reset-origin periods may be scored")
    errors = np.asarray([row["tcp_error"] for row in periods], dtype=float).reshape(-1, 6)
    if not np.isfinite(errors).all():
        raise ValueError("Finite measured TCP errors are required")
    opening = np.asarray([row["completed_feedback"]["score"]["opening_error"] for row in periods], dtype=float)
    if not np.isfinite(opening).all():
        raise ValueError("Finite measured opening errors are required")
    full = len(periods) == expected_periods
    stable = bool(full and all(row["task_success"] is True for row in periods[-40:]))
    if len(periods):
        position_rmse = float(np.sqrt(np.mean(np.sum(errors[:, :3] ** 2, axis=1))))
        rotation_rmse = float(np.sqrt(np.mean(np.sum(errors[:, 3:] ** 2, axis=1))))
        opening_rmse = float(np.sqrt(np.mean(opening**2)))
        minimum_margin = min(row["minimum_inner_margin_rad"] for row in periods)
    else:
        position_rmse = rotation_rmse = opening_rmse = minimum_margin = None
    return dict(
        completed_periods=len(periods),
        expected_periods=expected_periods,
        full_trajectory=full,
        stable_last_40=stable,
        successful_periods=sum(row["task_success"] is True for row in periods),
        grasp_periods=sum(row["target_grasp"] is True for row in periods),
        pose_caps_periods=sum(row["pose_caps_satisfied"] is True for row in periods),
        position_rmse_m=position_rmse,
        rotation_rmse_rad=rotation_rmse,
        opening_rmse=opening_rmse,
        minimum_inner_margin_rad=minimum_margin,
        inner_below_buffer_samples=sum(row["inner_below_buffer_samples"] for row in periods),
        inner_outside_samples=sum(row["inner_outside_samples"] for row in periods),
    )


def run_correct_reference_continuation(
    env,
    terminal,
    observers,
    receipt,
    *,
    timeline,
    tool,
    target_object,
    phase,
    score_gripper,
    compare_score,
    policy_factory=None,
    recorded=None,
    progress=None,
):
    """Use current feedback or replay recorded actions; task labels only score."""
    if (
        not isinstance(timeline, OracleTimeline)
        or phase not in ("producer", "checker")
        or (phase == "producer") != (callable(policy_factory) and recorded is None)
        or (phase == "checker") != (policy_factory is None and isinstance(recorded, dict))
        or receipt["control_steps_completed"] != 0
        or terminal["endpoint_receipt"]["endpoint_step"] != 0
    ):
        raise ValueError("An admitted reset and exactly one producer or checker mode are required")
    tool = np.asarray(tool, dtype=float)
    if tool.shape != (4, 4) or not np.isfinite(tool).all():
        raise ValueError("The fixed TCP tool is required")
    cache = SameStateEndpointCache(observers.endpoint)
    wrapped = SimpleNamespace(endpoint=cache, current=observers.current, render_cameras=observers.render_cameras)
    state = dict(segments=[], stop=None, complete=False, stop_recomputed=False, endpoint_cache_events=cache.events)
    receipt["correct_reference_partial"] = state
    previous = np.zeros(7)
    policy = owner = None
    recorded_segments = None if recorded is None else recorded["segments"]
    limit = timeline.periods if recorded_segments is None else len(recorded_segments)
    last = None
    for index in range(limit):
        start = receipt["control_steps_completed"]
        poses, openings, metadata = timeline.window(start)
        before = measured_snapshot(env, start)
        expected = None if recorded_segments is None else recorded_segments[index]
        if expected is not None and (expected["start_step"] != start or expected["reference"] != metadata):
            raise ValueError("Checker segments must use the recorded timeline and boundary")
        if phase == "producer":
            if index == 0:
                policy, owner, previous = policy_factory(env, terminal, receipt)
                if not np.array_equal(previous, np.zeros(7)):
                    raise ValueError("A reset policy cannot consume a historical action")
            else:
                from s2a.research.joint_control.preview.causal_reference_replan import advance_recorded_segment

                anchor = np.asarray(terminal["sample"]["source_pose"]) @ tool
                policy = advance_recorded_segment(
                    policy, owner, poses, openings, anchor, generated_step=start, runtime_receipt=receipt
                )
        if measured_snapshot(env, start) != before:
            raise ValueError("Opening an oracle segment cannot advance or alter physical state")
        row = dict(start_step=start, reference=metadata, signals=[], continuation=None, boundary=None)
        state["segments"].append(row)
        expected_periods = None if expected is None else expected["continuation"]["periods"]

        def choose(q, limits, jacobian, prior, offset):
            if phase == "checker":
                return dict(command=np.asarray(expected_periods[offset]["command"], dtype=float), newly_selected=False)
            if start + offset >= timeline.periods:
                return {"stop": {"reason": "oracle_reference_complete"}}
            completed = receipt["continuation_partial"]["periods"]
            if completed and completed[-1]["inner_below_buffer_samples"]:
                return {"stop": {"reason": "observed_joint_buffer_loss"}}
            if completed:
                from s2a.research.joint_control.preview.causal_reference_replan import completed_quality_margin

                signal = completed_quality_margin(quality_trigger_view(completed[-1]), current_step=start + offset)
                row["signals"].append(signal)
                if signal["replan"]:
                    return {"stop": {"reason": "requested_observation", "trigger": signal}}
            return policy.select(q, limits, jacobian, prior, offset)

        def completed(actual_env, command, offset):
            period = receipt["continuation_partial"]["periods"][-1]
            history = None
            if phase == "producer":
                history = policy.complete(
                    actual_env, command, offset, [sample["q"] for sample in period["actual_inner"]]
                )
            score = score_gripper(actual_env, openings[offset])
            if phase == "checker":
                compare_score(score, expected_periods[offset]["completed_feedback"]["score"])
            return dict(score=score, policy_history=history, policy_history_recomputed=phase == "producer")

        try:
            outcome = run_continuation(
                env,
                terminal,
                wrapped,
                receipt,
                start_step=start,
                previous_command=previous,
                targets=poses[:8],
                tool=tool,
                target_object=target_object,
                choose_action=choose,
                stop_on_outside=True,
                expected_periods=expected_periods,
                on_completed=completed,
                progress=progress,
            )
        finally:
            row["continuation"] = receipt.get("continuation_partial")
        periods = outcome["periods"]
        if phase == "checker":
            reported_stop = expected["continuation"]["stop"]
            if reported_stop is not None and reported_stop["reason"] == "actual_inner_joint_limit_violation":
                if outcome["stop"] != reported_stop:
                    raise ValueError("Physical stops must also be observed by the checker")
            elif outcome["stop"] is not None:
                raise ValueError("The checker observed an unexpected physical stop")
            else:
                outcome["stop"] = reported_stop
            row["boundary"] = expected["boundary"]
        else:
            row["boundary"] = boundary(outcome, start=start, total=timeline.periods)
        if periods:
            last = periods[-1]
            previous = np.asarray(last["command"], dtype=float).copy()
        if phase == "producer" and row["boundary"] == "stopped":
            state["stop"] = outcome["stop"]
            break
        if phase == "producer" and row["boundary"] == "reference_complete":
            state["complete"] = True
            state["stop"] = outcome["stop"]
            break
        if phase == "producer" and last is not None and last["inner_below_buffer_samples"]:
            state["stop"] = dict(
                reason="observed_joint_buffer_loss", at_step=start + len(periods), next_action_dispatched=False
            )
            break
        if phase == "checker" and index == len(recorded_segments) - 1:
            state["complete"], state["stop"] = recorded["complete"], recorded["stop"]
            break
        step = receipt["control_steps_completed"]
        sample, endpoint = cache(env, step)
        terminal = dict(sample=sample, endpoint_receipt=endpoint)
    periods = [item for row in state["segments"] for item in row["continuation"]["periods"]]
    state["summary"] = summarize(periods, expected_periods=timeline.periods)
    if state["complete"] != state["summary"]["full_trajectory"]:
        raise ValueError("Only all registered periods can complete the oracle timeline")
    return state
