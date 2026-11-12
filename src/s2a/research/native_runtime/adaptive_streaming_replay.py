"""Independently replay completed adaptive actions and current-image bindings."""

import copy
from types import SimpleNamespace

import numpy as np

from s2a.geometry import validate_transform
from s2a.references.action_reference import decode_action_targets
from s2a.research.native_runtime.development_continuation import run_continuation
from s2a.research.native_runtime.streaming_replay import _check_terminal, _current
from s2a.research.wam.tracking.runtime.adaptive_stream_protocol import AdaptiveStreamLedger
from s2a.research.wam.training.native.native_training import state_sha256


def _snapshot(env, step):
    current = _current(env, step)
    return {"step": current["step"], "sim_time_s": current["time"], "state_sha256": current["sha256"]}


def _stop_at(stop, step):
    if (
        type(stop) is not dict
        or type(stop.get("reason")) is not str
        or not stop["reason"]
        or stop.get("next_action_dispatched") is not False
    ):
        return False
    at_step, after_step = stop.get("at_step"), stop.get("after_step")
    return (type(at_step) is int and after_step is None and at_step == step) or (
        type(after_step) is int and at_step is None and after_step + 1 == step
    )


def _validate_recorded_stream(stream, responses, *, ledger):
    """Reject incomplete dispatches and inconsistent boundaries before replay."""
    if (
        type(stream) is not dict
        or type(stream.get("segments")) is not list
        or type(stream.get("complete")) is not bool
        or type(responses) not in (list, tuple)
        or len(responses) != len(stream["segments"])
        or len(responses) > ledger.max_requests
        or stream.get("requests_issued") != len(responses)
        or stream.get("responses_accepted") != len(responses)
    ):
        raise ValueError("A bounded recorded stream with one response per segment is required")
    segments = stream["segments"]
    expected_step, all_periods = ledger.current["step"], []
    for index, row in enumerate(segments):
        if (
            type(row) is not dict
            or row.get("inflight", "missing") is not None
            or type(row.get("start_step")) is not int
            or row["start_step"] != expected_step
            or type(row.get("continuation")) is not dict
            or type(row["continuation"].get("periods")) is not list
        ):
            raise ValueError("Consecutive completed segments without in-flight actions are required")
        outcome = row["continuation"]
        periods, boundary = outcome["periods"], outcome.get("stop")
        if (
            len(periods) > 8
            or outcome.get("steps_completed") != len(periods)
            or any(
                type(period.get("step")) is not int
                or period["step"] != expected_step + offset
                or type(period.get("actual_inner")) is not list
                or len(period["actual_inner"]) != 25
                for offset, period in enumerate(periods)
            )
        ):
            raise ValueError("Each recorded action must be consecutive with all 25 observed inner steps")
        expected_step += len(periods)
        all_periods.extend(periods)
        if index < len(segments) - 1:
            if len(periods) == 8:
                valid = row.get("boundary") == "window_end" and boundary is None
            else:
                trigger = boundary.get("trigger") if type(boundary) is dict else None
                valid = (
                    1 <= len(periods) < 8
                    and row.get("boundary") == "quality_trigger"
                    and _stop_at(boundary, expected_step)
                    and boundary.get("reason") == "requested_observation"
                    and type(trigger) is dict
                    and trigger.get("replan") is True
                    and type(trigger.get("current_step")) is int
                    and trigger.get("current_step") == expected_step
                    and type(trigger.get("reason")) is str
                    and bool(trigger["reason"])
                )
            if not valid:
                raise ValueError("Every intermediate segment must end at a completed refresh boundary")
        elif len(periods) < 8 and not _stop_at(boundary, expected_step):
            raise ValueError("The final partial segment must contain its actual stop boundary")
        elif len(periods) == 8 and boundary is None and row.get("boundary") != "window_end":
            raise ValueError("A completed final window must record its boundary")
    count = expected_step - ledger.current["step"]
    stop = stream.get("stop")
    if (
        count > ledger.max_periods
        or stream.get("control_steps_attempted") != count
        or stream.get("control_steps_completed") != count
        or stream.get("feedback_periods_committed") != count
        or not _stop_at(stop, expected_step)
        or (stream["complete"] and (count != ledger.max_periods or stop["reason"] != "period_budget_exhausted"))
        or (stop["reason"] == "request_budget_exhausted" and len(responses) != ledger.max_requests)
    ):
        raise ValueError("The producer must account for every dispatched action and its bounded stop")
    if all_periods:
        if any(period.get("inner_below_buffer_samples") for period in all_periods[:-1]):
            raise ValueError("No recorded action may follow an observed joint-buffer loss")
        if all_periods[-1].get("inner_outside_samples"):
            if stop["reason"] != "actual_inner_joint_limit_violation":
                raise ValueError("A recorded joint-limit violation must retain its physical stop")
        elif all_periods[-1].get("inner_below_buffer_samples") and stop["reason"] != "observed_joint_buffer_loss":
            raise ValueError("A recorded joint-buffer loss must retain its physical stop")
    if segments:
        last = segments[-1]["continuation"].get("stop")
        if last is not None and last != stop:
            trigger = last.get("trigger")
            if not (
                last.get("reason") == "requested_observation"
                and stop["reason"] == "request_budget_exhausted"
                and _stop_at(last, expected_step)
                and segments[-1].get("boundary") == "quality_trigger"
                and 1 <= len(segments[-1]["continuation"]["periods"]) < 8
                and type(trigger) is dict
                and trigger.get("replan") is True
                and type(trigger.get("current_step")) is int
                and trigger["current_step"] == expected_step
                and type(trigger.get("reason")) is str
                and bool(trigger["reason"])
            ):
                raise ValueError("The final recorded segment and producer stop disagree")
        if last is None and stop["reason"] not in (
            "period_budget_exhausted",
            "request_budget_exhausted",
            "observed_joint_buffer_loss",
        ):
            raise ValueError("A completed final window must explain its producer stop")
    return count


def replay_adaptive_streaming_continuation(
    env,
    terminal,
    observers,
    receipt,
    *,
    ledger,
    producer_stream,
    responses,
    capture_current,
    previous_command,
    tool,
    target_object,
    score_gripper=None,
    compare_score=None,
    state_reader=None,
    task_observer=None,
    progress=None,
):
    """Replay only recorded commands, decoding responses at reobserved anchors.

    ``capture_current`` independently renders one image from the current
    snapshot, source pose and opening and returns a path/digest relative to the
    checker's capture root. The response is accepted only if that image, state,
    clock and request identity agree. The caller owns producer fingerprint,
    physical resource budgets and checker-process isolation. Recorded quality
    triggers and policy stop reasons are not independently recomputed here.
    """
    if (
        type(ledger) is not AdaptiveStreamLedger
        or ledger.phase != "ready"
        or ledger.issued_requests != 0
        or ledger.completed_periods != 0
        or ledger.current["step"] + ledger.max_periods > 600
        or not callable(capture_current)
        or ((score_gripper is None) != (compare_score is None))
        or (score_gripper is not None and (not callable(score_gripper) or not callable(compare_score)))
    ):
        raise ValueError("One fresh bounded checker ledger and current-image capture are required")
    previous = np.asarray(previous_command, dtype=float).copy()
    tool = validate_transform(np.asarray(tool, dtype=float)).copy()
    if (
        previous.shape != (7,)
        or not np.isfinite(previous).all()
        or np.max(np.abs(previous)) > 1 + 1e-12
        or tool.shape != (4, 4)
        or receipt["control_steps_completed"] != ledger.current["step"]
    ):
        raise ValueError("The actual prefix, previous command and fixed TCP tool are required")
    count = _validate_recorded_stream(producer_stream, responses, ledger=ledger)
    start_attempted, start_completed = receipt["control_steps_attempted"], receipt["control_steps_completed"]
    replay = {
        "segments": [],
        "complete": False,
        "reported_producer_stop": copy.deepcopy(producer_stream["stop"]),
        "stop_reason_independently_verified": False,
        "automatic_retry": False,
    }
    receipt["adaptive_replay_partial"] = replay

    def totals():
        replay["control_steps_attempted"] = receipt["control_steps_attempted"] - start_attempted
        replay["control_steps_completed"] = receipt["control_steps_completed"] - start_completed
        replay["replay_periods_committed"] = ledger.completed_periods

    try:
        if _snapshot(env, ledger.current["step"]) != ledger.current:
            raise ValueError("The checker must begin at its registered current state")
        _check_terminal(terminal, _current(env, ledger.current["step"]))
        for index, (row, response) in enumerate(zip(producer_stream["segments"], responses, strict=True)):
            step = row["start_step"]
            before = _snapshot(env, step)
            if before != ledger.current:
                raise ValueError("Replay state differs from the current completed endpoint")
            _check_terminal(terminal, _current(env, step))
            sample = terminal["sample"]
            source_pose = validate_transform(np.asarray(sample["source_pose"], dtype=float)).copy()
            opening = float(sample["opening"])
            if source_pose.shape != (4, 4) or not np.isfinite(opening) or not 0 <= opening <= 1:
                raise ValueError("One independently measured current pose and opening are required")
            captured_pose = source_pose.copy()
            captured_pose.setflags(write=False)
            observation = {"snapshot": dict(before), "source_pose": captured_pose, "opening": opening}
            image = capture_current(env, observation, index)
            after_capture = _snapshot(env, step)
            request = ledger.issue(before_render=before, after_render=after_capture, image=image)
            actions = ledger.accept(response, after_wait=_snapshot(env, step))
            poses, openings = decode_action_targets(source_pose @ tool, actions, horizon=40)
            expected = row["continuation"]["periods"]
            partial = {"start_step": step, "request": request, "continuation": None}
            replay["segments"].append(partial)

            def recorded_action(_q, _limits, _jacobian, _previous, offset):
                current = ledger.current
                if _snapshot(env, current["step"]) != current:
                    raise ValueError("Replay state changed before recorded command dispatch")
                ledger.begin_period(before=current)
                return {"command": np.asarray(expected[offset]["command"], dtype=float), "newly_selected": False}

            def completed(actual_env, _command, offset):
                period = receipt["continuation_partial"]["periods"][-1]
                before_period = ledger.current
                after_period = _snapshot(actual_env, before_period["step"] + 1)
                endpoint = period["endpoint_receipt"]
                if (
                    endpoint["endpoint_step"] != after_period["step"]
                    or endpoint["sim_time"] != after_period["sim_time_s"]
                    or endpoint["sim_state_sha256"] != after_period["state_sha256"]
                    or state_sha256(period["actual_state"]) != after_period["state_sha256"]
                ):
                    raise ValueError("The replayed endpoint must match the actual current state")
                expected_feedback = expected[offset].get("completed_feedback", {})
                expected_score = expected_feedback.get("score")
                if (expected_score is None) != (score_gripper is None):
                    raise ValueError("Recorded gripper scores require an independent score and comparison")
                feedback = {"policy_history": None, "policy_history_recomputed": False}
                if score_gripper is not None:
                    score = score_gripper(actual_env, openings[offset])
                    compare_score(score, expected_score)
                    feedback["score"] = score
                if _snapshot(actual_env, after_period["step"]) != after_period:
                    raise ValueError("Independent scoring changed the physical state")
                ledger.complete_period(
                    before=before_period, after=after_period, inner_samples=len(period["actual_inner"])
                )
                totals()
                return feedback

            def render_checked(actual_env, size):
                current = ledger.current
                if _snapshot(actual_env, current["step"]) != current:
                    raise ValueError("Replay state changed before rendering")
                value = observers.render_cameras(actual_env, size)
                if _snapshot(actual_env, current["step"]) != current:
                    raise ValueError("Replay rendering changed the physical state")
                return value

            guarded = SimpleNamespace(
                endpoint=observers.endpoint, current=observers.current, render_cameras=render_checked
            )
            try:
                outcome = run_continuation(
                    env,
                    terminal,
                    guarded,
                    receipt,
                    start_step=step,
                    previous_command=previous,
                    targets=poses[:8],
                    tool=tool,
                    target_object=target_object,
                    choose_action=recorded_action,
                    stop_on_outside=True,
                    expected_periods=expected,
                    state_reader=state_reader,
                    task_observer=task_observer,
                    on_completed=completed,
                    progress=progress,
                    continuation_limit=8,
                )
            finally:
                partial["continuation"] = receipt.get("continuation_partial")
            if outcome["steps_completed"] != len(expected):
                raise ValueError("Independent replay did not complete every recorded action")
            if expected:
                previous = np.asarray(expected[-1]["command"], dtype=float).copy()
            if index < len(producer_stream["segments"]) - 1 and row["boundary"] == "quality_trigger":
                if not ledger.refresh_early(trigger=row["continuation"]["stop"]["trigger"]):
                    raise ValueError("A recorded refresh exceeded the checker budget")
            current = _snapshot(env, ledger.current["step"])
            sample, endpoint = observers.endpoint(env, current["step"])
            terminal = {"sample": sample, "endpoint_receipt": endpoint}
            if _snapshot(env, current["step"]) != current:
                raise ValueError("Replay endpoint observation changed the physical state")
            _check_terminal(terminal, _current(env, current["step"]))
            totals()
        if ledger.completed_periods != count:
            raise ValueError("Replay period count differs from the recorded physical count")
        replay["complete"] = True
        totals()
        return replay
    except BaseException as error:
        ledger.abort()
        replay["error"] = {"type": type(error).__name__, "message": str(error)}
        totals()
        raise
    finally:
        if progress is not None:
            progress(receipt)
