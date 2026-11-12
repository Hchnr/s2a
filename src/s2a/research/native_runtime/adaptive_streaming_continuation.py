"""Execute bounded, partially consumed references with causal refresh events."""

import copy
from types import SimpleNamespace

import numpy as np

from s2a.geometry import validate_transform
from s2a.research.native_runtime.development_continuation import current_joint_site_state, run_continuation
from s2a.research.native_runtime.streaming_continuation import (
    StreamingSegment,
    _prefix_safe,
    _terminal_input,
    measured_snapshot,
)
from s2a.research.native_runtime.streaming_dispatch import track_stream_dispatch
from s2a.research.wam.tracking.runtime.adaptive_stream_protocol import AdaptiveStreamLedger
from s2a.research.wam.training.native.native_training import state_sha256


def quality_trigger_view(period):
    """Expose only the completed-period information used by the quality trigger."""
    verification = period["decision"]["quality"]["verification"]
    history = period["completed_feedback"]["policy_history"]["past_inner_error"]
    keys = (
        "passed",
        "first_pose_caps_satisfied",
        "hard_pose_prefix",
        "prefix_pose_caps_satisfied",
        "prefix_pose_caps_enforced",
        "tcp_errors",
    )
    return {
        "step": period["step"],
        "endpoint_receipt": {"endpoint_step": period["endpoint_receipt"]["endpoint_step"]},
        "actual_inner": copy.deepcopy(period["actual_inner"]),
        "decision": {
            "quality": {"verification": {key: copy.deepcopy(verification[key]) for key in keys if key in verification}}
        },
        "completed_feedback": {
            "policy_history": {
                "past_inner_error": {
                    "last_completed_step": history.get("last_completed_step"),
                    "future_guarantee": history.get("future_guarantee"),
                }
            }
        },
    }


def _current_joints(env, ledger, state_reader):
    before = ledger.current
    q, limits, jacobian = (np.asarray(value, dtype=float) for value in state_reader(env))
    if (
        measured_snapshot(env, before["step"]) != before
        or q.shape != (7,)
        or limits.shape != (7, 2)
        or jacobian.shape != (6, 7)
        or not all(np.isfinite(value).all() for value in (q, limits, jacobian))
        or np.any(limits[:, 0] >= limits[:, 1])
    ):
        raise ValueError("Finite current joints, limits and Jacobian without state changes are required")
    return q, limits


def run_adaptive_streaming_continuation(
    env,
    terminal,
    observers,
    receipt,
    *,
    ledger,
    previous_command,
    tool,
    target_object,
    open_segment,
    refresh_signal,
    state_reader=None,
    task_observer=None,
    progress=None,
):
    """Run admitted references until a safety stop or an explicit budget ends.

    ``open_segment`` receives only the current measured snapshot, pose and
    opening, plus the preceding segment. It must issue and accept one reference
    without advancing physics. ``refresh_signal`` receives a restricted view of
    a completed action and its committed feedback; it cannot inspect task labels.
    The caller owns the registered model, worker lifecycle and independent replay.
    """
    if (
        type(ledger) is not AdaptiveStreamLedger
        or ledger.phase != "ready"
        or ledger.issued_requests != 0
        or ledger.completed_periods != 0
        or ledger.current["step"] + ledger.max_periods > 600
        or not callable(open_segment)
        or not callable(refresh_signal)
    ):
        raise ValueError("A fresh adaptive ledger within the absolute 600-period budget is required")
    state_reader = current_joint_site_state if state_reader is None else state_reader
    previous_command = np.asarray(previous_command, dtype=float).copy()
    tool = validate_transform(np.asarray(tool, dtype=float)).copy()
    if (
        previous_command.shape != (7,)
        or not np.isfinite(previous_command).all()
        or np.max(np.abs(previous_command)) > 1 + 1e-12
        or tool.shape != (4, 4)
        or receipt["control_steps_completed"] != ledger.current["step"]
    ):
        raise ValueError("The actual prefix, previous command and fixed TCP tool are required")
    start_attempted = receipt["control_steps_attempted"]
    start_completed = receipt["control_steps_completed"]
    stream = {"segments": [], "stop": None, "complete": False, "automatic_retry": False}
    receipt["adaptive_stream_partial"] = stream

    def totals():
        stream.update(
            requests_issued=ledger.issued_requests,
            responses_accepted=ledger.accepted_requests,
            feedback_periods_committed=ledger.completed_periods,
            control_steps_attempted=receipt["control_steps_attempted"] - start_attempted,
            control_steps_completed=receipt["control_steps_completed"] - start_completed,
        )

    def stop(reason):
        stream["stop"] = {"reason": reason, "at_step": ledger.current["step"], "next_action_dispatched": False}
        if ledger.phase != "done":
            ledger.abort()

    try:
        step = ledger.current["step"]
        snapshot = measured_snapshot(env, step)
        if snapshot != ledger.current:
            raise ValueError("The ledger must begin at the actual current state")
        _terminal_input(terminal, snapshot)
        q, limits = _current_joints(env, ledger, state_reader)
        if not _prefix_safe(receipt, q, limits, step, snapshot["sim_time_s"]):
            stop("observed_prefix_joint_buffer_loss")
            totals()
            return stream
        prior = None
        for _index in range(ledger.max_requests):
            step = ledger.current["step"]
            before = measured_snapshot(env, step)
            if before != ledger.current:
                raise ValueError("The physical state changed outside a completed period")
            q, limits = _current_joints(env, ledger, state_reader)
            if np.min(np.minimum(q - limits[:, 0], limits[:, 1] - q)) < 0.05:
                stop("observed_joint_buffer_loss")
                break
            observation = _terminal_input(terminal, before)
            old_issued, old_accepted = ledger.issued_requests, ledger.accepted_requests
            segment = open_segment(observation, prior)
            if (
                type(segment) is not StreamingSegment
                or not callable(segment.select)
                or not callable(segment.complete)
                or ledger.phase != "executing"
                or ledger.issued_requests != old_issued + 1
                or ledger.accepted_requests != old_accepted + 1
                or ledger.current != before
                or measured_snapshot(env, step) != before
            ):
                raise ValueError("One current reference must be accepted without advancing physics")
            targets = validate_transform(np.asarray(segment.targets, dtype=float)).copy()
            if targets.shape != (8, 4, 4):
                raise ValueError("Exactly eight targets from the admitted reference are required")
            row = {"start_step": step, "continuation": None, "inflight": None, "signals": []}
            stream["segments"].append(row)

            def choose(q, limits, jacobian, previous, offset):
                current = ledger.current
                if measured_snapshot(env, current["step"]) != current:
                    raise ValueError("State changed before action selection")
                if ledger.phase == "done":
                    return {"stop": {"reason": "period_budget_exhausted"}}
                past = receipt["continuation_partial"]["periods"]
                if past and past[-1]["inner_below_buffer_samples"]:
                    return {"stop": {"reason": "observed_joint_buffer_loss"}}
                if past:
                    signal = refresh_signal(quality_trigger_view(past[-1]), current_step=current["step"])
                    if (
                        type(signal) is not dict
                        or type(signal.get("replan")) is not bool
                        or type(signal.get("current_step")) is not int
                        or signal["current_step"] != current["step"]
                        or type(signal.get("reason")) is not str
                        or not signal["reason"]
                        or measured_snapshot(env, current["step"]) != current
                    ):
                        raise ValueError("A state-preserving current completed-period refresh signal is required")
                    row["signals"].append(copy.deepcopy(signal))
                    if signal["replan"]:
                        return {"stop": {"reason": "requested_observation", "trigger": copy.deepcopy(signal)}}
                decision = segment.select(q, limits, jacobian, previous, offset)
                if measured_snapshot(env, current["step"]) != current:
                    raise ValueError("Action selection must not advance physics")
                return decision

            def render_checked(actual_env, size):
                current = ledger.current
                if measured_snapshot(actual_env, current["step"]) != current:
                    raise ValueError("State changed before rendering")
                result = observers.render_cameras(actual_env, size)
                if measured_snapshot(actual_env, current["step"]) != current:
                    raise ValueError("Rendering changed the current state")
                return result

            def completed(actual_env, command, offset):
                period = receipt["continuation_partial"]["periods"][-1]
                inflight = row["inflight"]
                if (
                    type(inflight) is not dict
                    or inflight["inner_attempted"] != 25
                    or inflight["inner_completed"] != 25
                    or len(inflight["actual_inner"]) != 25
                    or not np.array_equal(inflight["command"], command)
                ):
                    raise ValueError("The dispatched action and all 25 inner samples must be observed")
                for recorded, observed in zip(period["actual_inner"], inflight["actual_inner"], strict=True):
                    if recorded["time"] != observed["time"] or any(
                        not np.array_equal(recorded[key], observed[key]) for key in ("q", "qvel")
                    ):
                        raise ValueError("The continuation and dispatch inner samples differ")
                before_period = ledger.current
                after_period = measured_snapshot(actual_env, before_period["step"] + 1)
                endpoint = period["endpoint_receipt"]
                if (
                    endpoint["endpoint_step"] != after_period["step"]
                    or endpoint["sim_time"] != after_period["sim_time_s"]
                    or endpoint["sim_state_sha256"] != after_period["state_sha256"]
                    or state_sha256(period["actual_state"]) != after_period["state_sha256"]
                ):
                    raise ValueError("The completed endpoint must match the actual current state")
                feedback = segment.complete(actual_env, command.copy(), offset, copy.deepcopy(period["actual_inner"]))
                if measured_snapshot(actual_env, after_period["step"]) != after_period:
                    raise ValueError("Feedback commit changed the physical state")
                ledger.complete_period(
                    before=before_period, after=after_period, inner_samples=len(period["actual_inner"])
                )
                row["inflight"] = None
                totals()
                return feedback

            guarded = SimpleNamespace(
                endpoint=observers.endpoint, current=observers.current, render_cameras=render_checked
            )
            try:
                with track_stream_dispatch(env, ledger, row):
                    outcome = run_continuation(
                        env,
                        terminal,
                        guarded,
                        receipt,
                        start_step=step,
                        previous_command=previous_command,
                        targets=targets,
                        tool=tool,
                        target_object=target_object,
                        choose_action=choose,
                        stop_on_outside=True,
                        state_reader=state_reader,
                        task_observer=task_observer,
                        on_completed=completed,
                        progress=progress,
                        continuation_limit=8,
                    )
            finally:
                row["continuation"] = receipt.get("continuation_partial")
            periods, boundary = outcome["periods"], outcome["stop"]
            if boundary is not None and boundary["reason"] == "requested_observation":
                if not periods or boundary["at_step"] != ledger.current["step"] or row["inflight"] is not None:
                    raise ValueError("An early refresh requires a fully completed current segment")
                row["boundary"] = "quality_trigger"
                if not ledger.refresh_early(trigger=boundary["trigger"]):
                    stop("request_budget_exhausted")
                    break
            elif boundary is not None:
                stream["stop"] = boundary
                if boundary["reason"] == "period_budget_exhausted":
                    stream["complete"] = True
                else:
                    ledger.abort()
                break
            elif outcome["steps_completed"] == 8:
                row["boundary"] = "window_end"
                if periods[-1]["inner_below_buffer_samples"]:
                    stop("observed_joint_buffer_loss")
                    break
                if ledger.phase == "done":
                    reason = (
                        "period_budget_exhausted"
                        if ledger.completed_periods == ledger.max_periods
                        else "request_budget_exhausted"
                    )
                    stop(reason)
                    stream["complete"] = reason == "period_budget_exhausted"
                    break
            else:
                raise ValueError("A segment must end at a registered event or after eight periods")
            previous_command = np.asarray(periods[-1]["command"], dtype=float).copy()
            before_endpoint = measured_snapshot(env, ledger.current["step"])
            sample, endpoint = observers.endpoint(env, ledger.current["step"])
            terminal = {"sample": sample, "endpoint_receipt": endpoint}
            if measured_snapshot(env, ledger.current["step"]) != before_endpoint:
                raise ValueError("Endpoint observation changed the physical state")
            _terminal_input(terminal, before_endpoint)
            prior = segment
        else:
            raise ValueError("The adaptive request loop ended without a registered stop")
        totals()
        return stream
    except BaseException as error:
        ledger.abort()
        stream["error"] = {"type": type(error).__name__, "message": str(error)}
        totals()
        raise
    finally:
        if progress is not None:
            progress(receipt)
