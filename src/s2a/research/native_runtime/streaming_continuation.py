"""Sequence existing eight-period continuations without future references."""

import copy
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Callable

import numpy as np

from s2a.geometry import validate_transform
from s2a.research.native_runtime.development_continuation import current_joint_site_state, run_continuation
from s2a.research.wam.tracking.runtime.causal_stream_protocol import CausalStreamLedger
from s2a.research.wam.training.native.native_training import state_sha256


@dataclass(frozen=True)
class StreamingSegment:
    """One admitted reference and its staged-action callbacks."""

    targets: np.ndarray
    select: Callable
    complete: Callable


def measured_snapshot(env, step):
    """Read the actual flattened simulation state and clock without advancing."""
    if type(step) is not int or step < 0:
        raise ValueError("A nonnegative current control step is required")
    state = np.asarray(env.sim.get_state().flatten())
    clock = float(env.sim.data.time)
    if not np.isfinite(clock) or clock < 0:
        raise ValueError("Finite nonnegative simulation time required")
    return {"step": step, "sim_time_s": clock, "state_sha256": state_sha256(state)}


def _terminal_input(terminal, snapshot):
    if type(terminal) is not dict or set(terminal) != {"sample", "endpoint_receipt"}:
        raise ValueError("The actual current endpoint and receipt are required")
    sample, endpoint = terminal["sample"], terminal["endpoint_receipt"]
    if (
        endpoint["endpoint_step"] != snapshot["step"]
        or endpoint["sim_time"] != snapshot["sim_time_s"]
        or endpoint["sim_state_sha256"] != snapshot["state_sha256"]
        or state_sha256(sample["sim_states"]) != snapshot["state_sha256"]
    ):
        raise ValueError("The endpoint must match the current physical state and clock")
    pose = validate_transform(np.asarray(sample["source_pose"], dtype=float)).copy()
    opening = float(sample["opening"])
    if pose.shape != (4, 4) or not np.isfinite(opening) or not 0 <= opening <= 1:
        raise ValueError("One current measured pose and gripper opening are required")
    pose.setflags(write=False)
    return {"snapshot": dict(snapshot), "source_pose": pose, "opening": opening}


def _prefix_safe(receipt, current_q, current_limits, step, clock):
    inner = receipt.get("final_prefix_inner")
    if type(inner) is not dict or inner.get("prefix_step") != step - 1 or len(inner.get("samples", [])) != 25:
        raise ValueError("The 25 actual final-prefix inner samples are required")
    limits = np.asarray(inner["limits"], dtype=float)
    samples = inner["samples"]
    q = np.asarray([item["q"] for item in samples], dtype=float)
    velocity = np.asarray([item["qvel"] for item in samples], dtype=float)
    times = np.asarray([item["time"] for item in samples], dtype=float)
    if (
        limits.shape != (7, 2)
        or q.shape != (25, 7)
        or velocity.shape != (25, 7)
        or times.shape != (25,)
        or not all(np.isfinite(value).all() for value in (limits, q, velocity, times))
        or np.any(limits[:, 0] >= limits[:, 1])
        or not np.allclose(np.diff(times), 0.002, atol=1e-12, rtol=0)
        or not np.isclose(times[-1], clock, atol=1e-10, rtol=0)
        or not np.allclose(q[-1], current_q, atol=1e-10, rtol=0)
        or not np.array_equal(limits, current_limits)
    ):
        raise ValueError("Final-prefix samples must match the current state and 25-step clock")
    margins = np.minimum(q - limits[:, 0], limits[:, 1] - q)
    below = int(np.sum(np.any(margins < 0.05, axis=1)))
    outside = int(np.sum(np.any(margins < 0, axis=1)))
    if inner.get("below_buffer_samples") != below or inner.get("outside_samples") != outside:
        raise ValueError("Final-prefix safety counts differ from the actual inner samples")
    return below == 0


def run_streaming_continuation(
    env,
    terminal,
    observers,
    receipt,
    *,
    ledger,
    max_windows,
    previous_command,
    tool,
    target_object,
    open_segment,
    state_reader=None,
    task_observer=None,
    progress=None,
):
    """Run separately admitted eight-period references under one causal ledger.

    ``open_segment`` gets only the current measured pose/opening and snapshot,
    plus the previous segment, and must issue/accept exactly one ledger request.
    It is trusted runtime code, not a model input or independent safety checker.
    The parent runtime must independently admit the total physical budget.
    """
    if (
        type(ledger) is not CausalStreamLedger
        or ledger.phase != "ready"
        or ledger.issued_requests != 0
        or type(max_windows) is not int
        or not 1 <= max_windows <= 6
        or max_windows != ledger.max_requests
        or not callable(open_segment)
    ):
        raise ValueError("One fresh ledger and explicit bounded development budget are required")
    state_reader = current_joint_site_state if state_reader is None else state_reader
    previous_command = np.asarray(previous_command, dtype=float).copy()
    tool = validate_transform(np.asarray(tool, dtype=float)).copy()
    if (
        previous_command.shape != (7,)
        or not np.isfinite(previous_command).all()
        or np.max(np.abs(previous_command)) > 1 + 1e-12
        or tool.shape != (4, 4)
    ):
        raise ValueError("A finite previous command and fixed TCP tool are required")
    start_attempted = receipt["control_steps_attempted"]
    start_completed = receipt["control_steps_completed"]
    stream = {"segments": [], "stop": None, "complete": False, "automatic_retry": False}
    receipt["streaming_partial"] = stream

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
        ledger.abort()

    try:
        step = ledger.current["step"]
        snapshot = measured_snapshot(env, step)
        if snapshot != ledger.current:
            raise ValueError("The ledger must begin at the actual current state")
        _terminal_input(terminal, snapshot)
        q, limits, _jacobian = state_reader(env)
        if not _prefix_safe(
            receipt, np.asarray(q, dtype=float), np.asarray(limits, dtype=float), step, snapshot["sim_time_s"]
        ):
            stop("observed_prefix_joint_buffer_loss")
            totals()
            return stream
        prior = None
        for _index in range(max_windows):
            step = ledger.current["step"]
            before = measured_snapshot(env, step)
            if before != ledger.current:
                raise ValueError("The physical state changed outside a completed period")
            q, limits, _jacobian = state_reader(env)
            q, limits = np.asarray(q, dtype=float), np.asarray(limits, dtype=float)
            if (
                q.shape != (7,)
                or limits.shape != (7, 2)
                or not np.isfinite(q).all()
                or not np.isfinite(limits).all()
                or np.any(limits[:, 0] >= limits[:, 1])
            ):
                raise ValueError("Finite measured joints and limits are required")
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
            row = {"start_step": step, "continuation": None, "inflight": None}
            stream["segments"].append(row)

            def choose(q, limits, jacobian, previous, offset):
                current = ledger.current
                if measured_snapshot(env, current["step"]) != current:
                    raise ValueError("State changed before action selection")
                past = receipt["continuation_partial"]["periods"]
                if past and past[-1]["inner_below_buffer_samples"]:
                    return {"stop": {"reason": "observed_joint_buffer_loss"}}
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

            missing = object()
            original_attribute = vars(env).get("step", missing)
            original_env_step = env.step

            def tracked_step(command, *args, **kwargs):
                if row["inflight"] is not None:
                    raise ValueError("A previous dispatched action remains incomplete")
                inflight = {
                    "step": ledger.current["step"],
                    "command": np.asarray(command, dtype=float).copy(),
                    "inner_attempted": 0,
                    "inner_completed": 0,
                    "actual_inner": [],
                }
                row["inflight"] = inflight
                original_sim_step = env.sim.step

                def tracked_inner(*inner_args, **inner_kwargs):
                    inflight["inner_attempted"] += 1
                    value = original_sim_step(*inner_args, **inner_kwargs)
                    inflight["inner_completed"] += 1
                    robot, sim = env.robots[0], env.sim
                    inflight["actual_inner"].append(
                        {
                            "time": float(sim.data.time),
                            "q": np.asarray(sim.data.qpos)[robot._ref_joint_pos_indexes].copy(),
                            "qvel": np.asarray(sim.data.qvel)[robot._ref_joint_vel_indexes].copy(),
                        }
                    )
                    return value

                env.sim.step = tracked_inner
                try:
                    return original_env_step(command, *args, **kwargs)
                finally:
                    env.sim.step = original_sim_step

            guarded_observers = SimpleNamespace(
                endpoint=observers.endpoint, current=observers.current, render_cameras=render_checked
            )
            env.step = tracked_step
            try:
                outcome = run_continuation(
                    env,
                    terminal,
                    guarded_observers,
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
                if original_attribute is missing:
                    del env.step
                else:
                    env.step = original_attribute
                row["continuation"] = receipt.get("continuation_partial")
            if outcome["stop"] is not None:
                stream["stop"] = outcome["stop"]
                ledger.abort()
                break
            if outcome["steps_completed"] != 8:
                raise ValueError("A complete segment must contain eight actual periods")
            periods = outcome["periods"]
            previous_command = np.asarray(periods[-1]["command"], dtype=float).copy()
            if periods[-1]["inner_below_buffer_samples"]:
                stop("observed_joint_buffer_loss")
                break
            before_endpoint = measured_snapshot(env, ledger.current["step"])
            sample, endpoint = observers.endpoint(env, ledger.current["step"])
            terminal = {"sample": sample, "endpoint_receipt": endpoint}
            if measured_snapshot(env, ledger.current["step"]) != before_endpoint:
                raise ValueError("Endpoint observation changed the physical state")
            _terminal_input(terminal, before_endpoint)
            prior = segment
        else:
            if ledger.phase != "done":
                raise ValueError("The registered request and physical budgets did not complete together")
            stream["complete"] = True
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
