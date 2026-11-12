"""Consume or independently replay a bounded tail of one generated H40."""

import copy
from types import SimpleNamespace

import numpy as np

from s2a.geometry import validate_transform
from s2a.references.finite_horizon_window import remaining_h40
from s2a.research.native_runtime.development_continuation import current_joint_site_state, run_continuation
from s2a.research.native_runtime.endpoint_cache import SameStateEndpointCache
from s2a.research.native_runtime.streaming_continuation import _terminal_input, measured_snapshot
from s2a.research.wam.training.native.native_training import state_sha256


def run_finite_reference_tail(
    env,
    terminal,
    observers,
    receipt,
    *,
    reference_poses,
    reference_openings,
    reference_start_step,
    consumed,
    max_segments,
    previous_command,
    previous_period,
    tool,
    target_object,
    open_segment,
    recorded_segments=None,
    recorded_stop=None,
    state_reader=None,
    task_observer=None,
    progress=None,
):
    """Consume generated endpoints in 8-period segments, with marked preview holds.

    ``open_segment`` receives only the already-generated preview, current TCP
    anchor, step, and (for replay) the recorded segment's actual periods. It
    returns ``(select, complete)`` callbacks for the fixed continuation runner.
    The caller owns the parent physical budget and provenance of the reference.
    In replay mode this function never claims to recompute a producer stop.
    """
    replay = recorded_segments is not None
    if (
        type(reference_start_step) is not int
        or reference_start_step < 0
        or type(consumed) is not int
        or consumed not in (0, 8, 16, 24, 32)
        or type(max_segments) is not int
        or not 1 <= max_segments <= (40 - consumed) // 8
        or reference_start_step + consumed + 8 * max_segments > 600
        or not callable(open_segment)
        or (replay and (type(recorded_segments) is not list or len(recorded_segments) > max_segments))
        or (not replay and recorded_stop is not None)
    ):
        raise ValueError("A finite generated reference and explicitly bounded segment count are required")
    if replay:
        try:
            lengths = [len(row["continuation"]["periods"]) for row in recorded_segments]
        except (KeyError, TypeError):
            raise ValueError("Complete recorded segment receipts are required") from None
        next_step = reference_start_step + consumed + sum(lengths)
        valid_stop = (
            type(recorded_stop) is dict
            and type(recorded_stop.get("reason")) is str
            and bool(recorded_stop["reason"])
            and recorded_stop.get("next_action_dispatched") is False
            and (
                (
                    type(recorded_stop.get("at_step")) is int
                    and "after_step" not in recorded_stop
                    and recorded_stop["at_step"] == next_step
                )
                or (
                    type(recorded_stop.get("after_step")) is int
                    and "at_step" not in recorded_stop
                    and recorded_stop["after_step"] + 1 == next_step
                )
            )
        )
        if (
            any(length < 0 or length > 8 for length in lengths)
            or any(length != 8 for length in lengths[:-1])
            or (recorded_stop is not None and not valid_stop)
            or (recorded_stop is None and (len(lengths) != max_segments or any(length != 8 for length in lengths)))
        ):
            raise ValueError("Recorded stops and dispatched segment lengths must agree")
    # Validate the complete reference before any action or policy callback.
    remaining_h40(reference_poses, reference_openings, consumed=consumed)
    tool = validate_transform(np.asarray(tool, dtype=float)).copy()
    previous = np.asarray(previous_command, dtype=float).copy()
    if (
        tool.shape != (4, 4)
        or previous.shape != (7,)
        or not np.isfinite(previous).all()
        or np.max(np.abs(previous)) > 1 + 1e-12
        or type(previous_period) is not dict
        or not np.array_equal(previous, np.asarray(previous_period["command"], dtype=float))
    ):
        raise ValueError("The previous actual command, completed period and fixed TCP are required")
    state_reader = current_joint_site_state if state_reader is None else state_reader
    step = reference_start_step + consumed
    current = measured_snapshot(env, step)
    _terminal_input(terminal, current)
    if (
        receipt["control_steps_completed"] != step
        or previous_period["step"] != step - 1
        or previous_period["endpoint_receipt"] != terminal["endpoint_receipt"]
        or state_sha256(previous_period["actual_state"]) != current["state_sha256"]
    ):
        raise ValueError("The last completed period must bind the actual current endpoint")

    cache = SameStateEndpointCache(observers.endpoint)

    def render_cameras(actual_env, size):
        current_step = receipt["control_steps_completed"]
        before = measured_snapshot(actual_env, current_step)
        result = observers.render_cameras(actual_env, size)
        if measured_snapshot(actual_env, current_step) != before:
            raise ValueError("Camera rendering cannot advance physical state")
        return result

    wrapped = SimpleNamespace(endpoint=cache, current=observers.current, render_cameras=render_cameras)
    tail = dict(
        segments=[],
        stop=None,
        complete=False,
        automatic_retry=False,
        replay=replay,
        stop_recomputed=False,
        endpoint_cache_events=cache.events,
    )
    receipt["finite_reference_tail_partial"] = tail
    last_period = previous_period
    iterations = len(recorded_segments) if replay else max_segments
    for index in range(iterations):
        offset = consumed + 8 * index
        step = reference_start_step + offset
        before = measured_snapshot(env, step)
        observation = _terminal_input(terminal, before)
        q, limits, _jacobian = state_reader(env)
        q, limits = np.asarray(q, dtype=float), np.asarray(limits, dtype=float)
        if (
            q.shape != (7,)
            or limits.shape != (7, 2)
            or not np.isfinite(q).all()
            or not np.isfinite(limits).all()
            or np.any(limits[:, 0] >= limits[:, 1])
        ):
            raise ValueError("Finite current joints and limits are required")
        if last_period["inner_below_buffer_samples"] or np.min(np.minimum(q - limits[:, 0], limits[:, 1] - q)) < 0.05:
            if replay:
                raise ValueError("Recorded actions cannot pass a current joint-buffer stop")
            tail["stop"] = dict(reason="observed_joint_buffer_loss", at_step=step, next_action_dispatched=False)
            break
        poses, openings, metadata = remaining_h40(reference_poses, reference_openings, consumed=offset)
        targets = poses[:8].copy()
        anchor = observation["source_pose"] @ tool
        expected = None
        if replay:
            registered = recorded_segments[index]
            expected = registered["continuation"]["periods"]
            if (
                registered["start_step"] != step
                or registered["reference_offset"] != offset
                or registered["generated_endpoints"] != metadata["generated_endpoints"]
                or registered["terminal_hold_endpoints"] != metadata["terminal_hold_endpoints"]
                or type(expected) is not list
                or len(expected) > 8
                or (index < iterations - 1 and len(expected) != 8)
            ):
                raise ValueError("Recorded segments must contain only ordered dispatched periods")
        select, complete = open_segment(poses.copy(), openings.copy(), anchor.copy(), step, copy.deepcopy(expected))
        if not callable(select) or not callable(complete) or measured_snapshot(env, step) != before:
            raise ValueError("Opening a segment must preserve the actual physical state")
        row = dict(start_step=step, reference_offset=offset, **metadata, continuation=None)
        tail["segments"].append(row)

        def choose(q, limits, jacobian, prior, local_offset):
            snapshot = measured_snapshot(env, step + local_offset)
            past = receipt["continuation_partial"]["periods"]
            if past and past[-1]["inner_below_buffer_samples"]:
                if replay:
                    raise ValueError("Recorded actions cannot pass an observed joint-buffer stop")
                return {"stop": {"reason": "observed_joint_buffer_loss"}}
            decision = select(q, limits, jacobian, prior, local_offset)
            if measured_snapshot(env, step + local_offset) != snapshot:
                raise ValueError("Action selection cannot advance physical state")
            if replay and (
                type(decision) is not dict
                or decision.get("newly_selected", False) is not False
                or "stop" in decision
                or "command" not in decision
                or not np.array_equal(np.asarray(decision["command"]), np.asarray(expected[local_offset]["command"]))
            ):
                raise ValueError("Replay may dispatch only the recorded command without selecting a candidate")
            return decision

        def completed(actual_env, command, local_offset):
            snapshot = measured_snapshot(actual_env, step + local_offset + 1)
            value = complete(actual_env, command, local_offset)
            if measured_snapshot(actual_env, step + local_offset + 1) != snapshot:
                raise ValueError("Feedback cannot advance physical state")
            return value

        try:
            outcome = run_continuation(
                env,
                terminal,
                wrapped,
                receipt,
                start_step=step,
                previous_command=previous,
                targets=targets,
                tool=tool,
                target_object=target_object,
                choose_action=choose,
                stop_on_outside=True,
                expected_periods=expected,
                on_completed=completed,
                state_reader=state_reader,
                task_observer=task_observer,
                progress=progress,
            )
        finally:
            row["continuation"] = receipt.get("continuation_partial")
        if outcome["periods"]:
            last_period = outcome["periods"][-1]
            previous = np.asarray(last_period["command"], dtype=float).copy()
        if outcome["stop"] is not None:
            tail["stop"] = outcome["stop"]
            break
        if outcome["steps_completed"] != 8:
            if not replay or index != iterations - 1 or recorded_stop is None:
                raise ValueError("A partial segment requires a recorded producer stop")
            break
        sample, endpoint = cache(env, step + 8)
        terminal = {"sample": sample, "endpoint_receipt": endpoint}
    if replay:
        if tail["stop"] is not None and tail["stop"] != recorded_stop:
            raise ValueError("The independently observed physical stop differs")
        tail["stop"] = recorded_stop
        if len(tail["segments"]) < max_segments and recorded_stop is None:
            raise ValueError("An incomplete recorded tail requires a producer stop")
    tail["complete"] = tail["stop"] is None and len(tail["segments"]) == max_segments
    tail["generated_periods"] = sum(len(row["continuation"]["periods"]) for row in tail["segments"])
    return tail
