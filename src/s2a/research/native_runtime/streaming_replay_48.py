"""Independently replay bounded long streams while verifying actual anchors."""

import numpy as np

from s2a.geometry import validate_transform
from s2a.research.native_runtime.development_continuation import run_continuation
from s2a.research.native_runtime.streaming_replay import _check_terminal, _current


def replay_streaming_continuation(
    env,
    terminal,
    observers,
    receipt,
    *,
    producer_stream,
    target_chunks,
    source_observations=None,
    max_windows,
    previous_command,
    tool,
    target_object,
    state_reader=None,
    task_observer=None,
    progress=None,
):
    """Replay recorded completed actions with separately supplied target chunks.

    Callers must verify the immutable producer artifact and independently decode
    its generated references. An interrupted in-flight action is rejected; it
    cannot be reconstructed by pretending it was a complete 25-sample period.
    A producer's reported pre-dispatch stop reason is not re-evaluated here.
    """
    if (
        type(producer_stream) is not dict
        or type(producer_stream.get("segments")) is not list
        or type(producer_stream.get("complete")) is not bool
        or type(target_chunks) not in (list, tuple)
        or type(max_windows) is not int
        or not 1 <= max_windows <= 6
        or len(producer_stream["segments"]) > max_windows
        or len(target_chunks) != len(producer_stream["segments"])
    ):
        raise ValueError("A bounded recorded stream and one target chunk per segment are required")
    segments = producer_stream["segments"]
    if source_observations is not None and (
        type(source_observations) not in (list, tuple) or len(source_observations) != len(segments)
    ):
        raise ValueError("One source observation per replayed segment is required")
    previous = np.asarray(previous_command, dtype=float).copy()
    tool = validate_transform(np.asarray(tool, dtype=float)).copy()
    if previous.shape != (7,) or not np.isfinite(previous).all() or tool.shape != (4, 4):
        raise ValueError("A finite previous command and fixed TCP tool are required")
    start_step = terminal["endpoint_receipt"]["endpoint_step"]
    expected_step = start_step
    validated_targets = []
    for index, (row, targets) in enumerate(zip(segments, target_chunks, strict=True)):
        if (
            type(row) is not dict
            or "inflight" not in row
            or row["inflight"] is not None
            or type(row.get("start_step")) is not int
            or row["start_step"] != expected_step
        ):
            raise ValueError("Consecutive completed segments without in-flight actions are required")
        outcome = row.get("continuation")
        if type(outcome) is not dict or type(outcome.get("periods")) is not list:
            raise ValueError("A completed producer continuation is required")
        periods = outcome["periods"]
        if (
            len(periods) > 8
            or outcome.get("steps_completed") != len(periods)
            or (index < len(segments) - 1 and (len(periods) != 8 or outcome.get("stop") is not None))
            or any(
                type(period.get("step")) is not int or period["step"] != expected_step + i
                for i, period in enumerate(periods)
            )
        ):
            raise ValueError("Each saved period must be consecutive within its eight-period segment")
        target = validate_transform(np.asarray(targets, dtype=float)).copy()
        if target.shape != (8, 4, 4):
            raise ValueError("Eight independently decoded target endpoints are required")
        validated_targets.append(target)
        expected_step += len(periods)
    count = expected_step - start_step
    reported_stop = producer_stream.get("stop")
    if (
        producer_stream.get("control_steps_attempted") != count
        or producer_stream.get("control_steps_completed") != count
        or producer_stream.get("feedback_periods_committed") != count
        or (
            producer_stream["complete"]
            and (reported_stop is not None or len(segments) != max_windows or count != 8 * max_windows)
        )
        or (
            not producer_stream["complete"]
            and (
                type(reported_stop) is not dict
                or not reported_stop.get("reason")
                or type(reported_stop.get("at_step")) is not int
                or reported_stop["at_step"] != expected_step
                or reported_stop.get("next_action_dispatched") is not False
                or (
                    segments
                    and len(segments[-1]["continuation"]["periods"]) < 8
                    and segments[-1]["continuation"].get("stop") is None
                )
            )
        )
    ):
        raise ValueError("The producer must close every recorded dispatch or report a pre-dispatch stop")

    start_attempted, start_completed = receipt["control_steps_attempted"], receipt["control_steps_completed"]
    replay = {
        "segments": [],
        "complete": False,
        "reported_producer_stop": producer_stream.get("stop"),
        "stop_reason_independently_verified": False,
        "automatic_retry": False,
    }
    receipt["streaming_replay_partial"] = replay

    def totals():
        replay["control_steps_attempted"] = receipt["control_steps_attempted"] - start_attempted
        replay["control_steps_completed"] = receipt["control_steps_completed"] - start_completed

    try:
        _check_terminal(terminal, _current(env, start_step))
        totals()
        for index, (row, targets) in enumerate(zip(segments, validated_targets, strict=True)):
            step = row["start_step"]
            before = _current(env, step)
            _check_terminal(terminal, before)
            if source_observations is not None:
                expected_source = source_observations[index]
                if type(expected_source) is not dict or set(expected_source) != {"source_pose", "opening"}:
                    raise ValueError("One captured source pose and opening are required")
                actual = terminal["sample"]
                np.testing.assert_allclose(actual["source_pose"], expected_source["source_pose"], rtol=0, atol=1e-10)
                np.testing.assert_allclose(actual["opening"], expected_source["opening"], rtol=0, atol=1e-10)
            expected = row["continuation"]["periods"]

            def recorded_action(_q, _limits, _jacobian, _previous, offset):
                return {"command": np.asarray(expected[offset]["command"], dtype=float), "newly_selected": False}

            partial = {"start_step": step, "continuation": None}
            replay["segments"].append(partial)
            try:
                outcome = run_continuation(
                    env,
                    terminal,
                    observers,
                    receipt,
                    start_step=step,
                    previous_command=previous,
                    targets=targets,
                    tool=tool,
                    target_object=target_object,
                    choose_action=recorded_action,
                    stop_on_outside=True,
                    expected_periods=expected,
                    state_reader=state_reader,
                    task_observer=task_observer,
                    progress=progress,
                    continuation_limit=8,
                )
            finally:
                partial["continuation"] = receipt.get("continuation_partial")
            if outcome["steps_completed"] != len(expected):
                raise ValueError("The independent replay did not complete every recorded action")
            if expected:
                previous = np.asarray(expected[-1]["command"], dtype=float).copy()
            next_step = step + len(expected)
            before_observation = _current(env, next_step)
            sample, endpoint = observers.endpoint(env, next_step)
            terminal = {"sample": sample, "endpoint_receipt": endpoint}
            if _current(env, next_step) != before_observation:
                raise ValueError("Replay endpoint observation changed the physical state")
            _check_terminal(terminal, before_observation)
            totals()
        replay["complete"] = True
        return replay
    except BaseException as error:
        replay["error"] = {"type": type(error).__name__, "message": str(error)}
        totals()
        raise
    finally:
        if progress is not None:
            progress(receipt)
