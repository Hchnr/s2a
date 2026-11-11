"""Validate actual per-case physical counts and report measured outcomes."""

import numpy as np

from scripts.simulation.runtime import replay_multicase_development_prefix as prefix


def _stream_periods(stream):
    segments = stream["segments"]
    prefix.require(
        len(segments) <= 5
        and stream["requests_issued"] == stream["responses_accepted"] == len(segments)
        and stream["automatic_retry"] is False,
        "At most five completed live reference exchanges are allowed",
    )
    periods = []
    for index, segment in enumerate(segments):
        continuation = segment["continuation"]
        current = continuation["periods"]
        prefix.require(
            segment["start_step"] == 8 + index * 8
            and segment["inflight"] is None
            and continuation["steps_completed"] == len(current)
            and 0 <= len(current) <= 8
            and (index == len(segments) - 1 or len(current) == 8),
            "Each next reference requires the previous eight completed periods",
        )
        periods.extend(current)
    prefix.require(
        stream["control_steps_attempted"]
        == stream["control_steps_completed"]
        == stream["feedback_periods_committed"]
        == len(periods)
        and (
            (stream["complete"] is True and stream["stop"] is None and len(periods) == 40)
            or (
                stream["complete"] is False
                and type(stream["stop"]) is dict
                and stream["stop"].get("next_action_dispatched") is False
            )
        ),
        "A physical stream must complete or preserve an explicit pre-dispatch stop",
    )
    if stream["stop"] is not None:
        stop, next_step = stream["stop"], 8 + len(periods)
        prefix.require(
            type(stop.get("reason")) is str
            and bool(stop["reason"])
            and (
                ("at_step" in stop and "after_step" not in stop and stop["at_step"] == next_step)
                or ("after_step" in stop and "at_step" not in stop and stop["after_step"] + 1 == next_step)
            ),
            "The stop must immediately follow the actually recorded actions",
        )
    return periods


def summarize_branch(branch, phase, *, producer=None):
    """Count producer work or independent replay without qualifying task success."""
    runtime, continuation = branch["runtime"], branch["runtime"]["continuation"]
    first = continuation["first_window"]["periods"]
    prefix.require(
        phase in ("producer", "checker")
        and branch["complete"] is branch["passed"] is runtime["complete"] is runtime["passed"] is True
        and runtime["cleanup_complete"] is runtime["reset_observation_verified"] is runtime["reset_image_match"] is True
        and runtime["previous_saved_command"] is None
        and len(first) == 8
        and [period["step"] for period in first] == list(range(8)),
        "A verified reset and the original eight physical periods are required",
    )
    budget = None
    if phase == "producer":
        stream = continuation["stream"]
        periods = _stream_periods(stream)
        segments, starts = runtime["segment_receipts"], runtime["segment_trace_starts"]
        prefix.require(
            runtime["newly_selected_commands"] == 8 + len(periods)
            and len(segments) == len(starts) == 1 + len(stream["segments"])
            and starts[0] == dict(shared=0, causal=0)
            and runtime["model_stop_sent"] is (stream["requests_issued"] < 5),
            "Each actual producer segment must retain its own receipt and history boundary",
        )
        shared = segments[0]["shared_shift_traces"]
        causal = segments[0]["causal_shift_fallback_traces"]
        for key, traces in (("shared", shared), ("causal", causal)):
            values = [start[key] for start in starts]
            prefix.require(
                all(type(value) is int and 0 <= value <= len(traces) for value in values) and values == sorted(values),
                "Shared solver history boundaries must be monotonic and inside the actual trace",
            )
        budget = dict(
            quality_responses=sum(item["quality_refiner"]["quality_response_evaluations"] for item in segments),
            baseline_responses=sum(item["quality_refiner"]["baseline_response_evaluations"] for item in segments),
            gripper_sequences=sum(item["gripper_proposals"]["candidate_sequences"] for item in segments),
            endpoint_queries=sum(item["quality_refiner"]["endpoint_queries_charged"] for item in segments),
            transition_steps=sum(item["quality_refiner"]["transition_steps_charged"] for item in segments),
        )
        prefix.require(
            all(type(value) is int and value >= 0 for value in budget.values())
            and budget["quality_responses"] <= 48 * 320
            and budget["baseline_responses"] <= 48 * 480
            and budget["gripper_sequences"] <= 48 * 6561
            and budget["endpoint_queries"] <= 9600
            and budget["transition_steps"] <= 124800,
            "The candidate computation exceeds its registered per-case budget",
        )
        model_requests, stop = stream["requests_issued"], stream["stop"]
    else:
        prefix.require(
            producer is not None
            and branch["candidate_modules_unloaded"] is True
            and runtime["newly_selected_commands"] == 0
            and continuation["stop_reason_independently_verified"] is False,
            "An isolated checker must replay only actual commands without recomputing stop decisions",
        )
        producer_stream = producer["runtime"]["continuation"]["stream"]
        expected = _stream_periods(producer_stream)
        replays = continuation["replays"]
        periods = [period for replay in replays for period in replay["periods"]]
        prefix.require(
            len(replays) == len(producer_stream["segments"])
            and len(periods) == len(expected) == continuation["actual_stream_periods"]
            and continuation["actual_model_requests"] == len(replays)
            and continuation["reported_producer_stop"] == producer_stream["stop"],
            "The checker must replay exactly the producer's actual reference boundaries and actions",
        )
        model_requests, stop = 0, continuation["reported_producer_stop"]
    count = 8 + len(periods)
    all_periods = first + periods
    counters = runtime["counters"]
    prefix.require(
        count <= 48
        and runtime["control_steps_attempted"] == runtime["control_steps_completed"] == count
        and counters["mj_step"]["attempted"] == counters["mj_step"]["completed"] == count * 25
        and counters["model_construct"]["completed"] == counters["context"]["completed"] == 3
        and [period["step"] for period in all_periods] == list(range(count))
        and all(len(period["actual_inner"]) == 25 for period in all_periods),
        "Recorded physical steps, models, contexts and inner samples must match the bounded run",
    )
    error = np.asarray([period["tcp_error"] for period in all_periods], dtype=float)
    prefix.require(error.shape == (count, 6) and np.isfinite(error).all(), "Measured finite TCP errors are required")
    return dict(
        control_steps=count,
        native_steps=count * 25,
        models=3,
        contexts=3,
        live_control_steps=len(periods),
        model_requests=model_requests,
        stop=stop,
        candidate_budget=budget,
        pose_caps_satisfied=sum(bool(period["pose_caps_satisfied"]) for period in all_periods),
        position_rmse_m=float(np.sqrt(np.mean(np.sum(error[:, :3] ** 2, axis=1)))),
        rotation_rmse_rad=float(np.sqrt(np.mean(np.sum(error[:, 3:] ** 2, axis=1)))),
        minimum_inner_margin_rad=float(min(period["minimum_inner_margin_rad"] for period in all_periods)),
        target_grasp_periods=sum(bool(period["target_grasp"]) for period in all_periods),
        task_success_periods=sum(bool(period["task_success"]) for period in all_periods),
    )
