"""Derive an explicitly bounded replay prefix from an actual reference tail."""

import copy


def capture_tail_prefix(tail, *, reference_start, capture_step):
    """Withhold recorded future actions and mark a planned observation boundary.

    The returned stop is an artificial replay boundary, not the producer's
    recorded quality or safety stop. The source tail is never mutated and no
    candidate actions are selected or dispatched by this function.
    """
    if (
        type(tail) is not dict
        or type(reference_start) is not int
        or reference_start < 0
        or type(capture_step) is not int
        or type(tail.get("segments")) is not list
        or not tail["segments"]
        or len(tail["segments"]) > 5
        or tail.get("automatic_retry") is not False
        or tail.get("replay") is not False
        or tail.get("stop_recomputed") is not False
    ):
        raise ValueError("A recorded H40 tail and explicit observation step are required")
    segments = tail["segments"]
    first_start = segments[0]["start_step"]
    first_offset = segments[0]["reference_offset"]
    if (
        type(first_start) is not int
        or type(first_offset) is not int
        or first_start != reference_start + first_offset
        or first_offset not in (0, 8, 16, 24, 32)
    ):
        raise ValueError("The first segment must preserve its generated-reference clock")
    total = 0
    for index, segment in enumerate(segments):
        offset = first_offset + 8 * index
        continuation = segment["continuation"]
        periods = continuation["periods"]
        if (
            offset >= 40
            or segment["start_step"] != reference_start + offset
            or segment["reference_offset"] != offset
            or segment["generated_endpoints"] != 40 - offset
            or segment["terminal_hold_endpoints"] != offset
            or type(periods) is not list
            or not 0 <= len(periods) <= 8
            or continuation["steps_completed"] != len(periods)
            or (index < len(segments) - 1 and (len(periods) != 8 or continuation["stop"] is not None))
            or [period["step"] for period in periods]
            != list(range(segment["start_step"], segment["start_step"] + len(periods)))
        ):
            raise ValueError("Recorded segments must contain consecutive completed actions")
        total += len(periods)
    original_end = first_start + total
    original_stop = tail.get("stop")
    if original_stop is not None:
        if (
            type(original_stop) is not dict
            or type(original_stop.get("reason")) is not str
            or not original_stop["reason"]
            or ("at_step" in original_stop) == ("after_step" in original_stop)
            or original_stop.get("next_action_dispatched") is not False
            or tail.get("complete") is not False
        ):
            raise ValueError("The original stop must follow its last completed action")
        value = original_stop.get("at_step", original_stop.get("after_step"))
        if type(value) is not int or value + ("after_step" in original_stop) != original_end:
            raise ValueError("The original stop must follow its last completed action")
    elif tail.get("complete") is not True or total != 40 - first_offset:
        raise ValueError("A tail without a stop must contain its full generated reference")
    if tail.get("generated_periods") != total or not first_start < capture_step < original_end:
        raise ValueError("The observation must be strictly inside the completed recorded tail")
    remaining = capture_step - first_start
    selected = []
    for segment in segments:
        if not remaining:
            break
        row = copy.deepcopy(segment)
        continuation = row["continuation"]
        count = min(remaining, len(continuation["periods"]))
        continuation["periods"] = continuation["periods"][:count]
        continuation["steps_completed"] = count
        continuation["stop"] = None
        selected.append(row)
        remaining -= count
    if remaining:
        raise ValueError("The recorded actions do not reach the observation boundary")
    result = copy.deepcopy(tail)
    result.update(
        segments=selected,
        stop=dict(reason="requested_observation", at_step=capture_step, next_action_dispatched=False),
        complete=False,
        automatic_retry=False,
        replay=False,
        stop_recomputed=False,
        endpoint_cache_events=[],
        generated_periods=capture_step - first_start,
    )
    return result
