"""Compare time-aligned generated references and preserve causal quality stops."""

import copy

import numpy as np

from s2a.geometry import pose_error, validate_transform
from s2a.research.native_runtime.adaptive_streaming_continuation import quality_trigger_view


def paired_errors(actual_tcp, old_targets, new_targets, offset):
    """Score one completed endpoint using the two references available at step 16."""
    actual = validate_transform(np.asarray(actual_tcp, dtype=float))
    old = validate_transform(np.asarray(old_targets, dtype=float))
    new = validate_transform(np.asarray(new_targets, dtype=float))
    if actual.shape != (4, 4) or old.shape != (40, 4, 4) or new.shape != (40, 4, 4):
        raise ValueError("A current TCP and two complete generated H40 references are required")
    if type(offset) is not int or not 0 <= offset < 32:
        raise ValueError("A completed offset within the 32-period comparison is required")
    return {"old": pose_error(old[8 + offset], actual), "new": pose_error(new[offset], actual)}


def quality_stop(previous_period, current_step, trigger):
    """Stop before dispatch if completed causal evidence requests a new reference."""
    if not callable(trigger):
        raise ValueError("A completed-period quality trigger is required")
    view = quality_trigger_view(previous_period)
    signal = trigger(view, current_step=current_step)
    if (
        type(signal) is not dict
        or type(signal.get("replan")) is not bool
        or type(signal.get("current_step")) is not int
        or signal["current_step"] != current_step
        or type(signal.get("reason")) is not str
        or not signal["reason"]
    ):
        raise ValueError("A valid current completed-period quality signal is required")
    if signal["replan"]:
        return {"stop": {"reason": "quality_refresh_required", "trigger": copy.deepcopy(signal)}}
    return None
