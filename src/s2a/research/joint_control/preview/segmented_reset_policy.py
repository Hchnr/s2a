"""Bind the existing reset policy to persistent, isolated reference segments."""

import copy

import numpy as np

from s2a.research.joint_control.preview.quality_feedback_refiner import QualityFeedbackRefiner
from s2a.research.joint_control.preview.reset_feedback_policy import build_reset_policy
from s2a.research.joint_control.preview.segmented_reset_history import segment_empty_reset_history
from s2a.research.joint_control.preview.streaming_feedback_boundary import advance_feedback


def build_segmented_reset_policy(env, inputs, runtime_receipt):
    """Convert a freshly built reset policy before any prediction or action."""
    if type(runtime_receipt) is not dict or any(
        key in runtime_receipt for key in ("reset_history", "segment_receipts", "segment_trace_starts")
    ):
        raise ValueError("A fresh runtime receipt is required")
    segment_receipt = {}
    policy, previous = build_reset_policy(env, inputs, segment_receipt)
    arm, inner = policy.arm_policy, policy.inner_history
    refiner = arm.tracking._refiner
    if (
        type(refiner) is not QualityFeedbackRefiner
        or refiner.history is not inner
        or refiner.receipt is not segment_receipt
        or refiner.start_step != 0
        or refiner.start_cursor != 0
        or policy.start_step != 0
        or policy.max_steps != 8
        or policy.pending is not None
        or arm.pending is not None
        or arm.coupled.completed_periods != 0
        or arm.tracking.history.completed_periods != 0
        or arm.tracking.history.last_plan is not None
        or not np.array_equal(previous, np.zeros(7))
    ):
        raise ValueError("The original reset policy must be fresh and internally aligned")
    owner, view = segment_empty_reset_history(inner)
    # The same physical history backs the view. Rebind both consumers before
    # the first staged prediction; every subsequent segment uses advance_feedback.
    refiner.history = policy.inner_history = view
    runtime_receipt["reset_history"] = copy.deepcopy(segment_receipt["reset_history"])
    runtime_receipt["segment_receipts"] = [segment_receipt]
    runtime_receipt["segment_trace_starts"] = [dict(shared=0, causal=0)]
    return policy, owner, previous


def advance_segmented_reset_policy(previous, owner, poses, openings, anchor, *, generated_step, runtime_receipt):
    """Advance only a fully completed segment and record its new counters."""
    if (
        type(runtime_receipt) is not dict
        or type(runtime_receipt.get("segment_receipts")) is not list
        or not runtime_receipt["segment_receipts"]
        or runtime_receipt["segment_receipts"][-1] is not previous.receipt
        or type(runtime_receipt.get("segment_trace_starts")) is not list
        or len(runtime_receipt["segment_trace_starts"]) != len(runtime_receipt["segment_receipts"])
    ):
        raise ValueError("The current policy must own the latest runtime segment receipt")
    first = runtime_receipt["segment_receipts"][0]
    traces = (first.get("shared_shift_traces"), first.get("causal_shift_fallback_traces"))
    if any(type(trace) is not list for trace in traces):
        raise ValueError("The original shared solver trace lists are required")
    starts = dict(shared=len(traces[0]), causal=len(traces[1]))
    next_receipt = {}
    policy = advance_feedback(
        previous, owner, poses, openings, anchor, generated_step=generated_step, receipt=next_receipt
    )
    runtime_receipt["segment_receipts"].append(next_receipt)
    # The original solver retains its trace lists across refiner instances.
    # Record boundaries instead of claiming that later traces belong to segment 0.
    runtime_receipt["segment_trace_starts"].append(starts)
    return policy
