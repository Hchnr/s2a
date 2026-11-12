"""Continue causal feedback after a newly observed, partially used segment."""

import numpy as np

from s2a.geometry import inverse, validate_transform
from s2a.research.joint_control.preview.loaded_gripper_feedback import LoadedGripperFeedback
from s2a.research.joint_control.preview.quality_feedback_refiner import QualityFeedbackRefiner
from s2a.research.joint_control.preview.refined_feedback import RefinedPreviewFeedback
from s2a.research.joint_control.preview.tracking_preview_transaction import TrackingPreviewTransaction
from s2a.research.response_models.causal.segmented_inner_error import SegmentedInnerHistory
from s2a.research.trajectory_search.margin_two_step_search import (
    POSITION_CAP_M,
    POSITION_MARGIN_M,
    ROTATION_CAP_RAD,
    ROTATION_MARGIN_RAD,
    SELECTION_TOLERANCE,
)


def completed_quality_margin(period, *, current_step):
    """Inspect the already audited next-point prediction of a completed action.

    This signal never consumes task success, object state, future observations,
    or a prediction from a command that was not completed. A first-only audit
    has no independently checked second point and requests a new reference.
    """
    if type(current_step) is not int or current_step < 1 or period.get("step") != current_step - 1:
        raise ValueError("The immediately preceding completed action is required")
    history = period["completed_feedback"]["policy_history"]["past_inner_error"]
    if (
        period["endpoint_receipt"].get("endpoint_step") != current_step
        or len(period["actual_inner"]) != 25
        or history.get("last_completed_step") != current_step - 1
        or history.get("future_guarantee") is not False
    ):
        raise ValueError("Completed endpoint, inner samples and causal history are required")
    verification = period["decision"]["quality"]["verification"]
    if verification.get("passed") is not True or verification.get("first_pose_caps_satisfied") is not True:
        raise ValueError("The completed action must retain its independent quality audit")
    if verification.get("hard_pose_prefix") != 2:
        return dict(replan=True, reason="no_audited_second_pose", current_step=current_step)
    if (
        verification.get("prefix_pose_caps_satisfied") is not True
        or verification.get("prefix_pose_caps_enforced") is not True
    ):
        raise ValueError("The two-point audit must have enforced and passed both pose caps")
    errors = np.asarray(verification["tcp_errors"], dtype=float)
    if errors.shape != (8, 6) or not np.isfinite(errors).all():
        raise ValueError("Eight finite audited model errors are required")
    position = float(np.linalg.norm(errors[1, :3]))
    rotation = float(np.linalg.norm(errors[1, 3:]))
    position_limit = POSITION_CAP_M - POSITION_MARGIN_M
    rotation_limit = ROTATION_CAP_RAD - ROTATION_MARGIN_RAD
    replan = bool(position >= position_limit - SELECTION_TOLERANCE or rotation >= rotation_limit - SELECTION_TOLERANCE)
    return dict(
        replan=replan,
        reason="quality_margin_consumed" if replan else "quality_margin_available",
        current_step=current_step,
        predicted_next_position_error_m=position,
        predicted_next_rotation_error_rad=rotation,
        position_margin_limit_m=position_limit,
        rotation_margin_limit_rad=rotation_limit,
    )


def advance_after_completed(previous, owner, poses, openings, anchor_tcp, *, generated_step, receipt):
    """Bind a current-image H40 after 1–8 completed periods of a segment.

    The caller must bind the reference to the actual current observation and
    independently own physical dispatch. This operation preserves the existing
    arm, gripper and inner-error histories without staging or dispatching an
    action. An incomplete or pending action cannot be crossed.
    """
    if type(previous) is not LoadedGripperFeedback or type(owner) is not SegmentedInnerHistory:
        raise ValueError("Registered feedback and its segmented history are required")
    if type(receipt) is not dict or receipt or receipt is previous.receipt:
        raise ValueError("One fresh segment receipt is required")
    arm, inner, gripper = previous.arm_policy, previous.inner_history, previous.gripper
    if type(arm) is not RefinedPreviewFeedback or type(arm.tracking) is not TrackingPreviewTransaction:
        raise ValueError("Original preview feedback transaction required")
    tracking, coupled = arm.tracking, arm.coupled
    refiner = tracking._refiner
    if (
        type(refiner) is not QualityFeedbackRefiner
        or refiner.history is not inner
        or refiner.receipt is not previous.receipt
        or refiner.start_step != previous.start_step
        or refiner.start_cursor != previous.cursor
        or arm.cursor != previous.cursor
        or getattr(inner, "_owner", None) is not owner
    ):
        raise ValueError("Original quality, history, reference and receipt binding required")
    past, gripper_state = inner.snapshot(), gripper.state()
    count = past["completed_periods"]
    if (
        previous.max_steps != 8
        or type(count) is not int
        or not 1 <= count <= previous.max_steps
        or type(generated_step) is not int
        or generated_step != previous.start_step + count
        or past["last_completed_step"] != generated_step - 1
        or previous.pending is not None
        or previous.last_inner_loss
        or inner._prediction is not None
        or arm.pending is not None
        or coupled.pending is not None
        or tracking.phase != "ready"
        or tracking.history.pending is not None
        or tracking.history.pending_plan is not None
        or coupled.completed_periods != arm.initial_periods + count
        or tracking.history.completed_periods != coupled.completed_periods
        or gripper_state.get("staged") is not False
        or gripper_state.get("faulted") is not False
    ):
        raise ValueError("A nonempty, fully completed and aligned safe segment prefix is required")
    poses = validate_transform(np.asarray(poses, dtype=float))
    anchor_tcp = validate_transform(np.asarray(anchor_tcp, dtype=float))
    openings = np.asarray(openings, dtype=float)
    if (
        poses.shape != (40, 4, 4)
        or anchor_tcp.shape != (4, 4)
        or openings.shape != (40,)
        or not np.isfinite(openings).all()
        or np.any((openings < 0) | (openings > 1))
    ):
        raise ValueError("One finite current-anchored H40 reference is required")
    q, limits, jacobian = arm.joint_reader(previous.env)
    q, limits, jacobian = (np.asarray(value, dtype=float) for value in (q, limits, jacobian))
    if (
        q.shape != (7,)
        or limits.shape != (7, 2)
        or jacobian.shape != (6, 7)
        or not np.isfinite(q).all()
        or not np.isfinite(limits).all()
        or not np.isfinite(jacobian).all()
        or np.any(limits[:, 0] >= limits[:, 1])
        or np.min(np.minimum(q - limits[:, 0], limits[:, 1] - q)) < 0.05
    ):
        raise ValueError("Current measured joints must preserve the buffer")
    _mass, velocity, site = arm.dynamics_reader(previous.env, jacobian)
    velocity, site = np.asarray(velocity, dtype=float), validate_transform(np.asarray(site, dtype=float))
    if velocity.shape != (7,) or not np.isfinite(velocity).all() or site.shape != (4, 4):
        raise ValueError("Finite actual endpoint velocity and site pose required")
    site_from_tcp = previous.site_from_tcp
    if not np.allclose(anchor_tcp, site @ site_from_tcp, atol=1e-9, rtol=0):
        raise ValueError("New reference anchor does not match the actual current TCP")
    twist = jacobian @ velocity
    coupled.current(site, twist)
    tracking.history.current(site, twist)

    view = owner.prepare_segment(generated_step)
    next_refiner = QualityFeedbackRefiner(
        view,
        receipt,
        start_step=generated_step,
        start_cursor=0,
        baseline=refiner.baseline,
        solver=refiner.solver,
        auditor=refiner.auditor,
        max_quality_response_evaluations=refiner.max_quality_response_evaluations,
    )
    next_arm = RefinedPreviewFeedback(
        previous.env,
        coupled,
        tracking,
        poses @ inverse(site_from_tcp),
        anchor_tcp @ inverse(site_from_tcp),
        site_from_tcp,
        0,
        arm.previous,
        arm.contract,
        receipt,
        preview_solver=arm.preview_solver,
        dynamics_reader=arm.dynamics_reader,
        joint_reader=arm.joint_reader,
    )
    policy = LoadedGripperFeedback(
        previous.env,
        next_arm,
        gripper,
        view,
        poses,
        openings,
        0,
        site_from_tcp,
        generated_step,
        receipt,
        dynamics_reader=previous.dynamics_reader,
        gripper_reader=previous.gripper_reader,
        initial_inner_loss=False,
        max_steps=8,
    )
    try:
        tracking._refiner = next_refiner
        owner.activate_segment(view)
    except BaseException:
        tracking._refiner = refiner
        raise
    return policy


def advance_recorded_segment(previous, owner, poses, openings, anchor, *, generated_step, runtime_receipt):
    """Record a partial-segment boundary while preserving the shared trace owner."""
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
    policy = advance_after_completed(
        previous, owner, poses, openings, anchor, generated_step=generated_step, receipt=next_receipt
    )
    runtime_receipt["segment_receipts"].append(next_receipt)
    runtime_receipt["segment_trace_starts"].append(starts)
    return policy
