"""Bind a completed feedback segment to one newly generated reference."""

import numpy as np

from s2a.geometry import inverse, validate_transform
from s2a.research.joint_control.preview.loaded_gripper_feedback import LoadedGripperFeedback
from s2a.research.joint_control.preview.quality_feedback_refiner import QualityFeedbackRefiner
from s2a.research.joint_control.preview.refined_feedback import RefinedPreviewFeedback
from s2a.research.joint_control.preview.tracking_preview_transaction import TrackingPreviewTransaction
from s2a.research.response_models.causal.segmented_inner_error import SegmentedInnerHistory


def advance_feedback(previous, owner, poses, openings, anchor_tcp, *, generated_step, receipt):
    """Continue one eight-period policy using only a new current-anchored H40.

    The caller remains responsible for binding the model request to the actual
    current camera image and for dispatching and independently checking actions.
    This function does not run a model, advance physics, or authorize a new case.
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
    next_step = previous.start_step + previous.max_steps
    if (
        previous.max_steps != 8
        or type(generated_step) is not int
        or generated_step != next_step
        or past["completed_periods"] != 8
        or past["last_completed_step"] != next_step - 1
        or previous.pending is not None
        or previous.last_inner_loss
        or inner._prediction is not None
        or arm.pending is not None
        or coupled.pending is not None
        or tracking.phase != "ready"
        or tracking.history.pending is not None
        or tracking.history.pending_plan is not None
        or coupled.completed_periods != arm.initial_periods + 8
        or tracking.history.completed_periods != coupled.completed_periods
        or gripper_state.get("staged") is not False
        or gripper_state.get("faulted") is not False
    ):
        raise ValueError("Eight fully completed and aligned safe periods required")

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
        raise ValueError("One finite current-anchored H40 reference required")
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
    if not np.allclose(anchor_tcp, site @ previous.site_from_tcp, atol=1e-9, rtol=0):
        raise ValueError("New reference anchor does not match the actual current TCP")
    twist = jacobian @ velocity
    coupled.current(site, twist)
    tracking.history.current(site, twist)

    view = owner.prepare_segment(next_step)
    next_refiner = QualityFeedbackRefiner(
        view,
        receipt,
        start_step=next_step,
        start_cursor=0,
        baseline=refiner.baseline,
        solver=refiner.solver,
        auditor=refiner.auditor,
        max_quality_response_evaluations=refiner.max_quality_response_evaluations,
    )
    site_from_tcp = previous.site_from_tcp
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
        next_step,
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
