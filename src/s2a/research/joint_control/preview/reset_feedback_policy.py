"""Build one causal feedback policy from the actual reset and generated reference."""

import copy
from dataclasses import asdict

import numpy as np

from s2a.geometry import inverse, validate_transform
from s2a.research.gripper.loaded_gripper_transaction import LoadedGripperTransaction
from s2a.research.joint_control.preview.loaded_gripper_feedback import LoadedGripperFeedback
from s2a.research.joint_control.preview.quality_feedback_refiner import QualityFeedbackRefiner
from s2a.research.joint_control.preview.refined_feedback import RefinedPreviewFeedback
from s2a.research.joint_control.preview.reset_history import make_reset_preview_history
from s2a.research.joint_control.preview.shared_shift_fallback import SharedShiftFallbackAudit
from s2a.research.joint_control.tracking.loaded_velocity_control import LoadedVelocityControl
from s2a.research.native_runtime.development_continuation import current_joint_site_state, current_robot_dynamics


def policy_inputs(case, terminal):
    """Copy only current observation, generated references and fixed controller data."""
    sample, chunk, controller = terminal["sample"], case["chunk"], case["controller"]
    poses = validate_transform(np.asarray(chunk["effective_targets_tcp"], dtype=float)).copy()
    anchor = validate_transform(np.asarray(chunk["anchor_tcp"], dtype=float)).copy()
    openings = np.asarray(chunk["openings"], dtype=float).copy()
    tool = validate_transform(np.asarray(case["tool"], dtype=float)).copy()
    source_from_site = validate_transform(np.asarray(controller["source_from_site"], dtype=float)).copy()
    opening = float(sample["opening"])
    if (
        poses.shape != (40, 4, 4)
        or anchor.shape != (4, 4)
        or tool.shape != (4, 4)
        or source_from_site.shape != (4, 4)
        or openings.shape != (40,)
        or not np.isfinite(openings).all()
        or np.any((openings < 0) | (openings > 1))
        or not np.isfinite(opening)
        or not 0 <= opening <= 1
    ):
        raise ValueError("A current measured opening and complete generated reference are required")
    return dict(
        poses=poses,
        anchor=anchor,
        openings=openings,
        tool=tool,
        source_from_site=source_from_site,
        contract=copy.deepcopy(controller["contract"]),
        gripper_config=copy.deepcopy(controller["gripper_config"]),
        velocity_envelope=controller["gripper_velocity_envelope"],
        gripper_parameters=copy.deepcopy(case["gripper_parameters"]),
        initial_opening=opening,
    )


def build_reset_policy(env, inputs, receipt):
    """Create independent arm/gripper histories without access to past or future truth."""
    _q, _limits, jacobian = current_joint_site_state(env)
    _mass, velocity, site = current_robot_dynamics(env, jacobian)
    twist = jacobian @ velocity
    site_from_tcp = inverse(inputs["source_from_site"]) @ inputs["tool"]
    np.testing.assert_allclose(site @ site_from_tcp, inputs["anchor"], atol=1e-9, rtol=0)
    adapter = SharedShiftFallbackAudit(receipt)

    def refiner_factory(history):
        return QualityFeedbackRefiner(
            history,
            receipt,
            start_step=0,
            start_cursor=0,
            solver=adapter.solve,
            auditor=adapter.audit,
            max_quality_response_evaluations=320,
        )

    fresh = make_reset_preview_history(site, twist, refiner_factory=refiner_factory)
    adapter.attach_history(fresh.tracking.history)
    arm = RefinedPreviewFeedback(
        env,
        fresh.coupled,
        fresh.tracking,
        inputs["poses"] @ inverse(site_from_tcp),
        inputs["anchor"] @ inverse(site_from_tcp),
        site_from_tcp,
        0,
        fresh.previous_command,
        inputs["contract"],
        receipt,
    )
    gripper = LoadedVelocityControl(
        inputs["gripper_parameters"], inputs["initial_opening"], velocity_envelope=inputs["velocity_envelope"]
    )
    if asdict(gripper.latch.config) != inputs["gripper_config"]:
        raise ValueError("The original gripper latch configuration must match")
    policy = LoadedGripperFeedback(
        env,
        arm,
        LoadedGripperTransaction(gripper),
        fresh.inner_error,
        inputs["poses"],
        inputs["openings"],
        0,
        site_from_tcp,
        0,
        receipt,
        initial_inner_loss=False,
        max_steps=8,
    )
    receipt["reset_history"] = {
        "completed_periods": 0,
        "previous_command": fresh.previous_command,
        "inner_error_history_available": False,
        "last_plan_available": False,
        "initial_inner_safety_observed": False,
    }
    return policy, fresh.previous_command
