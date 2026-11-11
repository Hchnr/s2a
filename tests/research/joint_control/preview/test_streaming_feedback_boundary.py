import numpy as np
import pytest

from s2a.research.joint_control.preview.loaded_gripper_feedback import LoadedGripperFeedback
from s2a.research.joint_control.preview.quality_feedback_refiner import QualityFeedbackRefiner
from s2a.research.joint_control.preview.refined_feedback import RefinedPreviewFeedback
from s2a.research.joint_control.preview.streaming_feedback_boundary import advance_feedback
from s2a.research.joint_control.preview.tracking_preview_transaction import TrackingPreviewTransaction
from s2a.research.response_models.causal.coupled_response_history import CoupledResponseHistory
from s2a.research.response_models.causal.segmented_inner_error import SegmentedInnerHistory


class Gripper:
    def __init__(self):
        self.staged = self.faulted = False
        self.latch = 0

    def state(self):
        return {"staged": self.staged, "faulted": self.faulted, "latch": self.latch}


def fixture():
    env, receipt, owner = object(), {}, SegmentedInnerHistory()
    inner = owner.begin_segment(100)
    quality = QualityFeedbackRefiner(inner, receipt, start_step=100, start_cursor=0)
    tracking = TrackingPreviewTransaction(refiner=quality)
    coupled = CoupledResponseHistory()
    coupled.completed_periods = tracking.history.completed_periods = 100
    contract = {"input_min": [-1] * 6, "input_max": [1] * 6, "output_min": [-0.05] * 6, "output_max": [0.05] * 6}

    def joint(_env):
        return np.zeros(7), np.tile([-1.0, 1.0], (7, 1)), np.zeros((6, 7))

    def dynamics(_env, _jac):
        return np.eye(7), np.zeros(7), np.eye(4)

    poses = np.tile(np.eye(4), (40, 1, 1))
    arm = RefinedPreviewFeedback(
        env,
        coupled,
        tracking,
        poses,
        np.eye(4),
        np.eye(4),
        0,
        np.zeros(7),
        contract,
        receipt,
        dynamics_reader=dynamics,
        joint_reader=joint,
    )
    gripper = Gripper()
    policy = LoadedGripperFeedback(env, arm, gripper, inner, poses, np.ones(40), 0, np.eye(4), 100, receipt)
    return policy, owner, poses


def complete_eight(policy, error):
    """Supply synthetic completed observations to the real history objects."""
    arm, inner = policy.arm_policy, policy.inner_history
    for offset in range(8):
        step = policy.start_step + offset
        inner.stage(step, np.zeros((25, 7)))
        inner.complete(step, np.full((25, 7), error))
        arm.coupled.pending = (np.eye(4), np.zeros(12))
        arm.coupled.observe_completed(np.eye(4), np.zeros(6))
        history = arm.tracking.history
        history.pending = (np.eye(4), np.zeros(12))
        history.stage_plan(np.full((2, 6), offset / 100))
        history.observe_completed(np.eye(4), np.zeros(6))
        arm.previous = np.full(7, offset / 100)
        policy.gripper.latch += 1


def test_two_boundaries_retain_completed_control_histories():
    first, owner, poses = fixture()
    complete_eight(first, 0.2)
    coupled, tracking, grip = first.arm_policy.coupled, first.arm_policy.tracking, first.gripper
    plan = tracking.history.last_plan.copy()
    command = first.arm_policy.previous.copy()
    second = advance_feedback(first, owner, poses, np.full(40, 0.7), np.eye(4), generated_step=108, receipt={})
    assert second.arm_policy.coupled is coupled and second.arm_policy.tracking is tracking and second.gripper is grip
    assert grip.latch == 8 and second.start_step == 108
    np.testing.assert_array_equal(second.arm_policy.previous, command)
    np.testing.assert_array_equal(tracking.history.last_plan, plan)
    assert second.inner_history.snapshot()["completed_periods"] == 0
    assert second.inner_history.snapshot()["total_completed_periods"] == 8
    assert second.inner_history.snapshot()["inner_position_error_rad"][24, 0] == pytest.approx(0.2)
    with pytest.raises(ValueError, match="expired"):
        first.inner_history.snapshot()
    complete_eight(second, 0.3)
    third = advance_feedback(second, owner, poses, np.ones(40), np.eye(4), generated_step=116, receipt={})
    assert third.gripper.latch == 16 and third.inner_history.snapshot()["total_completed_periods"] == 16
    assert third.inner_history.snapshot()["inner_position_error_rad"][24, 0] == pytest.approx(0.3)


def test_six_windows_retain_history_across_five_reference_boundaries():
    policy, owner, poses = fixture()
    coupled, tracking, gripper = policy.arm_policy.coupled, policy.arm_policy.tracking, policy.gripper
    for index in range(6):
        complete_eight(policy, 0.1 + index / 100)
        assert owner.total_snapshot()["completed_periods"] == 8 * (index + 1)
        assert gripper.latch == 8 * (index + 1)
        if index < 5:
            previous = policy
            policy = advance_feedback(
                policy, owner, poses, np.ones(40), np.eye(4), generated_step=108 + 8 * index, receipt={}
            )
            assert policy.arm_policy.coupled is coupled and policy.arm_policy.tracking is tracking
            assert policy.gripper is gripper and policy.inner_history.snapshot()["completed_periods"] == 0
            with pytest.raises(ValueError, match="expired"):
                previous.inner_history.snapshot()


@pytest.mark.parametrize(
    "corruption", ["time", "anchor", "gripper", "pending", "loss", "count", "opening", "inner_pending", "buffer"]
)
def test_rejects_invalid_boundary_without_replacing_old_history(corruption):
    policy, owner, poses = fixture()
    complete_eight(policy, 0.1)
    arm, inner, old_refiner = policy.arm_policy, policy.inner_history, policy.arm_policy.tracking._refiner
    anchor, opening, step = np.eye(4), np.ones(40), 108
    if corruption == "time":
        step += 1
    elif corruption == "anchor":
        anchor[0, 3] = 0.1
    elif corruption == "gripper":
        policy.gripper.staged = True
    elif corruption == "pending":
        arm.tracking.history.pending_plan = np.zeros((1, 6))
    elif corruption == "loss":
        policy.last_inner_loss = True
    elif corruption == "count":
        arm.coupled.completed_periods -= 1
    elif corruption == "inner_pending":
        inner.stage(108, np.zeros((25, 7)))
    elif corruption == "buffer":

        def near_limit(_env):
            return np.full(7, 0.96), np.tile([-1.0, 1.0], (7, 1)), np.zeros((6, 7))

        arm.joint_reader = near_limit
    else:
        opening[0] = 2.0
    with pytest.raises(ValueError):
        advance_feedback(policy, owner, poses, opening, anchor, generated_step=step, receipt={})
    assert arm.tracking._refiner is old_refiner
    assert inner.snapshot()["completed_periods"] == 8
    assert owner.total_snapshot()["completed_periods"] == 8


def test_activation_failure_restores_previous_refiner_and_history(monkeypatch):
    policy, owner, poses = fixture()
    complete_eight(policy, 0.1)
    tracking, inner = policy.arm_policy.tracking, policy.inner_history
    old_refiner = tracking._refiner

    def fail_activation(_view):
        raise ValueError("simulated concurrent history change")

    monkeypatch.setattr(owner, "activate_segment", fail_activation)
    with pytest.raises(ValueError, match="concurrent history change"):
        advance_feedback(policy, owner, poses, np.ones(40), np.eye(4), generated_step=108, receipt={})
    assert tracking._refiner is old_refiner
    assert inner.snapshot()["completed_periods"] == 8
