"""The reset policy and its completed observations survive a reference change."""

from types import SimpleNamespace

import numpy as np
import pytest

from s2a.research.joint_control.preview import reset_feedback_policy
from s2a.research.joint_control.preview.segmented_reset_policy import (
    advance_segmented_reset_policy,
    build_segmented_reset_policy,
)
from tests.research.joint_control.preview.test_reset_feedback_policy import source


def make_policy(monkeypatch):
    case, terminal = source()
    inputs = reset_feedback_policy.policy_inputs(case, terminal)
    jac = np.c_[np.eye(6), np.zeros(6)]
    limits = np.tile([-1.0, 1.0], (7, 1))
    monkeypatch.setattr(reset_feedback_policy, "current_joint_site_state", lambda _env: (np.zeros(7), limits, jac))
    monkeypatch.setattr(
        reset_feedback_policy, "current_robot_dynamics", lambda _env, _jac: (np.eye(7), np.zeros(7), np.eye(4))
    )
    env = SimpleNamespace(robots=[SimpleNamespace(controller=SimpleNamespace())])
    receipt = {}
    policy, owner, previous = build_segmented_reset_policy(env, inputs, receipt)
    policy.arm_policy.joint_reader = lambda _env: (np.zeros(7), limits, jac)
    policy.arm_policy.dynamics_reader = lambda _env, _jac: (np.eye(7), np.zeros(7), np.eye(4))
    return policy, owner, previous, receipt, inputs


def test_reset_wraps_original_policy_before_any_selection(monkeypatch):
    policy, owner, previous, receipt, _inputs = make_policy(monkeypatch)
    refiner = policy.arm_policy.tracking._refiner
    assert refiner.history is policy.inner_history
    assert policy.inner_history._owner is owner
    assert owner.total_snapshot()["available"] is False
    assert receipt["reset_history"]["inner_error_history_available"] is False
    assert receipt["segment_receipts"] == [policy.receipt]
    np.testing.assert_array_equal(previous, np.zeros(7))
    with pytest.raises(ValueError, match="Eight fully completed"):
        advance_segmented_reset_policy(
            policy, owner, policy.poses, policy.openings, np.eye(4), generated_step=8, runtime_receipt=receipt
        )
    assert len(receipt["segment_receipts"]) == 1
    assert policy.inner_history.snapshot()["completed_periods"] == 0


def test_boundary_retains_completed_history_and_uses_a_new_segment_receipt(monkeypatch):
    policy, owner, _previous, receipt, inputs = make_policy(monkeypatch)
    arm, old_inner, old_gripper = policy.arm_policy, policy.inner_history, policy.gripper
    # Feed synthetic completed observations to exercise the real history boundary;
    # this is not a physical rollout or a gripper-transaction success claim.
    for step in range(8):
        old_inner.stage(step, np.zeros((25, 7)))
        old_inner.complete(step, np.full((25, 7), 0.001 * (step + 1)))
        arm.coupled.pending = (np.eye(4), np.zeros(12))
        arm.coupled.observe_completed(np.eye(4), np.zeros(6))
        arm.tracking.history.pending = (np.eye(4), np.zeros(12))
        arm.tracking.history.stage_plan(np.full((2, 6), step / 100))
        arm.tracking.history.observe_completed(np.eye(4), np.zeros(6))
        arm.previous = np.full(7, step / 100)
    plan, command, gripper_state = arm.tracking.history.last_plan.copy(), arm.previous.copy(), old_gripper.state()
    receipt["segment_receipts"][0]["shared_shift_traces"].append({"synthetic": True})
    receipt["segment_receipts"][0]["causal_shift_fallback_traces"].append({"synthetic": True})
    new = advance_segmented_reset_policy(
        policy, owner, inputs["poses"], inputs["openings"], inputs["anchor"], generated_step=8, runtime_receipt=receipt
    )
    assert new.gripper is old_gripper and new.gripper.state() == gripper_state
    assert new.arm_policy.coupled is arm.coupled and new.arm_policy.tracking is arm.tracking
    np.testing.assert_array_equal(new.arm_policy.previous, command)
    np.testing.assert_array_equal(new.arm_policy.tracking.history.last_plan, plan)
    assert new.inner_history.snapshot()["total_completed_periods"] == 8
    np.testing.assert_allclose(new.inner_history.snapshot()["inner_position_error_rad"], 0.008)
    assert len(receipt["segment_receipts"]) == 2 and receipt["segment_receipts"][-1] is new.receipt
    assert receipt["segment_trace_starts"] == [dict(shared=0, causal=0), dict(shared=1, causal=1)]
    with pytest.raises(ValueError, match="expired"):
        old_inner.snapshot()
    with pytest.raises(ValueError, match="latest"):
        advance_segmented_reset_policy(
            policy,
            owner,
            inputs["poses"],
            inputs["openings"],
            inputs["anchor"],
            generated_step=8,
            runtime_receipt=receipt,
        )
