import copy

import numpy as np
import pytest

from s2a.research.joint_control.preview.causal_reference_replan import (
    advance_after_completed,
    advance_recorded_segment,
    completed_quality_margin,
)
from tests.research.joint_control.preview.test_streaming_feedback_boundary import complete_eight, fixture


def complete_prefix(policy, count, error=0.1):
    arm, inner = policy.arm_policy, policy.inner_history
    for offset in range(count):
        step = policy.start_step + offset
        inner.stage(step, np.zeros((25, 7)))
        inner.complete(step, np.full((25, 7), error))
        arm.coupled.pending = (np.eye(4), np.zeros(12))
        arm.coupled.observe_completed(np.eye(4), np.zeros(6))
        history = arm.tracking.history
        history.pending = (np.eye(4), np.zeros(12))
        history.stage_plan(np.full((8, 6), offset / 100))
        history.observe_completed(np.eye(4), np.zeros(6))
        arm.previous = np.full(7, offset / 100)
        policy.gripper.latch += 1


def completed_period(position=0.001, rotation=0.01):
    errors = np.zeros((8, 6))
    errors[1, 0], errors[1, 3] = position, rotation
    return {
        "step": 106,
        "endpoint_receipt": {"endpoint_step": 107},
        "actual_inner": [{}] * 25,
        "completed_feedback": {
            "policy_history": {"past_inner_error": {"last_completed_step": 106, "future_guarantee": False}}
        },
        "decision": {
            "quality": {
                "verification": {
                    "passed": True,
                    "first_pose_caps_satisfied": True,
                    "hard_pose_prefix": 2,
                    "prefix_pose_caps_satisfied": True,
                    "prefix_pose_caps_enforced": True,
                    "tcp_errors": errors,
                }
            }
        },
    }


def test_margin_trigger_uses_only_an_audited_completed_prediction():
    assert completed_quality_margin(completed_period(), current_step=107)["replan"] is False
    position = completed_quality_margin(completed_period(position=0.0095), current_step=107)
    rotation = completed_quality_margin(completed_period(rotation=0.047), current_step=107)
    assert position["replan"] and position["reason"] == "quality_margin_consumed"
    assert rotation["replan"] and rotation["reason"] == "quality_margin_consumed"
    first_only = completed_period()
    first_only["decision"]["quality"]["verification"]["hard_pose_prefix"] = 1
    assert completed_quality_margin(first_only, current_step=107)["reason"] == "no_audited_second_pose"
    incomplete = completed_period()
    incomplete["actual_inner"] = incomplete["actual_inner"][:-1]
    with pytest.raises(ValueError, match="Completed endpoint"):
        completed_quality_margin(incomplete, current_step=107)
    unaudited = completed_period()
    unaudited["decision"]["quality"]["verification"]["prefix_pose_caps_enforced"] = False
    with pytest.raises(ValueError, match="two-point audit"):
        completed_quality_margin(unaudited, current_step=107)


def test_midsegment_replan_preserves_completed_history_and_current_anchor():
    previous, owner, poses = fixture()
    complete_prefix(previous, 3, error=0.2)
    arm, inner, gripper = previous.arm_policy, previous.inner_history, previous.gripper
    command, plan = arm.previous.copy(), arm.tracking.history.last_plan.copy()
    receipt = {}
    poses = poses.copy()
    poses[:, 0, 3] = np.arange(40) * 0.001
    policy = advance_after_completed(
        previous, owner, poses, np.full(40, 0.7), np.eye(4), generated_step=103, receipt=receipt
    )
    assert policy.start_step == 103 and policy.cursor == 0 and policy.max_steps == 8
    assert policy.arm_policy.coupled is arm.coupled and policy.arm_policy.tracking is arm.tracking
    assert policy.gripper is gripper and gripper.latch == 3
    np.testing.assert_array_equal(policy.arm_policy.previous, command)
    np.testing.assert_array_equal(policy.arm_policy.tracking.history.last_plan, plan)
    np.testing.assert_array_equal(policy.poses, poses)
    assert policy.inner_history.snapshot()["completed_periods"] == 0
    assert policy.inner_history.snapshot()["total_completed_periods"] == 3
    assert policy.inner_history.snapshot()["inner_position_error_rad"][24, 0] == pytest.approx(0.2)
    with pytest.raises(ValueError, match="expired"):
        inner.snapshot()
    complete_eight(policy, 0.3)
    assert owner.total_snapshot()["completed_periods"] == 11 and gripper.latch == 11


@pytest.mark.parametrize("count", [1, 7, 8])
def test_replan_accepts_each_end_of_the_completed_prefix_range(count):
    previous, owner, poses = fixture()
    complete_prefix(previous, count)
    next_policy = advance_after_completed(
        previous, owner, poses, np.ones(40), np.eye(4), generated_step=100 + count, receipt={}
    )
    assert next_policy.start_step == 100 + count
    assert next_policy.inner_history.snapshot()["total_completed_periods"] == count


def test_replan_rejects_an_empty_segment_before_any_state_changes():
    previous, owner, poses = fixture()
    refiner = previous.arm_policy.tracking._refiner
    with pytest.raises(ValueError, match="nonempty"):
        advance_after_completed(previous, owner, poses, np.ones(40), np.eye(4), generated_step=100, receipt={})
    assert previous.arm_policy.tracking._refiner is refiner
    assert owner.total_snapshot()["completed_periods"] == 0


@pytest.mark.parametrize("corruption", ["time", "anchor", "pending", "inner_loss", "gripper", "buffer"])
def test_midsegment_replan_rejects_unsafe_boundary_without_switching(corruption):
    policy, owner, poses = fixture()
    complete_prefix(policy, 3)
    arm, inner, refiner = policy.arm_policy, policy.inner_history, policy.arm_policy.tracking._refiner
    anchor, step = np.eye(4), 103
    if corruption == "time":
        step += 1
    elif corruption == "anchor":
        anchor[0, 3] = 0.1
    elif corruption == "pending":
        arm.tracking.history.pending_plan = np.zeros((1, 6))
    elif corruption == "inner_loss":
        policy.last_inner_loss = True
    elif corruption == "gripper":
        policy.gripper.staged = True
    else:
        arm.joint_reader = lambda _env: (np.full(7, 0.96), np.tile([-1.0, 1.0], (7, 1)), np.zeros((6, 7)))
    with pytest.raises(ValueError):
        advance_after_completed(policy, owner, poses, np.ones(40), anchor, generated_step=step, receipt={})
    assert arm.tracking._refiner is refiner and inner.snapshot()["completed_periods"] == 3
    assert owner.total_snapshot()["completed_periods"] == 3


def test_activation_failure_restores_old_refiner_and_runtime_receipts(monkeypatch):
    policy, owner, poses = fixture()
    complete_prefix(policy, 3)
    inner, tracking, refiner = policy.inner_history, policy.arm_policy.tracking, policy.arm_policy.tracking._refiner
    policy.receipt.update(shared_shift_traces=[{}] * 3, causal_shift_fallback_traces=[{}] * 3)
    runtime = {"segment_receipts": [policy.receipt], "segment_trace_starts": [{"shared": 0, "causal": 0}]}
    before = copy.deepcopy(runtime)

    def fail(_view):
        raise ValueError("activation failed")

    monkeypatch.setattr(owner, "activate_segment", fail)
    with pytest.raises(ValueError, match="activation failed"):
        advance_recorded_segment(
            policy, owner, poses, np.ones(40), np.eye(4), generated_step=103, runtime_receipt=runtime
        )
    assert runtime == before and tracking._refiner is refiner and inner.snapshot()["completed_periods"] == 3


def test_recorded_boundary_preserves_shared_trace_indices():
    policy, owner, poses = fixture()
    complete_prefix(policy, 3)
    policy.receipt.update(shared_shift_traces=[{}] * 3, causal_shift_fallback_traces=[{}] * 3)
    runtime = {"segment_receipts": [policy.receipt], "segment_trace_starts": [{"shared": 0, "causal": 0}]}
    following = advance_recorded_segment(
        policy, owner, poses, np.ones(40), np.eye(4), generated_step=103, runtime_receipt=runtime
    )
    assert runtime["segment_receipts"][-1] is following.receipt
    assert runtime["segment_trace_starts"][-1] == {"shared": 3, "causal": 3}
