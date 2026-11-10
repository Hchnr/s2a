"""A complete action commits both histories only after observed execution."""

from types import SimpleNamespace

import numpy as np
import pytest

from s2a.research.joint_control.preview.loaded_gripper_feedback import LoadedGripperFeedback
from s2a.research.joint_control.preview.quality_feedback_refiner import QualityFeedbackRefiner, QualitySelectionStopped
from s2a.research.response_models.causal.causal_inner_error import CausalInnerError


class FakeGripper:
    def __init__(self):
        self.pending = None
        self.commits = self.aborts = 0

    def prepare(self, *args, **kwargs):
        self.pending = {
            "command": 1.0,
            "detail": {
                "plan": {"valid_sequences": 6561, "rejected_nonfinite_sequences": 0, "rejected_velocity_sequences": 0}
            },
        }
        return self.pending

    def commit(self, token, *, actual_command, completed_steps):
        assert token is self.pending and actual_command == 1.0 and completed_steps == 1
        self.pending = None
        self.commits += 1

    def abort(self, token):
        assert token is self.pending
        self.pending = None
        self.aborts += 1

    def state(self):
        return {"commits": self.commits, "staged": self.pending is not None, "faulted": False}


class FakeArm:
    def __init__(self, *, reject=False):
        self.reject = reject
        self.observed = 0

    def select(self, _q, _limits, _jacobian, _previous, _offset, *, gripper_command):
        if self.reject:
            raise QualitySelectionStopped("quality_no_witness", "test")
        return {"command": np.r_[np.zeros(6), gripper_command], "newly_selected": True}

    def observe(self, _env, command, _offset):
        assert command[6] == 1.0
        self.observed += 1
        return {"observed": self.observed}


class FakeHistory:
    def __init__(self):
        self.steps = []

    def complete(self, step, inner):
        self.steps.append(step)
        return np.zeros(7)

    def snapshot(self):
        return {"steps": list(self.steps)}


def fixture(*, reject=False, initial_inner_loss=False, max_steps=8):
    env, arm, gripper, history, receipt = object(), FakeArm(reject=reject), FakeGripper(), FakeHistory(), {}
    policy = LoadedGripperFeedback(
        env,
        arm,
        gripper,
        history,
        np.tile(np.eye(4), (40, 1, 1)),
        np.ones(40),
        8,
        np.eye(4),
        136,
        receipt,
        dynamics_reader=lambda _env, _jac: (np.eye(7), np.zeros(7), np.eye(4)),
        gripper_reader=lambda _env: {"opening": 0.9, "motor": np.zeros(2), "period_velocity": 0.0},
        initial_inner_loss=initial_inner_loss,
        max_steps=max_steps,
    )
    return env, policy, arm, gripper, history, receipt


def test_complete_action_commits_both_histories_only_after_25_inner_samples():
    env, policy, arm, gripper, history, receipt = fixture()
    limits = np.tile([-1.0, 1.0], (7, 1))
    decision = policy.select(np.zeros(7), limits, np.zeros((6, 7)), np.zeros(7), 0)
    assert decision["command"][6] == 1.0 and gripper.commits == arm.observed == 0
    with pytest.raises(ValueError, match="complete first"):
        policy.select(np.zeros(7), limits, np.zeros((6, 7)), np.zeros(7), 0)
    with pytest.raises(ValueError, match="25 completed"):
        policy.complete(env, decision["command"], 0, np.zeros((24, 7)))
    wrong = decision["command"].copy()
    wrong[6] = 0.0
    with pytest.raises(ValueError, match="differs"):
        policy.complete(env, wrong, 0, np.zeros((25, 7)))
    policy.complete(env, decision["command"], 0, np.zeros((25, 7)))
    assert gripper.commits == arm.observed == 1 and history.steps == [136]
    assert receipt["gripper_proposals"]["candidate_sequences"] == 6561


def test_no_witness_aborts_staged_gripper_before_dispatch():
    _env, policy, _arm, gripper, _history, receipt = fixture(reject=True)
    result = policy.select(np.zeros(7), np.tile([-1.0, 1.0], (7, 1)), np.zeros((6, 7)), np.zeros(7), 0)
    assert result["stop"]["reason"] == "quality_no_witness"
    assert result["stop"]["gripper_proposal"]["plan"]["valid_sequences"] == 6561
    assert result["stop"]["gripper_proposal_dispatched"] is False
    assert gripper.aborts == 1 and gripper.commits == 0 and policy.pending is None
    assert receipt["gripper_proposals"]["aborted_before_dispatch"] == 1


def test_buffer_loss_stops_before_next_gripper_or_arm_proposal():
    env, policy, _arm, gripper, _history, receipt = fixture()
    limits = np.tile([-1.0, 1.0], (7, 1))
    q = np.zeros(7)
    q[0] = 0.96
    result = policy.select(q, limits, np.zeros((6, 7)), np.zeros(7), 0)
    assert result["stop"]["reason"] == "observed_joint_buffer_loss"
    assert receipt["gripper_proposals"]["attempted"] == 0
    first = policy.select(np.zeros(7), limits, np.zeros((6, 7)), np.zeros(7), 0)
    inner = np.zeros((25, 7))
    inner[-1, 0] = 0.96
    policy.complete(env, first["command"], 0, inner)
    result = policy.select(np.zeros(7), limits, np.zeros((6, 7)), first["command"], 1)
    assert result["stop"]["reason"] == "observed_joint_buffer_loss" and gripper.commits == 1


def test_final_prefix_inner_loss_blocks_first_new_action():
    _env, policy, _arm, gripper, _history, receipt = fixture(initial_inner_loss=True)
    result = policy.select(np.zeros(7), np.tile([-1.0, 1.0], (7, 1)), np.zeros((6, 7)), np.zeros(7), 0)
    assert result["stop"]["reason"] == "observed_joint_buffer_loss"
    assert gripper.commits == 0 and receipt["gripper_proposals"]["attempted"] == 0


def test_explicit_16_step_contract_allows_a_later_causal_period():
    _env, policy, _arm, _gripper, _history, _receipt = fixture(max_steps=16)
    result = policy.select(np.zeros(7), np.tile([-1.0, 1.0], (7, 1)), np.zeros((6, 7)), np.zeros(7), 8)
    assert result["command"][6] == 1.0
    with pytest.raises(ValueError, match="aligned cursor"):
        fixture(max_steps=24)


def switch_fixture():
    inner, receipt, gripper = CausalInnerError(), {}, FakeGripper()
    for step in range(360, 368):
        inner.stage(step, np.zeros((25, 7)))
        inner.complete(step, np.zeros((25, 7)))
    refiner = QualityFeedbackRefiner(inner, receipt, start_step=360, start_cursor=16)
    coupled = SimpleNamespace(pending=None, completed_periods=368)
    tracking = SimpleNamespace(phase="ready", history=SimpleNamespace(completed_periods=368), _refiner=refiner)
    arm = SimpleNamespace(
        pending=None, coupled=coupled, tracking=tracking, initial_periods=360, cursor=16, previous=np.arange(7)
    )
    poses = np.tile(np.eye(4), (40, 1, 1))
    policy = LoadedGripperFeedback(
        object(), arm, gripper, inner, poses, np.ones(40), 16, np.eye(4), 360, receipt, max_steps=16
    )
    return policy, arm, refiner, inner, gripper


def test_switch_preserves_completed_arm_gripper_and_inner_history():
    policy, arm, refiner, inner, gripper = switch_fixture()
    new_poses = np.tile(np.eye(4), (40, 1, 1))
    new_poses[:, 0, 3] = 0.02
    new_openings = np.full(40, 0.9)
    before = inner.snapshot()
    action = arm.previous.copy()
    grip = gripper.state()
    result = policy.rebind_reference(new_poses, new_openings, np.eye(4), offset=8, next_cursor=8, generated_step=360)
    assert result == {"completed_offset": 8, "next_cursor": 8, "generated_step": 360, "cursor_base": 0}
    assert arm.cursor == policy.cursor == refiner.start_cursor == 0
    assert arm.coupled.completed_periods == arm.tracking.history.completed_periods == 368
    np.testing.assert_array_equal(arm.previous, action)
    np.testing.assert_array_equal(inner.snapshot()["inner_position_error_rad"], before["inner_position_error_rad"])
    assert gripper.state() == grip and inner.snapshot()["completed_periods"] == 8
    np.testing.assert_array_equal(arm.targets, new_poses)


@pytest.mark.parametrize("corruption", ["wrong_time", "pending", "inner_loss", "gripper_staged"])
def test_switch_rejects_uncompleted_history_or_misaligned_reference(corruption):
    policy, arm, _refiner, _inner, gripper = switch_fixture()
    cursor = 8
    if corruption == "wrong_time":
        cursor = 9
    elif corruption == "pending":
        arm.pending = {}
    elif corruption == "inner_loss":
        policy.last_inner_loss = True
    else:
        gripper.pending = {}
    with pytest.raises(ValueError):
        policy.rebind_reference(
            np.tile(np.eye(4), (40, 1, 1)), np.ones(40), np.eye(4), offset=8, next_cursor=cursor, generated_step=360
        )
    assert arm.cursor == policy.cursor == 16
