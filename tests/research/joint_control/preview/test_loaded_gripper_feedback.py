"""A complete action commits both histories only after observed execution."""

import numpy as np
import pytest

from s2a.research.joint_control.preview.loaded_gripper_feedback import LoadedGripperFeedback
from s2a.research.joint_control.preview.quality_feedback_refiner import QualitySelectionStopped


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
        return {"commits": self.commits}


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
