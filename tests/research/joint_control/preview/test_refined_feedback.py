"""Feedback histories follow only the action that was actually dispatched."""

from types import SimpleNamespace

import numpy as np
import pytest

from s2a.research.joint_control.preview.refined_feedback import RefinedPreviewFeedback
from s2a.research.joint_control.preview.tracking_preview_transaction import TrackingPreviewTransaction
from s2a.research.response_models.causal.coupled_response_history import CoupledResponseHistory

CONTRACT = {
    "input_min": [-1.0] * 6,
    "input_max": [1.0] * 6,
    "output_min": [-0.05] * 3 + [-0.5] * 3,
    "output_max": [0.05] * 3 + [0.5] * 3,
    "control_freq": 20,
    "control_frame": "MuJoCo world",
}
JAC = np.c_[np.eye(6), np.zeros(6)]
LIMITS = np.tile([-1.0, 1.0], (7, 1))


def fixture():
    controller = SimpleNamespace(kp=np.ones(6) * 150, kd=np.ones(6) * 25, goal_pos=np.zeros(3), goal_ori=np.eye(3))
    env = SimpleNamespace(robots=[SimpleNamespace(controller=controller)])
    plan = np.tile([0.1, 0, 0, 0, 0, 0], (8, 1))

    def preview(*args, **kwargs):
        return {"command": plan[0].copy(), "preview": {"planned_commands": plan[:4].copy()}}

    def refine(
        _env, _dynamics, _site, _twist, _poses, _anchor, _index, _seed, _previous, _contract, history, *, site_from_tcp
    ):
        history.stage_plan(plan)
        return {
            "command": plan[0].copy(),
            "planned_commands": plan.copy(),
            "nominal_response": np.zeros(12),
            "expected_goal_pos": np.array([0.005, 0, 0]),
            "expected_goal_ori": np.eye(3),
            "route": "test",
            "evaluations": 1,
            "optimization": None,
        }

    def dynamics(_env, _jacobian):
        return np.eye(7), np.zeros(7), np.eye(4)

    def joints(_env):
        return np.zeros(7), LIMITS, JAC

    receipt = {}
    policy = RefinedPreviewFeedback(
        env,
        CoupledResponseHistory(),
        TrackingPreviewTransaction(refiner=refine),
        np.tile(np.eye(4), (8, 1, 1)),
        np.eye(4),
        np.eye(4),
        0,
        np.zeros(7),
        CONTRACT,
        receipt,
        preview_solver=preview,
        dynamics_reader=dynamics,
        joint_reader=joints,
    )
    return env, policy, receipt


def test_no_next_decision_before_the_dispatched_action_is_observed():
    env, policy, receipt = fixture()
    action = policy.select(np.zeros(7), LIMITS, JAC, np.zeros(7), 0)
    with pytest.raises(ValueError, match="observed"):
        policy.select(np.zeros(7), LIMITS, JAC, action["command"], 1)
    env.robots[0].controller.goal_pos = np.array([0.005, 0, 0])
    observed = policy.observe(env, action["command"], 0)
    assert observed["completed_periods"] == 1
    assert observed["completed_history_ready"] is True
    policy.select(np.zeros(7), LIMITS, JAC, action["command"], 1)
    assert receipt["preview_calls"] == receipt["refiner_calls"] == {"attempted": 2, "completed": 2}


def test_observation_must_match_the_committed_action_and_controller_goal():
    env, policy, _receipt = fixture()
    action = policy.select(np.zeros(7), LIMITS, JAC, np.zeros(7), 0)
    bad = action["command"].copy()
    bad[0] += 0.01
    with pytest.raises(ValueError, match="differs"):
        policy.observe(env, bad, 0)
    with pytest.raises(AssertionError):
        policy.observe(env, action["command"], 0)
    env.robots[0].controller.goal_pos = np.array([0.005, 0, 0])
    policy.observe(env, action["command"], 0)


def test_select_rejects_a_fabricated_previous_command():
    _env, policy, _receipt = fixture()
    with pytest.raises(ValueError, match="actual last dispatch"):
        policy.select(np.zeros(7), LIMITS, JAC, np.ones(7) * 0.1, 0)


def test_select_rejects_inconsistent_completed_history_lengths():
    _env, policy, _receipt = fixture()
    policy.tracking.history.completed_periods = 1
    with pytest.raises(ValueError, match="same completed periods"):
        policy.select(np.zeros(7), LIMITS, JAC, np.zeros(7), 0)


def test_explicit_gripper_command_is_bound_to_the_completed_seven_dimensional_action():
    env, policy, _receipt = fixture()
    action = policy.select(np.zeros(7), LIMITS, JAC, np.zeros(7), 0, gripper_command=1.0)
    assert action["command"][6] == 1.0
    env.robots[0].controller.goal_pos = np.array([0.005, 0, 0])
    wrong = action["command"].copy()
    wrong[6] = 0.0
    with pytest.raises(ValueError, match="differs"):
        policy.observe(env, wrong, 0)
    policy.observe(env, action["command"], 0)
    assert policy.previous[6] == 1.0


@pytest.mark.parametrize("gripper_command", [True, 1.01, -1.01, float("nan"), float("inf"), "0"])
def test_invalid_gripper_command_is_rejected_before_preparing_arm_history(gripper_command):
    _env, policy, receipt = fixture()
    with pytest.raises(ValueError, match="bounded gripper"):
        policy.select(np.zeros(7), LIMITS, JAC, np.zeros(7), 0, gripper_command=gripper_command)
    assert policy.pending is None and policy.tracking.phase == "ready"
    assert receipt["preview_calls"] == receipt["refiner_calls"] == {"attempted": 0, "completed": 0}
