"""Reset feedback uses only generated references and actual initialized history."""

from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest

from s2a.research.gripper.gripper_position_hold import GripperPositionHoldConfig
from s2a.research.joint_control.preview import reset_feedback_policy as policy_module
from tests.research.joint_control.preview.test_refined_feedback import CONTRACT
from tests.research.joint_control.tracking.test_loaded_velocity_control import parameters


def source():
    poses = np.tile(np.eye(4), (40, 1, 1))
    case = {
        "chunk": {"effective_targets_tcp": poses, "anchor_tcp": np.eye(4), "openings": np.ones(40)},
        "controller": {
            "source_from_site": np.eye(4),
            "contract": CONTRACT,
            "gripper_config": asdict(GripperPositionHoldConfig()),
            "gripper_velocity_envelope": 0.3,
        },
        "tool": np.eye(4),
        "gripper_parameters": parameters(),
        "future_state": "must not enter policy",
    }
    terminal = {"sample": {"opening": 0.85, "native_success": True, "future_object": "scoring only"}}
    return case, terminal


def test_only_current_opening_generated_reference_and_fixed_contract_are_exposed():
    case, terminal = source()
    inputs = policy_module.policy_inputs(case, terminal)
    assert set(inputs) == {
        "poses",
        "anchor",
        "openings",
        "tool",
        "source_from_site",
        "contract",
        "gripper_config",
        "velocity_envelope",
        "gripper_parameters",
        "initial_opening",
    }
    assert inputs["initial_opening"] == 0.85
    case["chunk"]["effective_targets_tcp"][0, 0, 3] = 0.2
    assert inputs["poses"][0, 0, 3] == 0.0
    terminal["sample"]["opening"] = 1.1
    with pytest.raises(ValueError, match="current measured opening"):
        policy_module.policy_inputs(case, terminal)


def test_fresh_policy_has_no_plan_and_stops_on_initial_joint_buffer_loss(monkeypatch):
    case, terminal = source()
    inputs = policy_module.policy_inputs(case, terminal)
    jac = np.c_[np.eye(6), np.zeros(6)]
    limits = np.tile([-1.0, 1.0], (7, 1))
    monkeypatch.setattr(policy_module, "current_joint_site_state", lambda _env: (np.zeros(7), limits, jac))
    monkeypatch.setattr(policy_module, "current_robot_dynamics", lambda _env, _jac: (np.eye(7), np.zeros(7), np.eye(4)))
    env = SimpleNamespace(robots=[SimpleNamespace(controller=SimpleNamespace())])
    receipt = {}
    policy, previous = policy_module.build_reset_policy(env, inputs, receipt)
    assert policy.arm_policy.coupled.completed_periods == policy.arm_policy.tracking.history.completed_periods == 0
    assert (
        policy.arm_policy.tracking.history.last_plan is None
        and policy.inner_history.snapshot()["completed_periods"] == 0
    )
    np.testing.assert_array_equal(previous, np.zeros(7))
    assert receipt["reset_history"]["inner_error_history_available"] is False
    q = np.zeros(7)
    q[0] = 0.96
    decision = policy.select(q, limits, jac, previous, 0)
    assert decision["stop"]["reason"] == "observed_joint_buffer_loss"
    assert receipt["gripper_proposals"]["attempted"] == 0 and policy.pending is None
