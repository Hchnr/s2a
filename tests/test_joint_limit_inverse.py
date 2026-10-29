"""Candidate feasibility, no-limit preservation and unilateral recovery."""

import numpy as np

from s2a.joint_limit_inverse import refine_command
from s2a.joint_limit_response import build_joint_response, predict_joint_response
from s2a.osc import OSCInterface


def setup():
    q = np.zeros(7)
    model = build_joint_response(
        np.eye(7), np.c_[np.eye(6), np.zeros(6)], np.ones(6) * 100, np.ones(6) * 20, q, q, null_kp=10.0, dt=0.002
    )
    interface = OSCInterface(-np.ones(6), np.ones(6), -np.ones(6), np.ones(6))
    return model, interface


def test_no_limit_keeps_original_preview_command_exactly():
    model, interface = setup()
    seed = np.array([0.1, -0.1, 0.05, 0.01, 0.02, -0.01])
    result = refine_command(model, np.zeros(7), seed, np.zeros(6), interface, -np.ones(7), np.ones(7), np.ones(12))
    np.testing.assert_array_equal(result["command"], seed)
    assert result["evaluations"] == 0


def test_refinement_can_leave_outward_plateau_and_recover_feasible_target():
    model, interface = setup()
    lower, upper = -np.ones(7), np.ones(7)
    upper[0] = 0
    desired_command = np.array([-0.1, 0.1, 0, 0, 0, 0])
    response = predict_joint_response(
        model, np.zeros(7), desired_command, lower, upper, substeps=25, enforce_limits=True
    )
    target = np.r_[response["task_displacement"], response["task_velocity"]]
    seed = np.array([0.2, 0, 0, 0, 0, 0])
    result = refine_command(model, np.zeros(7), seed, np.zeros(6), interface, lower, upper, target)
    assert result["candidate_objective"] < result["seed_objective"] * 1e-4
    assert result["command"][0] < 0
    assert np.max(abs(result["command"])) <= 0.25
    assert result["evaluations"] <= 300
