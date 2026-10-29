"""Check attribution and maximum clearance on an uncontrollable joint."""

import numpy as np

from s2a.interior_joint_preview import build_joint_preview
from s2a.joint_constraint_diagnostic import diagnose
from s2a.osc import OSCInterface


def test_uncontrollable_joint_proves_position_error_budget_is_infeasible():
    q = np.zeros(7)
    q[-1] = 0.9475
    model = build_joint_preview(
        np.eye(7), np.c_[np.eye(6), np.zeros(6)], np.full(6, 100.0), np.full(6, 20.0), q, q, null_kp=10.0, dt=0.002
    )
    interface = OSCInterface(-np.ones(6), np.ones(6), -np.ones(6), np.ones(6))
    seed = np.zeros((2, 6))
    seed[:, 3] = 1e-9
    error = np.zeros(7)
    error[-1] = 0.005
    result = diagnose(model, np.zeros(7), seed, np.zeros(6), interface, np.eye(4), -np.ones(7), np.ones(7), error)
    checks = result["constraint_checks"]
    assert checks["all"]["status"] == 2
    assert checks["position_only"]["status"] == 2
    assert checks["without_position"]["success"]
    best = result["first_period_clearance"]
    assert best["success"]
    np.testing.assert_allclose(best["maximum_first_period_error_adjusted_margin_rad"], 0.0475, atol=1e-10, rtol=0)
    assert best["executable_plan"] is False
