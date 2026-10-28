"""Adaptive recovery preserves original pose caps and causal plan history."""

import numpy as np
import pytest

from s2a.fast_long_joint_preview import build_joint_preview, rollout_joint_preview
from s2a.osc import OSCInterface
from s2a.terminal_margin_hybrid import refine_long_plan
from s2a.terminal_margin_hybrid_v2 import refine_long_plan as strict_refine


@pytest.mark.parametrize("solver", [refine_long_plan, strict_refine])
@pytest.mark.parametrize(
    "position,selection",
    [(0.94, "interior_witness"), (0.96, "adaptive_margin_witness"), (1.001, "preventive_fallback")],
)
def test_fixed_adaptive_and_soft_routes_are_separate(position, selection, solver):
    q = np.zeros(7)
    q[0] = position
    model = build_joint_preview(
        np.eye(7), np.c_[np.eye(6), np.zeros(6)], np.ones(6) * 100, np.ones(6) * 20, q, q, null_kp=10.0, dt=0.002
    )
    seed = np.zeros((8, 6))
    seed[:, 3] = 1e-9
    interface = OSCInterface(-np.ones(6), np.ones(6), -np.ones(6), np.ones(6))
    target = rollout_joint_preview(model, np.zeros(7), seed, np.eye(4), -np.ones(7), np.ones(7), enforce_limits=True)
    r = solver(
        model,
        np.zeros(7),
        seed,
        np.zeros(6),
        interface,
        np.eye(4),
        np.eye(4),
        target["pose"],
        target["twist"],
        -np.ones(7),
        np.ones(7),
        initial_guess=seed,
    )
    assert r["selection"] == selection
    assert r["terminal_constraint_enforced"] == (selection == "interior_witness")
    if r["terminal_constraint_enforced"]:
        assert np.all(r["terminal_signed_velocities_rad_s"] <= 1e-10)
    assert r["response_evaluations"] == sum(
        r[k]
        for k in [
            "hard_response_evaluations",
            "adaptive_response_evaluations",
            "refinement_response_evaluations",
            "soft_response_evaluations",
        ]
    )
    assert r["response_evaluations"] <= 640
    if selection == "adaptive_margin_witness":
        assert not r["requested_margin_satisfied"]
        assert r["minimum_inner_margin_rad"] > 0
        assert r["joint_margin_rad"] < 0.05
        np.testing.assert_allclose(r["first_pose_caps"], [0.01, 0.05], atol=1e-12, rtol=0)
