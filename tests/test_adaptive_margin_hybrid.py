"""Adaptive recovery preserves original pose caps and causal plan history."""

import numpy as np
import pytest

from s2a.adaptive_margin_hybrid import refine_long_plan
from s2a.adaptive_plan_history import AdaptivePlanHistory
from s2a.fast_long_joint_preview import build_joint_preview, rollout_joint_preview
from s2a.osc import OSCInterface


@pytest.mark.parametrize(
    "position,selection",
    [(0.94, "interior_witness"), (0.96, "adaptive_margin_witness"), (1.001, "preventive_fallback")],
)
def test_fixed_adaptive_and_soft_routes_are_separate(position, selection):
    q = np.zeros(7)
    q[0] = position
    model = build_joint_preview(
        np.eye(7), np.c_[np.eye(6), np.zeros(6)], np.ones(6) * 100, np.ones(6) * 20, q, q, null_kp=10.0, dt=0.002
    )
    seed = np.zeros((8, 6))
    seed[:, 3] = 1e-9
    interface = OSCInterface(-np.ones(6), np.ones(6), -np.ones(6), np.ones(6))
    target = rollout_joint_preview(model, np.zeros(7), seed, np.eye(4), -np.ones(7), np.ones(7), enforce_limits=True)
    r = refine_long_plan(
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


def test_warm_plan_is_available_only_after_completed_actual_period_and_reset_clears_it():
    h = AdaptivePlanHistory()
    plan = np.ones((8, 6)) * 0.1
    h.stage_plan(plan)
    assert h.shifted(8) is None
    plan[:] = 0
    h.commit_prediction(np.eye(4), np.zeros(6), np.zeros(12))
    h.observe_completed(np.eye(4), np.zeros(6))
    np.testing.assert_array_equal(h.shifted(8), np.ones((8, 6)) * 0.1)
    assert h.pending_plan is None
    h.reset()
    assert h.shifted(8) is None
    assert h.pending_plan is None
