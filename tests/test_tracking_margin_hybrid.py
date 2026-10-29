"""Primary tracking-stage cap violations remain visible through native selection."""

import numpy as np

from s2a.fast_long_joint_preview import build_joint_preview, rollout_joint_preview
from s2a.osc import OSCInterface
from s2a.tracking_margin_hybrid import refine_long_plan


def test_selector_keeps_bounds_and_reports_primary_old_cap_violation():
    q = np.zeros(7)
    q[0] = 0.94
    model = build_joint_preview(
        np.eye(7), np.c_[np.eye(6), np.zeros(6)], np.ones(6) * 100, np.ones(6) * 20, q, q, null_kp=10.0, dt=0.002
    )
    seed = np.zeros((8, 6))
    seed[:, 0] = 0.25
    seed[:, 3] = 1e-9
    interface = OSCInterface(-np.ones(6), np.ones(6), -np.ones(6), np.ones(6))
    target = rollout_joint_preview(model, np.zeros(7), seed, np.eye(4), -np.ones(7), np.ones(7), enforce_limits=False)
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
    )
    assert r["selection"] == "interior_witness"
    assert not r["first_pose_caps_enforced"]
    assert not r["original_first_pose_caps_satisfied"]
    assert r["minimum_inner_margin_rad"] >= 0.05 - 1e-10
    assert r["terminal_constraint_enforced"]
    assert np.all(r["terminal_signed_velocities_rad_s"] <= 1e-10)
