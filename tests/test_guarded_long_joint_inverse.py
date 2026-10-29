"""First executed pose protection during longer action-space anticipation."""

import numpy as np
import pytest

from s2a.fast_guarded_long_joint_inverse import refine_long_plan as fast_refine_long_plan
from s2a.geometry import pose_error
from s2a.guarded_long_joint_inverse import refine_long_plan
from s2a.long_joint_preview import build_joint_preview, rollout_joint_preview
from s2a.osc import OSCInterface


@pytest.mark.parametrize("solver", [refine_long_plan, fast_refine_long_plan])
@pytest.mark.parametrize("exact_first", [True, False])
def test_future_reference_changes_plan_without_sacrificing_exact_first_pose(exact_first, solver):
    q = np.zeros(7)
    model = build_joint_preview(
        np.eye(7), np.c_[np.eye(6), np.zeros(6)], np.ones(6) * 100, np.ones(6) * 20, q, q, null_kp=10.0, dt=0.002
    )
    seed = np.zeros((8, 6))
    seed[:, 3] = 1e-9
    known = seed.copy()
    if not exact_first:
        known[0, 0] = 0.01
        known[:, 3] = 0.01
    known[1:, 0] = [0.02, 0.04, 0.06, 0.08, 0.06, 0.04, 0.02]
    lo, hi = -np.ones(7), np.ones(7)
    target = rollout_joint_preview(model, np.zeros(7), known, np.eye(4), lo, hi, enforce_limits=True)
    original = rollout_joint_preview(model, np.zeros(7), seed, np.eye(4), lo, hi, enforce_limits=True)
    interface = OSCInterface(-np.ones(6), np.ones(6), -np.ones(6), np.ones(6))
    result = solver(
        model, np.zeros(7), seed, np.zeros(6), interface, np.eye(4), np.eye(4), target["pose"], target["twist"], lo, hi
    )
    assert result["accepted"]
    assert result["objective"] < result["seed_objective"] * 1e-6
    before = pose_error(target["pose"][0], original["pose"][0])
    after = pose_error(target["pose"][0], result["rollout"]["pose"][0])
    for a, b in ((0, 3), (3, 6)):
        assert np.linalg.norm(after[a:b]) <= np.linalg.norm(before[a:b]) + 1e-10
    assert np.max(abs(np.diff(np.vstack((np.zeros(6), result["planned_commands"])), axis=0))) <= 0.25 + 1e-12
