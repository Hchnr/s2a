"""Hard-model witness and measured-state margin fallback remain distinguishable."""

import numpy as np
import pytest

from s2a.fast_long_joint_preview import build_joint_preview, rollout_joint_preview
from s2a.geometry import pose_error
from s2a.interior_joint_hybrid import refine_long_plan
from s2a.osc import OSCInterface


@pytest.mark.parametrize("position,selection", [(0.94, "interior_witness"), (0.96, "preventive_fallback")])
def test_selection_and_pose_caps_preserve_explicit_failed_margin_reason(position, selection):
    q = np.zeros(7)
    q[0] = position
    model = build_joint_preview(
        np.eye(7), np.c_[np.eye(6), np.zeros(6)], np.ones(6) * 100, np.ones(6) * 20, q, q, null_kp=10.0, dt=0.002
    )
    seed = np.zeros((8, 6))
    seed[:, 0], seed[:, 3] = 0.05, 1e-9
    lo, hi = -np.ones(7), np.ones(7)
    target = rollout_joint_preview(model, np.zeros(7), seed, np.eye(4), lo, hi, enforce_limits=True)
    interface = OSCInterface(-np.ones(6), np.ones(6), -np.ones(6), np.ones(6))
    result = refine_long_plan(
        model, np.zeros(7), seed, np.zeros(6), interface, np.eye(4), np.eye(4), target["pose"], target["twist"], lo, hi
    )
    assert result["selection"] == selection
    if selection == "interior_witness":
        assert result["interior_diagnostic"]["witness_found"]
        assert result["interior_diagnostic"]["minimum_inner_margin_rad"] >= 0.05 - 1e-10
    else:
        assert result["interior_diagnostic"]["status"] == "initial_outside_requested_interior"
        assert not result["interior_diagnostic"]["witness_found"]
    error = pose_error(target["pose"][0], result["rollout"]["pose"][0])
    assert np.linalg.norm(error[:3]) <= result["first_pose_caps"][0] + 1e-10
    assert np.linalg.norm(error[3:]) <= result["first_pose_caps"][1] + 1e-10
    assert result["response_evaluations"] <= 320
    assert np.max(abs(np.diff(np.vstack((np.zeros(6), result["planned_commands"])), axis=0))) <= 0.25 + 1e-12
