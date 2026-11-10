"""Check first-warning admission and saved-action-only independent replay."""

import sys

import numpy as np
import pytest

from scripts.simulation.runtime import compare_early_warning_tracking as early


def case_at_first_warning():
    joints = "".join(f'<joint name="robot0_joint{i}" range="-1 1"/>' for i in range(1, 8))
    states = np.zeros((3, 15))
    states[-1, 7] = 0.91
    return {
        "record_xml": f"<mujoco><worldbody><body>{joints}</body></worldbody></mujoco>",
        "saved": {"sim_states": states},
        "spec": {"prefix_steps": 2},
    }


def test_first_warning_must_occur_at_the_bounded_endpoint():
    case = case_at_first_warning()
    result = early.first_warning_from_saved(case)
    assert result["first_warning_endpoint"] == 2
    assert result["minimum_margin_rad"] == pytest.approx(0.09)
    case["saved"]["sim_states"][1, 7] = 0.91
    with pytest.raises(ValueError, match="First causal warning"):
        early.first_warning_from_saved(case)


def test_first_warning_must_still_have_original_joint_buffer():
    case = case_at_first_warning()
    case["saved"]["sim_states"][-1, 7] = 0.97
    with pytest.raises(ValueError, match="buffer"):
        early.first_warning_from_saved(case)


def test_checker_consumes_saved_action_without_importing_either_decision_module(monkeypatch):
    for name in (early.POLICY_MODULE, early.PREVIEW_MODULE, early.previous.POLICY_MODULE):
        monkeypatch.delitem(sys.modules, name, raising=False)
    action, states = early.make_action("checker", "guarded_preview", {}, None, {}, [{"command": [0.0] * 7}])
    result = action(None, None, None, None, 0)
    np.testing.assert_array_equal(result["command"], np.zeros(7))
    assert result["newly_selected"] is False
    assert states is None
    assert early.POLICY_MODULE not in sys.modules
    assert early.PREVIEW_MODULE not in sys.modules
