"""The physical probe's candidate decisions remain bounded and reference-causal."""

from itertools import product

import numpy as np
import pytest

from s2a.controllers.osc import OSCInterface
from s2a.geometry import transform
from s2a.research.joint_control.tracking.tabletop_projection_feedback import select_tabletop_feedback


def example(mode, *, cap=0.05):
    interface = OSCInterface(-np.ones(6), np.ones(6), -0.05 * np.ones(6), 0.05 * np.ones(6))
    vertices = np.array(list(product((-0.01, 0.01), repeat=3)))
    table = transform(np.array([0.0, 0.0, -0.05]), np.eye(3))
    previous = np.array([0.1, -0.1, 0.0, 0.0, 0.0, 0.0, -0.6])
    result = select_tabletop_feedback(
        np.eye(4),
        np.eye(4),
        np.eye(4),
        vertices,
        table,
        [0.2, 0.2, 0.05],
        previous,
        interface,
        mode=mode,
        clearance=0.001,
        numerical_margin=1e-9,
        max_translation=cap,
        command_step_limit=0.25,
    )
    return result, previous


def test_original_and_projected_commands_keep_the_same_controller_limits_and_gripper():
    original, previous = example("original")
    nominal, _ = example("nominal")
    bounded, _ = example("pose_bounded")
    assert original["reference_translation_m"] == 0
    assert 0 < nominal["reference_translation_m"] < bounded["reference_translation_m"]
    for result in (original, nominal, bounded):
        assert result["newly_selected"]
        assert result["command"][6] == previous[6]
        assert np.max(np.abs(result["command"][:6] - previous[:6])) <= 0.25
        np.testing.assert_array_equal(result["selected_target_tcp"][:3, :3], np.eye(3))
        np.testing.assert_array_equal(result["original_target_tcp"], np.eye(4))
    assert bounded["rate_limited"]


def test_excessive_correction_stops_before_any_action_is_selected():
    result, _ = example("pose_bounded", cap=0.005)
    assert result["newly_selected"] is False
    assert result["stop"]["reason"] == "projection_correction_exceeds_limit"
    assert result["stop"]["projection"]["projected_tcp"] is None
    assert "command" not in result


def test_unknown_variant_is_rejected():
    with pytest.raises(ValueError):
        example("unsafe")
