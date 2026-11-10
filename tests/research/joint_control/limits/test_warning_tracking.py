"""Exercise fixed-frame target conversion and rate-limited safety feedback."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from s2a.research.joint_control.limits.warning_tracking import select_tracking_action, site_references


def transform(translation, rotation):
    result = np.eye(4)
    result[:3, :3] = Rotation.from_rotvec(rotation).as_matrix()
    result[:3, 3] = translation
    return result


def test_reference_conversion_accounts_for_both_tool_and_site_offsets():
    source = transform([0.3, -0.2, 0.4], [0.2, 0.1, -0.3])
    tcp = transform([0.0, 0.0, 0.12], [0.0, 0.0, 0.4])
    site = transform([0.03, -0.01, 0.0], [0.0, 0.1, 0.0])
    actual, anchor = site_references(np.asarray([source @ tcp]), source @ tcp, tcp, site)
    np.testing.assert_allclose(actual[0], source @ site, atol=1e-12, rtol=0)
    np.testing.assert_allclose(anchor, source @ site, atol=1e-12, rtol=0)


CONTRACT = {
    "input_min": [-1.0] * 6,
    "input_max": [1.0] * 6,
    "output_min": [-0.05] * 3 + [-0.5] * 3,
    "output_max": [0.05] * 3 + [0.5] * 3,
}
LIMITS = np.tile([-1.0, 1.0], (7, 1))


def solve(*args, innovation):
    np.testing.assert_array_equal(innovation, np.zeros(12))
    return {"command": np.zeros(6)}


@pytest.mark.parametrize("position,expected_sign", [(0.95, -1), (-0.95, 1)])
def test_guard_uses_correct_side_and_preserves_box_rate(position, expected_sign):
    q, jacobian = np.zeros(7), np.zeros((6, 7))
    q[4], jacobian[0, 4] = position, 1.0
    args = (np.eye(7), jacobian, q, np.zeros(7), LIMITS, np.eye(4), np.tile(np.eye(4), (8, 1, 1)), np.eye(4), 0)
    result = select_tracking_action(*args, np.zeros(6), CONTRACT, guard=True, solver=solve)
    assert result["guard_active"] is True
    assert np.sign(result["retreat_site_delta"][0]) == expected_sign
    assert abs(result["command"][0]) <= 0.25
    assert result["rate_limited"] is True
    baseline = select_tracking_action(*args, np.zeros(6), CONTRACT, guard=False, solver=solve)
    np.testing.assert_array_equal(baseline["command"], np.zeros(6))


def test_guard_stays_inactive_away_from_limits():
    result = select_tracking_action(
        np.eye(7),
        np.zeros((6, 7)),
        np.zeros(7),
        np.zeros(7),
        LIMITS,
        np.eye(4),
        np.tile(np.eye(4), (8, 1, 1)),
        np.eye(4),
        0,
        np.zeros(6),
        CONTRACT,
        guard=True,
        solver=solve,
    )
    assert result["guard_active"] is False
    np.testing.assert_array_equal(result["command"], np.zeros(6))


def test_preview_must_respect_existing_rate_and_box():
    def bad_preview(*args, **kwargs):
        return {"command": np.ones(6)}

    with pytest.raises(ValueError, match="box and rate"):
        select_tracking_action(
            np.eye(7),
            np.zeros((6, 7)),
            np.zeros(7),
            np.zeros(7),
            LIMITS,
            np.eye(4),
            np.tile(np.eye(4), (8, 1, 1)),
            np.eye(4),
            0,
            np.zeros(6),
            CONTRACT,
            guard=False,
            solver=bad_preview,
        )
