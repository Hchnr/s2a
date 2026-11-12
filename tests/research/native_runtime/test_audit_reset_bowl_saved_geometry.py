"""Read-only geometry audit must reject inconsistent metrics and state changes."""

import copy

import pytest

from scripts.simulation.runtime.audit_reset_bowl_saved_geometry import check_point


def point():
    return dict(
        tcp_world_m=[0.3, 0.4, 0.0],
        body_origin_world_m=[0.0, 0.0, 0.0],
        tcp_minus_body_world_m=[0.3, 0.4, 0.0],
        body_origin_distance_m=0.5,
        body_origin_xy_distance_m=0.5,
        tcp_above_body_origin_m=0.0,
        state_change=dict(
            changed_indices=[4],
            max_absolute_change=2e-16,
            quaternion_norms=[
                dict(state_indices=[4, 5, 6, 7], max_absolute_change=2e-16, norm_before=1.0, norm_after=1.0)
            ],
        ),
    )


def test_geometry_audit_rejects_altered_distance_or_nonquaternion_change():
    expected = point()
    check_point(expected)
    changed = copy.deepcopy(expected)
    changed["body_origin_distance_m"] = 0.4
    with pytest.raises(AssertionError):
        check_point(changed)
    changed = copy.deepcopy(expected)
    changed["state_change"]["changed_indices"] = [2]
    with pytest.raises(ValueError, match="quaternion changes"):
        check_point(changed)
