"""A vertex audit catches invalid box intervals and geometric classifications."""

import copy

import numpy as np
import pytest

from scripts.simulation.runtime.audit_bowl_pad_alignment import PADS, TARGET, check_point


def fixture():
    boxes = []
    for name in PADS + TARGET:
        center = [-2, 0, 0] if name == PADS[0] else [2, 0, 0] if name == PADS[1] else [0, 0, 0]
        radius = 0.2 if name in PADS else 0.5
        boxes.append(
            dict(
                name=name,
                center_world_m=center,
                rotation_world=np.eye(3),
                half_size_m=[radius] * 3,
                jaw_axis_interval_m=[center[0] - radius, center[0] + radius],
            )
        )
    a = dict(
        left_pad_world_m=[-2, 0, 0],
        right_pad_world_m=[2, 0, 0],
        pad_midpoint_world_m=[0, 0, 0],
        jaw_axis_world=[1, 0, 0],
        pad_center_separation_m=4,
        body_origin_world_m=[0.1, 0.3, 0.4],
        body_from_pad_midpoint_world_m=[0.1, 0.3, 0.4],
        body_axis_projection_m=0.1,
        body_axis_perpendicular_m=0.5,
        site_rotation_world=np.eye(3),
        site_position_world_m=[0, 0, 0],
        body_from_site_in_site_m=[0.1, 0.3, 0.4],
        box_projections=boxes,
        target_axis_interval_m=[-0.5, 0.5],
    )
    return dict(pad_alignment=a, contacts=dict(left_pad=[], right_pad=[], other=[], bilateral=False))


def test_vertex_audit_recovers_projection_gap():
    value = check_point(fixture())
    assert value["right_pad_projection_gap_m"] == pytest.approx(1.3)


def test_vertex_audit_rejects_corrupt_interval_or_bilateral_flag():
    point = fixture()
    corrupted = copy.deepcopy(point)
    corrupted["pad_alignment"]["box_projections"][12]["jaw_axis_interval_m"][0] += 0.01
    with pytest.raises(AssertionError):
        check_point(corrupted)
    corrupted = copy.deepcopy(point)
    corrupted["contacts"]["bilateral"] = True
    with pytest.raises(ValueError, match="bilateral"):
        check_point(corrupted)
