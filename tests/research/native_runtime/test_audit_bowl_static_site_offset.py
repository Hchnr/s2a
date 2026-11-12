"""The static hypothesis moves only pads and keeps the original evidence intact."""

import copy

import numpy as np
import pytest

from scripts.simulation.runtime.audit_bowl_static_site_offset import analyze_point, classify


def test_classification_respects_explicit_numerical_boundary():
    assert classify(2e-8, 1e-8) == "separated"
    assert classify(-2e-8, 1e-8) == "overlap"
    assert classify(1e-9, 1e-8) == "near"
    with pytest.raises(ValueError, match="positive tolerance"):
        classify(0, 0)


def test_only_pad_centers_are_shifted_in_site_coordinates():
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

    def box(name, center):
        return dict(name=name, center_world_m=center, rotation_world=np.eye(3), half_size_m=[0.01] * 3)

    boxes = [box("left", [0, 0, 0]), box("right", [0.2, 0, 0])]
    boxes += [box(f"target{i}", [0, -0.019, 0]) for i in range(40)]
    point = dict(
        pad_alignment=dict(box_projections=boxes, site_rotation_world=rotation),
        contacts=dict(left_pad=[], right_pad=[], other=[], bilateral=False),
    )
    saved = copy.deepcopy(point)
    result = analyze_point(point, [-0.02, 0, 0], 1e-8)
    np.testing.assert_allclose(result["offset"]["pad_translation_world_m"], [0, -0.02, 0])
    assert result["offset"]["pads"]["left"]["strictly_overlapping_targets"]
    assert not result["offset"]["pads"]["right"]["strictly_overlapping_targets"]
    np.testing.assert_array_equal(
        point["pad_alignment"]["site_rotation_world"], saved["pad_alignment"]["site_rotation_world"]
    )
    for actual, expected in zip(
        point["pad_alignment"]["box_projections"], saved["pad_alignment"]["box_projections"], strict=True
    ):
        assert actual["name"] == expected["name"]
        for key in ("center_world_m", "rotation_world", "half_size_m"):
            np.testing.assert_array_equal(actual[key], expected[key])
