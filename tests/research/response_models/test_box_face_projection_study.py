"""The study must keep each original target and joint-range admission."""

import numpy as np
import pytest

from s2a.research.joint_control.tracking.directional_slide_box_projection import (
    project_directional_slide_boxes_reference,
)
from scripts.tracking.response_models.box_face_projection_study import compare_endpoint


def inputs():
    boxes = [("table", np.eye(4), np.array([0.3, 0.3, 0.02])), ("stove", np.eye(4), np.array([0.1, 0.1, 0.01]))]
    decision = {"original_target_tcp": np.eye(4), "decision_vertices_tcp": np.array([[0.01, 0, 0], [-0.01, 0, 0]])}
    config = dict(
        position_uncertainty_m=0.01,
        rotation_uncertainty_rad=0.05,
        additional_shape_uncertainty_m=0.0,
        clearance_m=0.001,
        numerical_margin_m=1e-9,
        max_translation_m=0.05,
        max_box_combinations_per_endpoint=36,
    )
    vertical = project_directional_slide_boxes_reference(
        decision["original_target_tcp"],
        decision["decision_vertices_tcp"],
        [0, 1, 0],
        -0.03,
        -0.04,
        0,
        boxes,
        position_uncertainty=0.01,
        rotation_uncertainty=0.05,
        additional_shape_uncertainty=0,
        clearance=0.001,
        numerical_margin=1e-9,
        max_translation=0.05,
    )
    return (
        decision,
        {"slide_position_m": -0.03, "slide_axis_tcp": [0, 1, 0], "directional": vertical},
        (-0.04, 0),
        boxes,
        config,
    )


def test_endpoint_comparison_includes_vertical_candidate():
    result = compare_endpoint(*inputs())
    assert result["status"] == "evaluated"
    assert result["faces"]["face_combinations_checked"] == 36
    assert result["faces"]["required_translation_m"] <= result["vertical"]["required_translation_m"] + 1e-12


def test_endpoint_outside_joint_range_is_not_reinterpreted():
    decision, previous, interval, boxes, config = inputs()
    previous.update(slide_position_m=-0.041, directional=None)
    assert compare_endpoint(decision, previous, interval, boxes, config)["status"] == "joint_out_of_range"
    previous["directional"] = {"required_translation_m": 0.01}
    with pytest.raises(ValueError, match="same joint-range admission"):
        compare_endpoint(decision, previous, interval, boxes, config)
