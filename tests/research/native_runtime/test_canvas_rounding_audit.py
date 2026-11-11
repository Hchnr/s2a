"""The comparison detects changed image data and exact interpolation ties."""

import numpy as np
import pytest

from s2a.research.native_runtime.canvas_rounding_audit import bilinear_half_integer_mask, compare_canvases


def test_exact_half_integer_positions_and_seam():
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    image[:, 1, :] = 5
    ties = bilinear_half_integer_mask([image, image])
    assert ties.shape == (960, 640, 3)
    assert ties[0, 1, 0] and ties[0, 2, 0] and ties[0, 3, 0]
    assert not ties[0, 0, 0] and not ties[478:482].any()
    with pytest.raises(ValueError, match="256x256"):
        bilinear_half_integer_mask([image, image[:-1]])


def test_canvas_comparison_preserves_channel_and_pixel_denominators():
    original = np.zeros((960, 640, 3), dtype=np.uint8)
    one = original.copy()
    one[4, 5, :2] = 1
    ties = np.zeros_like(original, dtype=bool)
    ties[4, 5, 0] = True
    result = compare_canvases(original, one.copy(), one, original.copy(), ties)
    assert result == {
        "one_matches_saved": True,
        "two_matches_recorded": True,
        "different_channel_values": 2,
        "different_pixels": 1,
        "max_channel_difference": 1,
        "different_values_on_exact_ties": 1,
    }
    altered = original.copy()
    altered[0, 0, 0] = 1
    assert compare_canvases(original, one, one, altered, ties)["two_matches_recorded"] is False
