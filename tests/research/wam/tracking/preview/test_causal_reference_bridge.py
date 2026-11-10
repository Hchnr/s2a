"""Check absolute endpoint alignment and the SE(3)/opening bridge contract."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from s2a.geometry import validate_transform
from s2a.research.wam.tracking.preview.causal_reference_bridge import build_reference_bridge


def references():
    old, latest = np.tile(np.eye(4), (40, 1, 1)), np.tile(np.eye(4), (40, 1, 1))
    old_rotation = Rotation.from_euler("x", 90, degrees=True).as_matrix()
    relative_rotation = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    old[:, :3, :3] = old_rotation
    latest[:, :3, :3] = old_rotation @ relative_rotation
    old[:, 0, 3] = np.arange(40) / 100
    latest[:, 1, 3] = np.arange(40) / 50
    return old, np.linspace(0.1, 0.3, 40), latest, np.linspace(0.9, 0.7, 40)


def bridge(refs, **overrides):
    kwargs = {"old_step": 344, "latest_step": 360, "start_offset": 8, "transition_steps": 8}
    kwargs.update(overrides)
    return build_reference_bridge(*refs, **kwargs)


def test_same_absolute_endpoint_geodesic_and_exact_prefix_and_suffix():
    refs = references()
    before = [item.copy() for item in refs]
    old, old_openings, latest, latest_openings = refs
    result = bridge(refs)
    poses, openings, weights = result["poses"], result["openings"], result["weights"]
    np.testing.assert_array_equal(poses[:8], old[16:24])
    np.testing.assert_array_equal(openings[:8], old_openings[16:24])
    np.testing.assert_array_equal(poses[15:], latest[15:])
    np.testing.assert_array_equal(openings[15:], latest_openings[15:])
    np.testing.assert_array_equal(weights[:16], [0] * 8 + [i / 8 for i in range(1, 9)])
    for index in range(8, 15):
        alpha = (index - 7) / 8
        old_index = index + 16
        np.testing.assert_allclose(
            poses[index, :3, 3], (1 - alpha) * old[old_index, :3, 3] + alpha * latest[index, :3, 3]
        )
        np.testing.assert_allclose(
            openings[index], (1 - alpha) * old_openings[old_index] + alpha * latest_openings[index]
        )
        expected = (
            Rotation.from_euler("x", 90, degrees=True).as_matrix()
            @ Rotation.from_euler("z", 90 * alpha, degrees=True).as_matrix()
        )
        np.testing.assert_allclose(poses[index, :3, :3], expected, atol=1e-14)
    validate_transform(poses)
    assert result["origin_step"] == result["latest_step"] == 360
    for item, original in zip(refs, before, strict=True):
        np.testing.assert_array_equal(item, original)
    poses[0, 0, 3] = -1
    assert refs[0][16, 0, 3] != -1


@pytest.mark.parametrize(
    "overrides",
    [
        {"old_step": 360},
        {"old_step": -1},
        {"latest_step": 344},
        {"old_step": True},
        {"start_offset": -1},
        {"transition_steps": 0},
        {"transition_steps": 33},
        {"old_step": 320},
    ],
)
def test_rejects_noncausal_or_insufficiently_overlapping_chunks(overrides):
    with pytest.raises(ValueError):
        bridge(references(), **overrides)


@pytest.mark.parametrize("corruption", ["shape", "rotation", "last_row", "nan", "opening_range", "opening_shape"])
def test_rejects_invalid_geometry_or_opening_contract(corruption):
    refs = list(references())
    if corruption == "shape":
        refs[2] = refs[2][:-1]
    elif corruption == "rotation":
        refs[0][0, 0, 0] += 0.1
    elif corruption == "last_row":
        refs[2][0, 3, 3] = 2
    elif corruption == "nan":
        refs[0][0, 0, 3] = np.nan
    elif corruption == "opening_range":
        refs[1][0] = 1.1
    else:
        refs[3] = refs[3][:-1]
    with pytest.raises(ValueError):
        bridge(refs)
