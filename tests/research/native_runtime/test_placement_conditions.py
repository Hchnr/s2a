"""Native placement conditions retain contact and exact boundary semantics."""

import numpy as np
import pytest

from s2a.research.native_runtime.placement_conditions import placement_conditions


def test_object_placement_requires_height_distance_and_contact():
    kwargs = dict(task_id=8, target=[0.01, 0, 0.1], support=[0, 0, 0], contact=True)
    assert placement_conditions(**kwargs)["reconstructed_success"] is True
    assert not placement_conditions(**dict(kwargs, contact=False))["reconstructed_success"]
    assert not placement_conditions(**dict(kwargs, target=[0.03, 0, 0.1]))["reconstructed_success"]
    assert not placement_conditions(**dict(kwargs, target=[0.01, 0, -0.1]))["reconstructed_success"]
    assert placement_conditions(**dict(kwargs, task_id=6, target=[0, 0, 0]))["reconstructed_success"]


def test_wine_site_uses_recorded_rotation_and_strict_height_bounds():
    rotation = np.array([[0.0, -1, 0], [1, 0, 0], [0, 0, 1]])
    kwargs = dict(
        task_id=9,
        target=[0.01, 0, 0.02],
        support=[0, 0, 0],
        contact=True,
        site_position=[0, 0, 0],
        site_rotation=rotation,
        site_size=[0.02, 0.015, 0.005],
    )
    result = placement_conditions(**kwargs)
    assert result["reconstructed_success"] is True
    assert result["site_local_m"] == [0.0, 0.01, 0.02]
    assert not placement_conditions(**dict(kwargs, target=[0.015, 0, 0.02]))["reconstructed_success"]
    assert not placement_conditions(**dict(kwargs, target=[0.01, 0, 0.0]))["reconstructed_success"]
    assert not placement_conditions(**dict(kwargs, contact=False))["reconstructed_success"]


def test_nonfinite_or_unregistered_state_is_rejected():
    with pytest.raises(ValueError):
        placement_conditions(6, [float("nan"), 0, 0], [0, 0, 0], True)
    with pytest.raises(ValueError):
        placement_conditions(7, [0, 0, 0], [0, 0, 0], True)
    with pytest.raises(ValueError):
        placement_conditions(9, [0, 0, 0], [0, 0, 0], True)
