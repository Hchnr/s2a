"""Use known rotations and endpoints to audit the independent reference port."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from scripts.verification.reference_ports.check_bounded_release_bridge import verify_selected_reference


def references():
    old = np.repeat(np.eye(4)[None, :, :], 40, axis=0)
    latest = old.copy()
    latest[:, :3, :3] = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    latest[:, 0, 3] = 1
    return old, np.full(40, 0.2), latest, np.full(40, 0.8)


def decision(source="bridge"):
    _old, _old_opening, latest, latest_opening = references()
    poses, openings = latest[16:].copy(), latest_opening[16:].copy()
    if source == "bridge":
        for index in range(8):
            alpha = (index + 1) / 8
            poses[index, :3, :3] = Rotation.from_euler("z", 90 * alpha, degrees=True).as_matrix()
            poses[index, 0, 3] = alpha
            openings[index] = 0.2 + alpha * 0.6
    return {
        "step": 376,
        "absolute_first_endpoint": 377,
        "latest_source_step": 360,
        "committed_source_step": 344,
        "source": source,
        "reason": "bounded_bridge" if source == "bridge" else "not_closing",
        "poses": poses,
        "openings": openings,
    }


@pytest.mark.parametrize("source", ["bridge", "latest"])
def test_verifies_same_time_bridge_or_completed_closing_episode(source):
    checked = verify_selected_reference(decision(source), *references())
    assert checked == {"source": source, "geometry_independently_verified": True, "selection_recomputed": False}


def test_rejects_modified_bridge_and_invalid_boundary():
    changed = decision()
    changed["poses"][2, 0, 3] += 0.01
    with pytest.raises(AssertionError):
        verify_selected_reference(changed, *references())
    wrong_time = decision()
    wrong_time["absolute_first_endpoint"] += 1
    with pytest.raises(ValueError, match="boundary"):
        verify_selected_reference(wrong_time, *references())
