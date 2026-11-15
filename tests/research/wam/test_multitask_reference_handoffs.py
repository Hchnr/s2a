"""Verify that handoffs compare the same physical endpoint and SO(3) orientation."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from scripts.tracking.response_models.audit_multitask_reference_handoffs import compare_handoff


def test_handoff_uses_old_consumed_index_and_new_first_endpoint():
    old = dict(
        start_step=10,
        consumed=2,
        boundary="quality_trigger",
        poses=np.tile(np.eye(4), (40, 1, 1)),
        openings=np.ones(40),
    )
    new = dict(start_step=12, poses=np.tile(np.eye(4), (40, 1, 1)), openings=np.ones(40))
    old["poses"][2, 0, 3] = 0.03
    old["poses"][2, :3, :3] = Rotation.from_euler("z", 0.2).as_matrix()
    old["openings"][2], new["openings"][0] = 0.1, 0.9
    result = compare_handoff(old, new)
    assert result["common_endpoint_step"] == 13
    assert result["position_difference_m"] == pytest.approx(0.03)
    assert result["rotation_difference_rad"] == pytest.approx(0.2)
    assert result["old_closed_new_open"] is True
    new["start_step"] = 13
    with pytest.raises(ValueError, match="share a future endpoint"):
        compare_handoff(old, new)
