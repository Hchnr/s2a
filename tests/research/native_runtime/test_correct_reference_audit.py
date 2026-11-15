"""Historical comparison must reject a shifted target, including the tail."""

import numpy as np
import pytest

from s2a.references.oracle_timeline import OracleTimeline
from scripts.verification.reference_ports.audit_reset_correct_reference import audit_targets


def sample_timeline():
    poses = np.repeat(np.eye(4)[None, :, :], 48, axis=0)
    poses[:, 0, 3] = np.arange(48) * 0.001
    openings = np.linspace(0.1, 0.9, 48)
    windows = [
        dict(step=start, effective_targets_tcp=poses[start : start + 40], openings=openings[start : start + 40])
        for start in (0, 8)
    ]
    rows = [
        dict(step=i, desired_tcp=poses[i].copy(), gripper=dict(preview=dict(original_reference=openings[i])))
        for i in range(9)
    ]
    return OracleTimeline(windows, periods=9), rows


@pytest.mark.parametrize("field", ["tail_pose", "opening"])
def test_target_audit_rejects_mismatched_original_target(field):
    timeline, rows = sample_timeline()
    assert audit_targets(timeline, rows)["tcp_max_difference"] == 0
    if field == "tail_pose":
        rows[-1]["desired_tcp"][0, 3] += 0.001
    else:
        rows[0]["gripper"]["preview"]["original_reference"] += 0.001
    with pytest.raises(ValueError, match="timelines differ"):
        audit_targets(timeline, rows)
