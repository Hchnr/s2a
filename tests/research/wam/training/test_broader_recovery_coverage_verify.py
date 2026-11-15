"""Independent scalar checks detect mismatched state-neighbor claims."""

import copy

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from scripts.verification.wam.training import audit_broader_recovery_coverage as primary
from scripts.verification.wam.training import verify_broader_recovery_coverage as run


def anchor(name, position, angle, opening, *, task=0):
    pose = np.eye(4)
    pose[0, 3] = position
    pose[:3, :3] = Rotation.from_euler("z", angle).as_matrix()
    return dict(anchor_id=name, source_kind="training", task_index=task, tcp_pose=pose, opening=opening)


def test_scalar_joint_checks_one_task_matched_anchor_in_order():
    query = dict(task_index=0, tcp_pose=np.eye(4), opening=0.5)
    anchors = [
        anchor("wrong_task", 0.0, 0.0, 0.5, task=1),
        anchor("wrong_rotation", 0.01, 0.6, 0.5),
        anchor("first_valid", 0.04, 0.3, 0.6),
        anchor("closer_valid", 0.01, 0.1, 0.5),
    ]
    result = run.scalar_joint(query, anchors, max_position_m=0.05, max_rotation_rad=0.5, max_opening_abs=0.2)
    assert result["anchor_id"] == "first_valid"
    assert result["position_m"] == pytest.approx(0.04)
    assert result["rotation_rad"] == pytest.approx(0.3)
    assert result["opening_abs"] == pytest.approx(0.1)
    assert run.scalar_joint(query, anchors[:2], max_position_m=0.05, max_rotation_rad=0.5, max_opening_abs=0.2) is None


def test_match_comparison_detects_wrong_id_missing_match_and_wrong_metrics():
    claim = dict(anchor_id="a", source_kind="training", position_m=0.02, rotation_rad=0.3, opening_abs=0.1)
    assert max(run.compare_match(claim, copy.deepcopy(claim)).values()) == 0
    for bad in (None, dict(claim, anchor_id="b"), dict(claim, position_m=0.021)):
        with pytest.raises(ValueError):
            run.compare_match(claim, bad)
    assert max(run.compare_match(None, None).values()) == 0


@pytest.mark.parametrize("fraction", [0.0, 0.5, 0.9, 1.0])
def test_scalar_linear_quantile_matches_reference(fraction):
    values = [0.2, 1.5, 0.1, 0.4, 0.3, 3.0]
    assert run.scalar_quantile(values, fraction) == pytest.approx(np.percentile(values, 100 * fraction))
    with pytest.raises(ValueError, match="Finite values"):
        run.scalar_quantile([], fraction)


def test_summary_checker_rejects_changed_joint_count():
    def row(position, matched):
        closest = dict(anchor_id="a", source_kind="training", position_m=position, rotation_rad=0.2, opening_abs=0.1)
        group = dict(nearest=closest, joint={"0.05": closest if matched else None, "0.1": closest})
        return dict(previous=group, all=group, sensitivity=group)

    rows = [row(0.01, True), row(0.06, False)]
    claimed = primary.summarize(rows, "all")
    run.compare_summary(rows, claimed, "all")
    corrupted = copy.deepcopy(claimed)
    corrupted["joint_counts"]["0.05"] += 1
    with pytest.raises(ValueError, match="summary differs"):
        run.compare_summary(rows, corrupted, "all")
