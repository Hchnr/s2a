"""The independent scalar/SciPy check must reject wrong neighbor evidence."""

import copy

import numpy as np
import pytest

from s2a.research.wam.training.state_anchor_coverage import nearest_position_anchor
from scripts.verification.wam.training.verify_wam_recovery_state_coverage import compare_row, scalar_nearest
from tests.research.wam.training.test_state_anchor_coverage import anchor


def test_independent_scalar_and_scipy_match_position_selected_neighbor():
    query = dict(task_index=0, tcp_pose=np.eye(4), opening=0.7)
    anchors = [anchor("first", 0.2, np.pi, 0.1), anchor("closer", 0.1, 0.3, 0.6, kind="recovery")]
    claimed = nearest_position_anchor(query, anchors)
    differences = compare_row(query, anchors, claimed)
    assert max(differences.values()) < 1e-12 and scalar_nearest(query, anchors)["anchor_id"] == "closer"
    changed = copy.deepcopy(claimed)
    changed["anchor_id"] = "first"
    with pytest.raises(ValueError, match="selected same-task"):
        compare_row(query, anchors, changed)
    changed = copy.deepcopy(claimed)
    changed["rotation_rad"] += 0.01
    with pytest.raises(ValueError, match="metrics differ"):
        compare_row(query, anchors, changed)


def test_independent_search_keeps_first_candidate_on_exact_tie_and_checks_pi():
    query = dict(task_index=0, tcp_pose=np.eye(4), opening=0.7)
    values = [anchor("baseline", 0.1, np.pi, 0.1), anchor("recovery", 0.1, 0.0, 0.7, kind="recovery")]
    result = scalar_nearest(query, values)
    assert result["anchor_id"] == "baseline" and result["rotation_rad"] == pytest.approx(np.pi)
    with pytest.raises(ValueError, match="same-task"):
        scalar_nearest(dict(query, task_index=2), values)
