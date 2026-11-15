"""Coverage measures one current pose and a task-matched position neighbor."""

import copy
import json

import numpy as np
import pytest

from s2a.research.wam.training.state_anchor_coverage import nearest_position_anchor, summarize_distances
from scripts.verification.wam.training import audit_wam_recovery_state_coverage as run


def anchor(name, position, angle, opening, *, task=0, kind="baseline"):
    c, s = np.cos(angle), np.sin(angle)
    pose = np.eye(4)
    pose[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    pose[0, 3] = position
    return dict(
        anchor_id=name,
        source_kind=kind,
        source_identity=dict(task_index=task, demo=name),
        anchor_step=0,
        task_index=task,
        tcp_pose=pose,
        opening=opening,
    )


def test_neighbor_is_chosen_by_task_matched_position_with_rotation_and_opening_from_same_anchor():
    query = dict(task_index=0, tcp_pose=np.eye(4), opening=0.8)
    baseline = [anchor("old", 0.3, np.pi, 0.2), anchor("other-task", 0.0, 0, 0.8, task=1)]
    nearest = nearest_position_anchor(query, baseline)
    assert nearest["anchor_id"] == "old" and nearest["position_m"] == pytest.approx(0.3)
    assert nearest["rotation_rad"] == pytest.approx(np.pi) and nearest["opening_abs"] == pytest.approx(0.6)
    new = anchor("new", 0.1, np.pi / 2, 0.4, kind="recovery")
    selected = nearest_position_anchor(query, baseline + [new])
    assert selected["anchor_id"] == "new" and selected["rotation_rad"] == pytest.approx(np.pi / 2)
    assert selected["opening_abs"] == pytest.approx(0.4) and selected["position_m"] <= nearest["position_m"]


def test_ties_retain_first_baseline_and_descriptive_bins_are_not_qualification():
    query = dict(task_index=0, tcp_pose=np.eye(4), opening=0.8)
    old = anchor("old", 0.02, 0.1, 0.4)
    new = anchor("new", 0.02, 0.0, 0.8, kind="recovery")
    assert nearest_position_anchor(query, [old, new])["anchor_id"] == "old"
    stats = summarize_distances([0.0, 0.02, 0.1])
    assert stats["count"] == 3 and stats["median_m"] == pytest.approx(0.02)
    assert stats["at_most_m"] == {"0.01": 1, "0.02": 2, "0.05": 2, "0.1": 3}
    with pytest.raises(ValueError, match="same task"):
        nearest_position_anchor(dict(query, task_index=2), [old, new])


def test_fixed_coverage_roster_and_no_fresh_or_model_authority():
    config = json.loads(run.CONFIG.read_text())
    run.validate_config(config)
    for change in (dict(request_count=214), dict(training_authorized=True), dict(fresh_cases=1)):
        with pytest.raises(ValueError, match="roster and scope"):
            run.validate_config(dict(config, **change))
    bad = copy.deepcopy(config)
    bad["development_audit"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="source identity"):
        run.validate_config(bad)
