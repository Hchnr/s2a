"""The read-only audit must reject altered contact evidence."""

import copy

import pytest

from scripts.simulation.runtime.audit_reset_bowl_saved_contacts import check_contacts, summarize


def contact():
    return dict(
        left_pad=[
            dict(target_geom="akita_black_bowl_1_g1", other_geom="gripper0_finger1_pad_collision", distance_m=0.001)
        ],
        right_pad=[],
        other=[],
        bilateral=False,
    )


def test_audit_checks_category_geometry_and_bilateral_predicate():
    value = contact()
    check_contacts(value)
    for key, replacement in (
        ("target_geom", "akita_black_bowl_1_g0"),
        ("other_geom", "gripper0_finger2_pad_collision"),
        ("distance_m", float("inf")),
    ):
        changed = copy.deepcopy(value)
        changed["left_pad"][0][key] = replacement
        with pytest.raises(ValueError):
            check_contacts(changed)
    changed = copy.deepcopy(value)
    changed["bilateral"] = True
    with pytest.raises(ValueError, match="bilateral"):
        check_contacts(changed)


def test_audit_reports_original_label_mismatch_and_other_geometry():
    rows = [dict(endpoint_step=i, target_grasp=(i == 7), contacts=contact()) for i in range(1, 49)]
    rows[0]["contacts"]["other"].append(
        dict(target_geom="akita_black_bowl_1_g2", other_geom="table", distance_m=-0.001)
    )
    summary, others = summarize(rows)
    assert summary["grasp_label_mismatch_steps"] == [7]
    assert summary["endpoint_counts"] == dict(left_pad=48, right_pad=0, bilateral=0, other=1)
    assert others["table"]["endpoint_steps"] == [1]
