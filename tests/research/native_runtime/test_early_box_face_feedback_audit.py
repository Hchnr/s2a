"""Audit classifies active contact pairs per sample, not per contact point."""

from scripts.verification.response_models.audit_early_box_face_feedback import contact_pair_counts


def test_pair_counts_distinguish_bowl_from_table_and_deduplicate():
    bowl = dict(active_constraint=True, kind="robot_environment", geom1="finger1", geom2="bowl")
    table = dict(active_constraint=True, kind="robot_environment", geom1="finger2", geom2="table")
    inactive = {**table, "active_constraint": False}
    samples = [{"contacts": [bowl, bowl]}, {"contacts": [bowl, inactive]}, {"contacts": [table]}]
    assert contact_pair_counts(samples) == [
        {"geometries": ["bowl", "finger1"], "active_inner_samples": 2},
        {"geometries": ["finger2", "table"], "active_inner_samples": 1},
    ]
