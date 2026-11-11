"""Multiple contact points must count as one contact sample per inner step."""

from scripts.verification.reference_ports import correct_replayed_contact_pair_counts as correction


def test_multiple_points_for_one_pair_count_once_per_inner_step():
    contact = {
        "kind": "robot_environment",
        "active_constraint": True,
        "geom1": "finger",
        "geom2": "object",
        "wrench_contact_frame": [2, 0, 0, 0, 0, 0],
    }
    samples = [{"step": 80, "inner_index": i, "contacts": [contact] * 4} for i in range(25)]
    pair = correction.summarize_pairs(samples)[0]["contact_pairs"][0]
    assert pair["contact_samples"] == 25 and pair["contact_points"] == 100
    assert pair["maximum_force_n"] == 2.0


def test_correction_binds_the_original_audit_and_all_saved_pairs():
    result = correction.audit()
    assert result["legacy_pair_samples_actually_counted_contact_points"] is True
    assert result["branch_periods"]["historical"]["pair_periods_with_multiple_points"] == 29
    assert result["branch_periods"]["innovation"]["pair_periods_with_multiple_points"] == 29
    for name in ("historical", "innovation"):
        row = next(item for item in result["branch_periods"][name]["periods"] if item["step"] == 122)
        table = next(item for item in row["contact_pairs"] if "table_collision" in item["geometries"])
        assert table["contact_samples"] == 25
