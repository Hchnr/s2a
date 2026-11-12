"""Detect aliased snapshot fields against independently copied box centers."""

from scripts.simulation.runtime.audit_bowl_pad_alignment_snapshot_failure import mismatch_summary


def test_mismatch_summary_does_not_treat_reused_coordinates_as_snapshots():
    def point(left, right, body):
        return dict(
            body_origin_world_m=[body, 0, 0],
            pad_alignment=dict(
                left_pad_world_m=[9, 0, 0],
                right_pad_world_m=[8, 0, 0],
                body_origin_world_m=[3, 0, 0],
                box_projections=[dict(center_world_m=[left, 0, 0]), dict(center_world_m=[right, 0, 0])],
            ),
        )

    result = mismatch_summary([point(1, 2, 1), point(9, 8, 3)])
    assert result["left_pad_world_m"]["mismatch_count"] == 1
    assert result["right_pad_world_m"]["mismatch_count"] == 1
    assert result["left_pad_world_m"]["all_equal_to_final"] is True
    assert result["body_origin_world_m"]["mismatch_count"] == 1
