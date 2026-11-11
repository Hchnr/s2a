"""The normal-controller shadow keeps all original qualified cases."""

import pytest

from scripts.wam.tracking.preview.shadow_normal_correct_close_deferral import aggregate, load_inputs


def test_actual_ordered_matrix_and_frozen_sources_are_complete():
    inputs = load_inputs()
    assert len(inputs["cases"]) == 18
    assert sum(case["steps"] for case in inputs["cases"]) == 4907
    assert inputs["cases"][6]["case_id"] == "correct_06"
    assert "tracking_correct_reference_ordered_20261027" in inputs["cases"][6]["refs"]["result"]["path"]


def test_incomplete_case_denominator_is_rejected():
    with pytest.raises(ValueError, match="All original cases"):
        aggregate([])
