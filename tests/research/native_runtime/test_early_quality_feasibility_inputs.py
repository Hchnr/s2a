"""The quality audit has exactly the registered development-state matrix."""

import pytest

from scripts.tracking.trajectory_search import audit_early_quality_feasibility as quality


def test_exact_two_cases_and_eight_offsets_are_registered():
    assert len(quality.BRANCHES) == 16
    assert quality.branch_key(2, 0) == "case_02_offset_00"
    assert quality.branch_key(6, 7) == "case_06_offset_07"
    for case, offset in ((0, 0), (2, 8), (6, -1)):
        with pytest.raises(ValueError, match="Fixed saved"):
            quality.branch_key(case, offset)
