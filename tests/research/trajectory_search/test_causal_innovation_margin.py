"""The saved-state comparison must independently enforce its tighter margin."""

import numpy as np
import pytest

from scripts.tracking.trajectory_search.run_causal_innovation_margin import check_margin


def test_audited_candidate_must_stay_inside_the_registered_corrected_margin():
    row = {
        "witness_found": True,
        "certified_model_witness": True,
        "independent_verification": {
            "passed": True,
            "first_pose_caps_satisfied": True,
            "first_pose_caps_enforced": True,
            "first_tcp_error": np.zeros(6),
        },
        "independent_corrected_error": np.array([0.006, 0, 0, 0, 0, 0]),
        "innovation_corrected_first_tcp_error": np.array([0.006, 0, 0, 0, 0, 0]),
    }
    assert check_margin(row, np.array([0.006, 0, 0]), np.eye(3)) is True
    row["independent_corrected_error"][0] = row["innovation_corrected_first_tcp_error"][0] = 0.009
    with pytest.raises(ValueError, match="exceeds the fixed margin"):
        check_margin(row, np.array([0.009, 0, 0]), np.eye(3))


def test_failed_or_changed_independent_audit_cannot_authorize_a_candidate():
    assert check_margin({"witness_found": False}, np.zeros(3), np.eye(3)) is False
    with pytest.raises(ValueError, match="independent quality and safety audit"):
        check_margin({"witness_found": True, "independent_verification": {"passed": False}}, np.zeros(3), np.eye(3))
    with pytest.raises(AssertionError):
        check_margin(
            {
                "witness_found": True,
                "certified_model_witness": True,
                "independent_verification": {
                    "passed": True,
                    "first_pose_caps_satisfied": True,
                    "first_pose_caps_enforced": True,
                    "first_tcp_error": np.zeros(6),
                },
                "independent_corrected_error": np.zeros(6),
                "innovation_corrected_first_tcp_error": np.ones(6) * 0.001,
            },
            np.zeros(3),
            np.eye(3),
        )
