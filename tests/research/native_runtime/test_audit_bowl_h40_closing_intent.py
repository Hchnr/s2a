"""Record modified versus original references and endpoint limit excursions."""

import copy

import numpy as np
import pytest

from scripts.simulation.runtime.audit_bowl_h40_closing_intent import compare_openings, endpoint_checks


def test_distinguishes_original_and_modified_opening_and_partial_execution():
    model = np.linspace(1, 0.5, 40)

    def period(i, ref, actual):
        return dict(
            step=8 + i,
            completed_feedback=dict(score=dict(opening_reference=ref, opening=actual, opening_error=actual - ref)),
        )

    baseline = [period(i, model[i], model[i]) for i in range(40)]
    candidate = [period(i, np.clip(3 * model[min(i + 8, 39)] - 2, 0, 1), 0.4) for i in range(5)]
    points, summary = compare_openings(baseline, candidate, model)
    assert len(points) == 5 and points[0]["transformed_source_index"] == 8
    assert summary["baseline_original"]["rmse"] == 0
    assert summary["candidate_original"]["rmse"] != summary["candidate_modified"]["rmse"]
    bad = copy.deepcopy(candidate)
    bad[0]["completed_feedback"]["score"]["opening_reference"] = model[0]
    with pytest.raises(ValueError, match="independent index"):
        compare_openings(baseline, bad, model)


def test_reports_nominal_finger_range_and_pose_failure_without_inner_claim():
    periods = [
        dict(
            step=4,
            tcp_error=[0.011, 0, 0, 0, 0, 0],
            pose_caps_satisfied=False,
            completed_feedback=dict(score=dict(qpos=[-0.0004, -0.002])),
        )
    ]
    result = endpoint_checks(periods)
    assert result["pose_cap_failure_endpoints"][0]["endpoint_step"] == 5
    assert result["finger_nominal_range_outside_endpoints"][0]["endpoint_step"] == 5
    assert result["finger_inner_samples_recorded"] is False
