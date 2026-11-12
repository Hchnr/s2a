"""Validate original versus shifted-reference accounting independently."""

import copy

import numpy as np
import pytest

from scripts.simulation.runtime.audit_bowl_h40_opening_lead import compare_openings


def traces():
    opening = np.linspace(1, 0, 40)
    baseline, candidate = [], []
    for i in range(40):
        for output, reference, actual in (
            (baseline, opening[i], opening[i] + 0.0),
            (candidate, opening[min(i + 4, 39)], opening[min(i + 4, 39)]),
        ):
            output.append(
                dict(
                    step=8 + i,
                    target_grasp=False,
                    task_success=False,
                    completed_feedback=dict(
                        score=dict(opening=actual, opening_reference=reference, opening_error=actual - reference)
                    ),
                )
            )
    return baseline, candidate, opening


def test_scores_modified_and_original_targets_separately():
    baseline, candidate, opening = traces()
    points, summary = compare_openings(baseline, candidate, opening)
    assert points[0]["lead_source_index"] == 4 and points[-1]["lead_source_index"] == 39
    assert summary["baseline_original"]["rmse"] == summary["candidate_lead"]["rmse"] == 0
    assert summary["candidate_original"]["rmse"] > 0


def test_detects_reference_or_stored_error_tampering():
    baseline, candidate, opening = traces()
    changed = copy.deepcopy(candidate)
    changed[0]["completed_feedback"]["score"]["opening_reference"] = opening[0]
    with pytest.raises(ValueError, match="independently indexed"):
        compare_openings(baseline, changed, opening)
    changed = copy.deepcopy(candidate)
    changed[5]["completed_feedback"]["score"]["opening_error"] = 0.9
    with pytest.raises(ValueError, match="stored own-reference errors"):
        compare_openings(baseline, changed, opening)
