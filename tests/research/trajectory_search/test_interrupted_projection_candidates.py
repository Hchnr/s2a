"""Selection cannot conceal an interrupted ledger or change old numerical sources."""

import json
from copy import deepcopy

import pytest

from scripts.tracking.trajectory_search.check_interrupted_projection_candidates import (
    ADDED,
    CONFIG,
    SAFE_INDICES,
    registered_indices,
    source_delta,
)


def assessment():
    return dict(
        complete=True,
        all_records_self_consistent=True,
        source_unchanged=True,
        saved_production_responses=236,
        safe_buffered_count=13,
        original_result_and_ledger_missing=True,
        best_minimax_safe=[dict(index=110)],
        saved_responses=[dict(index=i, safe_buffered=i in SAFE_INDICES) for i in range(236)],
    )


def test_fixed_unsafe_all_safe_and_repeat():
    assert registered_indices(assessment(), json.loads(CONFIG.read_text())) == [0, *SAFE_INDICES, 110]


@pytest.mark.parametrize("fault", ["budget", "missing", "reorder", "unsafe", "extra", "minimax", "claimed_complete"])
def test_incomplete_or_changed_evidence_cannot_change_selection(fault):
    data, spec = assessment(), json.loads(CONFIG.read_text())
    if fault == "budget":
        spec["independent_response_budget"] += 1
    elif fault == "missing":
        data["saved_responses"].pop()
    elif fault == "reorder":
        data["saved_responses"][1]["index"] = 2
    elif fault == "unsafe":
        data["saved_responses"][73]["safe_buffered"] = False
    elif fault == "extra":
        data["saved_responses"][1]["safe_buffered"] = True
    elif fault == "minimax":
        data["best_minimax_safe"][0]["index"] = 73
    else:
        data["original_result_and_ledger_missing"] = False
    with pytest.raises(ValueError):
        registered_indices(data, spec)


def test_original_numerical_sources_and_expected_new_files_are_bound():
    old = dict(current_map_ref="map", driver_ref="driver", sources={"src/s2a/robot.py": "sha"})
    new = deepcopy(old)
    new["sources"].update({path: "new" for path in ADDED})
    assert source_delta(old, new)["original_sources_unchanged"]
    for fault in ("changed", "deleted", "extra", "map"):
        broken = deepcopy(new)
        if fault == "changed":
            broken["sources"]["src/s2a/robot.py"] = "changed"
        elif fault == "deleted":
            del broken["sources"]["src/s2a/robot.py"]
        elif fault == "extra":
            broken["sources"]["src/s2a/new.py"] = "new"
        else:
            broken["current_map_ref"] = "changed"
        with pytest.raises(ValueError):
            source_delta(old, broken)
