"""Independent progress profiles cannot relax other contracts or numerical sources."""

from copy import deepcopy

import pytest

from scripts.tracking.trajectory_search.cap_distance_search_audit import audit_cap_search
from scripts.tracking.trajectory_search.check_small_progress_native_repair import ADDED, CHANGED, source_delta
from scripts.tracking.trajectory_search.objective_search_audit import validate_progress_configuration
from tests.research.trajectory_search.test_cap_distance_audit import archive
from tests.research.trajectory_search.test_state_vector_gn_search_audit import vector_inputs


def configs():
    _, args, _ = vector_inputs(curvature=20.0)
    original = deepcopy(args[-1])
    current = deepcopy(original)
    current.update(minimum_predicted_reduction=1e-5, minimum_actual_reduction=1e-5)
    return current, original


def test_independent_full_ledger_accepts_only_explicit_two_field_profile():
    evaluate, args, _ = vector_inputs(curvature=20.0)
    original = deepcopy(args[-1])
    current = deepcopy(original)
    current.update(minimum_predicted_reduction=1e-5, minimum_actual_reduction=1e-5)
    modified = (*args[:-1], current)
    answer = archive(evaluate, modified)
    assert audit_cap_search(answer, *modified, progress_reference=original)["passed"]
    with pytest.raises(ValueError):
        audit_cap_search(answer, *modified)


@pytest.mark.parametrize(
    "fault", ["safety_buffer", "reserve_entry_fraction", "max_responses", "partial", "other_floor", "reference"]
)
def test_independent_profile_rejects_other_changes(fault):
    current, original = configs()
    if fault == "partial":
        current["minimum_actual_reduction"] = 1e-4
    elif fault == "other_floor":
        current["minimum_predicted_reduction"] = 1e-6
    elif fault == "reference":
        original["safety_buffer"] *= 2
    else:
        current[fault] *= 2
    with pytest.raises(ValueError):
        validate_progress_configuration(current, original)


def snapshots():
    old = dict(
        current_map_ref=dict(sha256="map"),
        driver_ref=dict(sha256="driver"),
        sources={"src/s2a/controllers/osc.py": "same", sorted(CHANGED)[0]: "old"},
    )
    new = deepcopy(old)
    new["sources"][sorted(CHANGED)[0]] = "new"
    new["sources"][sorted(ADDED)[0]] = "added"
    return old, new


def test_source_delta_allows_only_registered_audit_repair_and_additions():
    old, new = snapshots()
    r = source_delta(old, new)
    assert r["numerical_sources_unchanged"] and len(r["changed"]) == len(r["added"]) == 1


@pytest.mark.parametrize("fault", ["numerical", "deleted", "added", "map", "driver"])
def test_original_numerical_or_unregistered_sources_cannot_drift(fault):
    old, new = snapshots()
    if fault == "numerical":
        new["sources"]["src/s2a/controllers/osc.py"] = "changed"
    elif fault == "deleted":
        del new["sources"]["src/s2a/controllers/osc.py"]
    elif fault == "added":
        new["sources"]["src/s2a/unregistered.py"] = "new"
    elif fault == "map":
        new["current_map_ref"]["sha256"] = "changed"
    else:
        new["driver_ref"]["sha256"] = "changed"
    with pytest.raises(ValueError):
        source_delta(old, new)
