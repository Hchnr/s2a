"""Event proposals remain offline and validation identities remain disjoint."""

import copy
import json

import numpy as np
import pytest

from s2a.research.wam.training.clean.clean_training_data import VALIDATION_DEMO_KEYS
from s2a.research.wam.training.recovery_event_coverage import event_coverage, validation_identity_check
from scripts.verification.wam.training import audit_recovery_event_coverage as run


def metadata():
    groups = [dict(task_index=i, demo=f"demo_{i + 100}", initial_state_sha256=f"{i + 1:064x}") for i in range(3)]
    fresh = [dict(task_index=i % 3, demo=f"demo_{i + 200}", initial_state_sha256=f"{i + 10:064x}") for i in range(12)]
    records = [
        dict(task_index=task, demo=demo, initial_state_sha256=f"{i + 100:064x}")
        for i, (task, demo) in enumerate(sorted(VALIDATION_DEMO_KEYS))
    ]
    preflight = dict(complete=True, passed=True, training_fresh_identity_disjoint=True, records=records)
    prepared = dict(records=copy.deepcopy(records))
    verification = dict(
        complete=True,
        passed=True,
        eligible_for_model_validation=True,
        fresh_authorized=False,
        native_stable_cases=6,
        physics_endpoints_verified=1218,
        current_images_verified=24,
        cases=[
            dict(name=f"task{row['task_index']}_validation_{row['demo']}", native_stable_last40=True) for row in records
        ],
    )
    return groups, dict(exclusion=dict(fresh_sources=fresh)), preflight, prepared, verification


def test_existing_windows_and_event_candidate_use_actual_future_only_offline():
    opening = np.full(141, 0.8)
    success = np.zeros(141, dtype=bool)
    opening[106], opening[107], opening[108] = 0.34, 0.2, 0.35
    success[131:] = True
    result = event_coverage(opening, success)
    assert result["periods"] == 140 and result["covered_case_transitions"] == 96
    assert result["low_opening_total"] == 2 and result["low_opening_covered"] == 0
    assert result["successful_endpoints_total"] == 10 and result["successful_endpoints_covered"] == 0
    candidate = result["selected_anchor"]
    assert candidate["event_action_index"] == 105 and candidate["event_endpoint_index"] == 106
    assert candidate["anchor_step"] == 98 and candidate["event_lead_periods"] == 8
    assert candidate["selection_uses_future_training_labels"] is True and not candidate["current_image_available"]
    assert event_coverage(np.full(141, 0.8), success)["selected_anchor"] is None


def test_near_terminal_event_reserves_a_full_horizon_without_faking_lead():
    opening = np.full(141, 0.8)
    opening[136] = 0.1
    candidate = event_coverage(opening, np.zeros(141, dtype=bool))["selected_anchor"]
    assert candidate["anchor_step"] == 100 and candidate["event_lead_periods"] == 36
    with pytest.raises(ValueError, match="boolean"):
        event_coverage(opening, np.zeros(141, dtype=int))


def test_synchronized_validation_checks_demo_and_initial_state_identity():
    args = metadata()
    result = validation_identity_check(*args)
    assert result["identity_sets_disjoint"] and result["synchronized_validation_sources"] == 6
    assert not result["unknown_near_duplicates_excluded"] and not result["validation_future_labels_read"]
    bad = copy.deepcopy(args)
    for records in (bad[2]["records"], bad[3]["records"]):
        records[0]["initial_state_sha256"] = bad[0][0]["initial_state_sha256"]
    with pytest.raises(ValueError, match="identities differ"):
        validation_identity_check(*bad)
    bad = copy.deepcopy(args)
    bad[4]["cases"][0]["native_stable_last40"] = False
    with pytest.raises(ValueError, match="identities differ"):
        validation_identity_check(*bad)


def test_event_audit_config_never_authorizes_model_or_fresh():
    config = json.loads(run.CONFIG.read_text())
    run.validate_config(config)
    for change in (dict(opening_below=0.4), dict(new_current_images=1), dict(fresh_cases=1)):
        with pytest.raises(ValueError, match="zero-execution"):
            run.validate_config(dict(config, **change))
