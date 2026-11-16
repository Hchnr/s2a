"""Event audit candidates must remain attached to measured source endpoints."""

import copy
import json

import numpy as np
import pytest

from scripts.simulation.runtime import prepare_reset_correct_reference_cohort as preparation
from scripts.verification.wam.training import audit_broader_recovery_events as run
from scripts.verification.wam.training.audit_recovery_h40_contract import Evidence, load_data


def test_event_config_rejects_changed_scope_or_source():
    config = json.loads(run.CONFIG.read_text())
    run.validate_config(config)
    for change in (
        dict(opening_below=0.4),
        dict(event_lead_periods=7),
        dict(fresh_cases=1),
        dict(training_authorized=True),
    ):
        with pytest.raises(ValueError, match="fixed event roster"):
            run.validate_config(dict(config, **change))
    changed = copy.deepcopy(config)
    changed["h40_audit"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="fixed event roster"):
        run.validate_config(changed)


def test_candidate_requires_measured_replayed_endpoint_and_does_not_invent_image():
    config = json.loads(run.CONFIG.read_text())
    recovery = json.loads(run.prefix.checked(preparation.configured_ref(config["recovery_audit"])).read_text())
    prepared = json.loads(run.prefix.checked(preparation.configured_ref(config["training_preparation"])).read_text())
    normalization = json.loads(run.prefix.checked(preparation.configured_ref(config["normalization"])).read_text())
    row = recovery["cases"][0]
    producer, checker = [
        json.loads(run.prefix.checked(row[key]).read_text()) for key in ("producer_ref", "checker_ref")
    ]
    identity, original = run.h40_source.training_source(row["case"], prepared)
    evidence = Evidence()
    data = load_data(evidence, producer["arrays_ref"])
    tool = np.asarray(original["tool_source_from_tcp"])
    candidate = dict(
        anchor_step=17,
        events=[],
        current_image_available=False,
        full_h40=True,
        selection_uses_future_training_labels=True,
    )
    checked = run.candidate_details(evidence, data, producer, checker, identity, tool, normalization, candidate)
    assert checked["current_endpoint"] == producer["endpoint_receipts"][17]
    assert checked["existing_image_ref"] is None and checked["normalized_shape"] == [40, 10]
    bad = dict(checker, endpoint_receipts=list(checker["endpoint_receipts"]))
    bad["endpoint_receipts"][17] = dict(bad["endpoint_receipts"][17], sim_state_sha256="wrong")
    with pytest.raises(ValueError, match="independently replayed endpoint"):
        run.candidate_details(evidence, data, producer, bad, identity, tool, normalization, candidate)
    bad_candidate = dict(candidate, current_image_available=True)
    with pytest.raises(ValueError, match="existing current image"):
        run.candidate_details(evidence, data, producer, checker, identity, tool, normalization, bad_candidate)
