"""Check frozen candidate identity and supported opening application."""

import copy

import numpy as np
import pytest

from scripts.wam.training.native import supported_opening_event_validation as validation
from tests.research.gripper.test_causal_opening_event_metrics import config


def candidate_inputs():
    selection, manifest, registration = {"id": "selection"}, {"id": "manifest"}, {"id": "registration"}
    result = dict(
        complete=True,
        passed=True,
        full_fits=1,
        validation_future_reads=0,
        candidate_ref={"id": "candidate"},
        registration_ref=registration,
    )
    candidate = dict(
        schema="s2a.supported-opening-event-candidate.v1",
        complete=True,
        registration_ref=registration,
        selection_ref=selection,
        manifest_ref=manifest,
        threshold=0.7,
        support_future_steps=16,
        closed_below=0.35,
        commanded_opening=0.1,
        training_demos=30,
        training_queries=300,
        validation_future_reads=0,
    )
    return candidate, result, selection, manifest


def test_candidate_is_bound_to_one_fit_and_the_frozen_threshold():
    candidate, result, selection, manifest = candidate_inputs()
    validation.admit_candidate(candidate, result, selection, manifest, 0.7)
    for key, value in (("threshold", 0.8), ("support_future_steps", 17), ("validation_future_reads", 1)):
        changed = copy.deepcopy(candidate)
        changed[key] = value
        with pytest.raises(ValueError, match="frozen"):
            validation.admit_candidate(changed, result, selection, manifest, 0.7)
    result["full_fits"] = 2
    with pytest.raises(ValueError, match="frozen"):
        validation.admit_candidate(candidate, result, selection, manifest, 0.7)


def test_candidate_changes_only_openings_with_later_generated_support():
    raw = np.full((2, 40), 0.8)
    raw[0, 16] = 0.2
    probability = np.full((2, 40), 0.9)
    changed = validation.apply_supported(raw, probability, 0.7, config())
    assert changed[0, 0] == 0.1 and changed[0, 15] == 0.1
    assert changed[0, 16] == raw[0, 16] and changed[0, 17] == raw[0, 17]
    np.testing.assert_array_equal(changed[1], raw[1])
