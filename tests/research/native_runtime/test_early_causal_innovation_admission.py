"""The step122 search may use only completed step121 feedback."""

import copy

import numpy as np
import pytest

from scripts.simulation.runtime import prepare_quality_rate_repair_recorded_feedback as physical
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.tracking.trajectory_search import early_causal_innovation_admission as admission


def test_actual_completed_state_and_reference_are_bound():
    inputs = admission.load_inputs()
    stopped = inputs["stop_inputs"]
    assert set(stopped) == {
        "state",
        "previous",
        "shifted_seed",
        "past_inner_error_rad",
        "translation_innovation",
        "rotation_discrepancy",
    }
    assert stopped["state"].shape == (79,) and stopped["shifted_seed"].shape == (8, 6)
    assert stopped["past_inner_error_rad"].shape == (25, 7)
    assert inputs["config"]["reference_cursor"] == 2 and inputs["targets"].shape == (40, 4, 4)


def test_future_step122_outcome_does_not_change_completed_inputs():
    config = admission.FIXED
    producer, checker = prefix.read(admission.PRODUCER), prefix.read(admission.CHECKER)
    audit = prefix.read(
        {"path": str(admission.ROOT / config["innovation_audit"]), "sha256": config["innovation_audit_sha256"]}
    )
    original = admission.select_completed(producer, checker, audit)
    modified = copy.deepcopy(producer)
    future = next(row for row in physical.periods(modified) if row["step"] == 122)
    future["actual_state"][1] += 1.0
    future["task_success"] = not future["task_success"]
    altered = admission.select_completed(modified, checker, audit)
    for key in original:
        np.testing.assert_array_equal(original[key], altered[key])


def test_changed_completed_state_is_rejected_by_replay():
    config = admission.FIXED
    producer, checker = prefix.read(admission.PRODUCER), prefix.read(admission.CHECKER)
    audit = prefix.read(
        {"path": str(admission.ROOT / config["innovation_audit"]), "sha256": config["innovation_audit_sha256"]}
    )
    changed = copy.deepcopy(producer)
    completed = next(row for row in physical.periods(changed) if row["step"] == 121)
    completed["actual_state"][1] += 0.1
    with pytest.raises(AssertionError):
        admission.select_completed(changed, checker, audit)
