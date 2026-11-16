"""The 24-case H40 audit retains measured labels and fixed source identities."""

import copy
import json

import numpy as np
import pytest

from s2a.research.wam.training.broader_recovery_h40 import clipping_flags
from s2a.research.wam.training.recovery_h40_contract import audit_window
from scripts.simulation.runtime import prepare_reset_correct_reference_cohort as preparation
from scripts.verification.wam.training import audit_broader_recovery_h40 as run


def test_fixed_config_rejects_changed_roster_normalization_and_authority():
    config = json.loads(run.CONFIG.read_text())
    run.validate_config(config)
    for change in (
        dict(registered_cases=23),
        dict(window_label_rows=2881),
        dict(pointwise_exception_cases=[]),
        dict(training_authorized=True),
        dict(fresh_cases=1),
    ):
        with pytest.raises(ValueError, match="fixed H40 roster"):
            run.validate_config(dict(config, **change))
    changed = copy.deepcopy(config)
    changed["normalization"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="fixed H40 roster"):
        run.validate_config(changed)


def test_actual_training_source_rejects_forbidden_or_changed_identity():
    config = json.loads(run.CONFIG.read_text())
    prepared = json.loads(run.prefix.checked(preparation.configured_ref(config["training_preparation"])).read_text())
    audit = json.loads(run.prefix.checked(preparation.configured_ref(config["recovery_audit"])).read_text())
    case = audit["cases"][0]["case"]
    identity, anchor = run.training_source(case, prepared)
    assert identity == anchor["source_identity"] and identity["demo"] == "demo_40"
    altered = copy.deepcopy(prepared)
    altered["exclusion"]["forbidden_demo_keys"].append([identity["task_index"], identity["demo"]])
    with pytest.raises(ValueError, match="outside validation and fresh"):
        run.training_source(case, altered)
    altered_case = copy.deepcopy(case)
    altered_case["initial_state_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="outside validation and fresh"):
        run.training_source(altered_case, prepared)


def test_actual_window_rebuilds_labels_and_clipping_from_measured_endpoints():
    config = json.loads(run.CONFIG.read_text())
    prepared = json.loads(run.prefix.checked(preparation.configured_ref(config["training_preparation"])).read_text())
    normalization = json.loads(run.prefix.checked(preparation.configured_ref(config["normalization"])).read_text())
    audit = json.loads(run.prefix.checked(preparation.configured_ref(config["recovery_audit"])).read_text())
    case = audit["cases"][0]
    _, anchor = run.training_source(case["case"], prepared)
    producer = json.loads(run.prefix.checked(case["producer_ref"]).read_text())
    with np.load(run.prefix.checked(producer["arrays_ref"]), allow_pickle=False) as archive:
        data = {key: np.asarray(archive[key]).copy() for key in archive.files}
    result = audit_window(data, 0, np.asarray(anchor["tool_source_from_tcp"]), normalization)
    flags = clipping_flags(data["action128"][:40], normalization)
    np.testing.assert_array_equal(flags.sum(axis=0), result["clipped_per_active_slot"])
    np.testing.assert_array_equal(result["observed_openings"], data["opening"][1:41])
    np.testing.assert_array_equal(result["actual_gripper_commands"], data["sent_commands"][:40, 6])
    assert len(result["normalized_sha256"]) == 64


def test_state_pair_deduplication_does_not_merge_different_transitions():
    states = np.array([[0.0, 0.0], [1.0, 1.0], [0.0, 0.0], [1.0, 1.0]])
    assert len(run.state_pair_hashes(states, [0, 1, 2])) == 2
    assert run.command_counts([-1, 0, 1, 0]) == {"-1.0": 1, "0.0": 2, "1.0": 1}
