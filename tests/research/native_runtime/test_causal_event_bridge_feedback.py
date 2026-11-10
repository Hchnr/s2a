"""The bridge execution must preserve causal inputs and independent checking."""

import copy
import os
import subprocess
import sys

import pytest

from scripts.simulation.runtime import run_causal_event_bridge_feedback as bridge


def test_reference_document_rejects_changed_source_or_saved_transition():
    inputs = bridge.binding.load_inputs()
    document = bridge.reference_document(inputs)
    reference, audit = bridge.checked_reference(inputs, document)
    assert reference["origin_step"] == 360 and audit["checked_endpoints"] == 40
    changed = copy.deepcopy(document)
    changed["case_id"] = "another_case"
    with pytest.raises(ValueError):
        bridge.checked_reference(inputs, changed)
    changed = copy.deepcopy(document)
    changed["reference"]["poses"][8, 0, 3] += 0.001
    with pytest.raises(ValueError):
        bridge.checked_reference(inputs, changed)


def test_checker_entry_does_not_import_any_candidate_policy_or_bridge_producer():
    program = (
        "import sys; from scripts.simulation.runtime import run_causal_event_bridge_feedback as b; "
        "b.require_checker_independence(); "
        "assert b.PRODUCER_MODULE not in sys.modules"
    )
    subprocess.run(
        [sys.executable, "-c", program],
        check=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "LD_LIBRARY_PATH": ""},
        capture_output=True,
    )
