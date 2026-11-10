"""Freeze intervention provenance and keep the replay free of candidate code."""

import copy
import os
import subprocess
import sys

import pytest

from scripts.simulation.runtime import run_causal_event_bridge_factorial as factorial


def test_real_reference_document_rejects_changed_component_or_provenance():
    inputs = factorial.binding.load_inputs()
    document = factorial.reference_document(inputs)
    references, audits = factorial.checked_references(inputs, document)
    assert set(references) == set(factorial.binding.MODES)
    assert all(value["checked_endpoints"] == 40 for value in audits.values())
    bad = copy.deepcopy(document)
    bad["case_id"] = "different_case"
    with pytest.raises(ValueError):
        factorial.checked_references(inputs, bad)
    bad = copy.deepcopy(document)
    bad["references"][factorial.binding.MODES[0]]["openings"][8] += 0.001
    with pytest.raises(ValueError):
        factorial.checked_references(inputs, bad)


def test_checker_does_not_import_mixed_reference_or_candidate_control_implementations():
    program = (
        "import sys; from scripts.simulation.runtime import run_causal_event_bridge_factorial as f; "
        "f.require_checker_independence(); "
        "assert f.CONSTRUCTOR_MODULE not in sys.modules"
    )
    subprocess.run(
        [sys.executable, "-c", program],
        check=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "LD_LIBRARY_PATH": ""},
        capture_output=True,
    )
