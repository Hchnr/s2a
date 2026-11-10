"""Bind the two new interventions to the existing physically replayed baselines."""

import copy
import os
import subprocess
import sys

import pytest

from scripts.simulation.runtime import bind_causal_event_bridge_factorial as binding


def test_real_input_binding_and_independent_component_reconstruction():
    inputs = binding.load_inputs()
    assert inputs["case"]["spec"]["case_id"] == "task1_init0_s42_terminal058"
    assert set(inputs["old_baselines"]) == set(inputs["bridge_baselines"]) == {"producer", "checker"}
    for mode in binding.MODES:
        result = binding.construct_reference(inputs, mode)
        assert binding.verify_reference(inputs, mode, result)["checked_endpoints"] == 40
        changed = copy.deepcopy(result)
        changed["openings"][8] += 0.001
        with pytest.raises(ValueError):
            binding.verify_reference(inputs, mode, changed)


def test_input_binding_does_not_import_the_producers_or_candidate_policies():
    program = (
        "import sys; from scripts.simulation.runtime import bind_causal_event_bridge_factorial as b; "
        "b.bridge.require_checker_independence(); "
        "assert 'scripts.simulation.runtime.construct_causal_reference_factorial' not in sys.modules"
    )
    subprocess.run(
        [sys.executable, "-c", program],
        check=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "LD_LIBRARY_PATH": ""},
        capture_output=True,
    )
