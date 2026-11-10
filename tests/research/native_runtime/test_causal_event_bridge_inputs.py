"""The registered bridge must use only the verified previously generated chunks."""

import copy
import os
import subprocess
import sys

import numpy as np
import pytest

from scripts.simulation.runtime import prepare_causal_event_bridge_feedback as bridge


def test_real_inputs_bind_replayed_stop_and_same_time_reference_pair():
    inputs = bridge.load_inputs()
    case, config = inputs["case"], inputs["config"]
    assert case["spec"]["case_id"] == "task1_init0_s42_terminal058"
    assert set(inputs["direct_branches"]) == {"producer", "checker"}
    old, latest = bridge.selected_references(case)
    result = bridge.prepare_reference(case, config)
    audit = bridge.verify_reference(case, config, result)
    assert audit["passed"] and audit["checked_endpoints"] == 40
    np.testing.assert_array_equal(result["poses"][:8], old["fixed_targets"][:8])
    np.testing.assert_array_equal(result["openings"][:8], old["fixed_openings"][:8])
    np.testing.assert_array_equal(result["poses"][15:], latest["targets"][15:])
    np.testing.assert_array_equal(result["openings"][15:], latest["openings"][15:])
    changed = copy.deepcopy(result)
    changed["openings"][8] += 0.001
    with pytest.raises(ValueError):
        bridge.verify_reference(case, config, changed)


def test_checker_side_of_input_binding_does_not_import_candidate_or_producer():
    program = (
        "import sys; from scripts.simulation.runtime import prepare_causal_event_bridge_feedback as b; "
        "b.switch.base.checker_action([]); "
        "assert all(name not in sys.modules for name in b.switch.base.FORBIDDEN_CHECKER_MODULES); "
        "assert 's2a.research.wam.tracking.preview.causal_reference_bridge' not in sys.modules"
    )
    subprocess.run(
        [sys.executable, "-c", program],
        check=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "LD_LIBRARY_PATH": ""},
        capture_output=True,
    )
