"""Restoration uses the same available references and both prior branches."""

import subprocess
import sys

import numpy as np
import pytest

from scripts.simulation.runtime import run_replan_quality_feedback as run


def test_checker_import_does_not_load_candidate_modules():
    code = (
        "import sys; from scripts.simulation.runtime import run_replan_quality_feedback as x; "
        "assert x.RESTORE_MODULE not in sys.modules; "
        "assert x.FULL_MODULE not in sys.modules; "
        "assert x.base.quality.QUALITY_MODULE not in sys.modules; "
        "assert x.base.quality.baseline.POLICY_MODULE not in sys.modules; "
        "assert x.base.quality.baseline.PREVIEW_MODULE not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)


def test_restoration_retains_both_replan_cases_and_full_comparisons():
    inputs = run.load_restoration_inputs()
    assert [case["spec"]["case_index"] for case in inputs["cases"]] == [2, 6]
    for ref in (run.FULL_PRODUCER, run.FULL_CHECKER):
        assert inputs["source_sha256"][ref["path"]] == ref["sha256"]
    for case in inputs["cases"]:
        selected = run.switch.selected_case(case, "new")
        assert selected["completed_history"].completed_periods == selected["spec"]["prefix_steps"] == 16
        assert selected["intervention_spec"]["cursor"] == 0
        assert len(selected["commands"]) == 16 and len(selected["saved"]["sim_states"]) == 17
        np.testing.assert_array_equal(selected["fixed_targets"], case["reference_chunks"]["new"][:8])


def test_restoration_has_an_independent_namespace_and_cannot_mix_modes():
    assert run.branch_key(2) == "case_02_quality_new"
    assert run.branch_key(2, full=True) == "case_02_full_quality_new"
    assert run.branch_key(2, restore=True) == "case_02_restoration_quality_new"
    with pytest.raises(ValueError, match="Only one"):
        run.branch_key(2, full=True, restore=True)
