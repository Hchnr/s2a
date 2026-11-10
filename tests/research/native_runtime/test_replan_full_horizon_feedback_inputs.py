"""Both cases remain in the registered full-horizon physical comparison."""

import subprocess
import sys

import numpy as np

from scripts.simulation.runtime import run_replan_quality_feedback as run


def test_checker_import_does_not_load_full_horizon_or_other_candidate_modules():
    code = (
        "import sys; from scripts.simulation.runtime import run_replan_quality_feedback as x; "
        "assert x.FULL_MODULE not in sys.modules; "
        "assert x.base.quality.QUALITY_MODULE not in sys.modules; "
        "assert x.base.quality.baseline.POLICY_MODULE not in sys.modules; "
        "assert x.base.quality.baseline.PREVIEW_MODULE not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)


def test_both_cases_retain_the_same_new_reference_without_future_feedback():
    inputs = run.load_full_inputs()
    assert [case["spec"]["case_index"] for case in inputs["cases"]] == [2, 6]
    for case in inputs["cases"]:
        selected = run.switch.selected_case(case, "new")
        assert selected["spec"]["prefix_steps"] == selected["completed_history"].completed_periods == 16
        assert selected["intervention_spec"]["cursor"] == 0
        assert len(selected["commands"]) == 16 and len(selected["saved"]["sim_states"]) == 17
        np.testing.assert_array_equal(selected["fixed_targets"], case["reference_chunks"]["new"][:8])


def test_explicit_full_branch_does_not_replace_the_original_branch():
    assert run.branch_key(2) == "case_02_quality_new"
    assert run.branch_key(2, full=True) == "case_02_full_quality_new"
