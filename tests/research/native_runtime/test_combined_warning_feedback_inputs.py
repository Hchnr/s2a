"""Check the selected reference and history on both reviewed development prefixes."""

import subprocess
import sys

import numpy as np
import pytest

from scripts.simulation.runtime import run_combined_warning_feedback as run


def test_checker_import_does_not_load_candidate_decision_modules():
    code = (
        "import sys; from scripts.simulation.runtime import run_combined_warning_feedback as x; "
        "assert x.quality.QUALITY_MODULE not in sys.modules; "
        "assert x.quality.baseline.POLICY_MODULE not in sys.modules; "
        "assert x.quality.baseline.PREVIEW_MODULE not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)


def test_real_inputs_end_at_first_warning_and_keep_only_completed_history():
    inputs = run.load_inputs()
    assert [item["spec"]["prefix_steps"] for item in inputs["cases"]] == [9, 9]
    for case in inputs["cases"]:
        assert case["first_warning"]["endpoint"] == 9
        assert case["first_warning"]["warning"]["reasons"] == ["position"]
        assert case["completed_history"].completed_periods == 9
        assert len(case["commands"]) == 9 and len(case["saved"]["sim_states"]) == 10
        assert case["intervention_spec"]["chunk_index"] == 1
        assert case["intervention_spec"]["cursor"] == 1
        np.testing.assert_array_equal(case["fixed_targets"], case["targets"][1:9])


def test_guard_stops_before_next_dispatch_after_any_observed_inner_buffer_loss():
    receipt = {"continuation_partial": {"periods": []}}
    calls = []

    def select(*args):
        calls.append(args)
        return {"command": np.zeros(7), "newly_selected": True}

    guarded = run.guarded_action(select, receipt)
    q, limits = np.zeros(7), np.tile([-1.0, 1.0], (7, 1))
    assert "command" in guarded(q, limits, None, None, 0)
    receipt["continuation_partial"]["periods"].append({"inner_below_buffer_samples": 1})
    result = guarded(q, limits, None, None, 1)
    assert result["stop"]["reason"] == "observed_joint_buffer_loss" and len(calls) == 1


def test_original_first_decision_must_match_before_dispatch():
    selected = {"command": np.zeros(7), "refined_plan": np.zeros((8, 6)), "route": "ok"}
    row = {"command": [0] * 6 + [-1], "tracking_refinement": {"planned_commands": np.zeros((8, 6)), "route": "ok"}}
    receipt = {"continuation_partial": {"periods": []}}
    guarded = run.guarded_action(
        lambda *_args: selected, receipt, first_decision_check=lambda s: run.first_decision_comparison(s, row)
    )
    q, limits = np.zeros(7), np.tile([-1.0, 1.0], (7, 1))
    guarded(q, limits, None, None, 0)
    assert receipt["first_decision_reconstruction"]["passed"] is True
    assert receipt["first_decision_reconstruction"]["original_gripper_command"] == -1
    assert receipt["first_decision_reconstruction"]["comparison_gripper_command"] == 0
    row["command"][0] = 0.1
    with pytest.raises(ValueError, match="first decision"):
        guarded(q, limits, None, None, 0)


def test_checker_confirms_buffer_stop_but_does_not_recompute_quality_stop():
    expected = {"reason": "observed_joint_buffer_loss", "at_step": 10, "next_action_dispatched": False}
    continuation = {"periods": [{"inner_below_buffer_samples": 2}], "stop": None}
    result = run.reconcile_checker_stop(continuation, expected, mode="original", start_step=9)
    assert result["buffer_stop_verified_from_replayed_inner_samples"] is True
    with pytest.raises(ValueError, match="observed loss"):
        run.reconcile_checker_stop(
            {"periods": [{"inner_below_buffer_samples": 0}], "stop": None}, expected, mode="original", start_step=9
        )
    expected = {"reason": "quality_no_witness", "at_step": 9, "next_action_dispatched": False}
    result = run.reconcile_checker_stop({"periods": [], "stop": None}, expected, mode="quality", start_step=9)
    assert result["decision_stop_recorded_not_recomputed"] is True
