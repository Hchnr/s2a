"""The quality candidate inherits the matched current reference and past only."""

import subprocess
import sys

import numpy as np

from scripts.simulation.runtime import run_replan_quality_feedback as run


def test_checker_import_does_not_load_candidate_decision_modules():
    code = (
        "import sys; from scripts.simulation.runtime import run_replan_quality_feedback as x; "
        "assert x.base.quality.QUALITY_MODULE not in sys.modules; "
        "assert x.base.quality.baseline.POLICY_MODULE not in sys.modules; "
        "assert x.base.quality.baseline.PREVIEW_MODULE not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)


def test_quality_input_uses_same_current_new_reference_as_baseline():
    inputs = run.load_inputs()
    for case in inputs["cases"]:
        selected = run.switch.selected_case(case, "new")
        assert selected["spec"]["prefix_steps"] == selected["completed_history"].completed_periods == 16
        assert selected["intervention_spec"]["cursor"] == 0
        assert "reference_chunks" not in selected and "scoring_targets" not in selected
        np.testing.assert_array_equal(selected["fixed_targets"], case["reference_chunks"]["new"][:8])


def test_history_is_completed_before_current_actual_endpoint_scoring():
    events = []
    receipt = {"continuation_partial": {"periods": [{"step": 16, "tcp_error": np.zeros(6)}]}}
    case = {"scoring_targets": {name: np.tile(np.eye(4), (8, 1, 1)) for name in ("old", "new")}}

    def history(_env, _command, offset):
        events.append(("completed", offset))
        return {"periods": 1}

    def tcp_reader(_env, _case):
        events.append(("scored", 0))
        return np.eye(4)

    callback = run.after_quality_completed(history, receipt, case, tcp_reader=tcp_reader)
    result = callback(None, np.zeros(7), 0)
    assert events == [("completed", 0), ("scored", 0)]
    assert result["policy_history"]["periods"] == 1
    np.testing.assert_array_equal(result["reference_errors"]["new"], np.zeros(6))
