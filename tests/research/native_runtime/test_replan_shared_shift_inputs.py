"""Cross-task inputs preserve prior replays and hide scoring-only data."""

import copy
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.simulation.runtime import prepare_replan_shared_shift as binding
from scripts.simulation.runtime import run_replan_reference_switch as switch


def test_registered_config_and_prior_inputs_are_exact():
    inputs = binding.load_inputs()
    assert [case["spec"]["case_index"] for case in inputs["cases"]] == [2, 6]
    receipt = binding.input_receipt(inputs)
    assert receipt["restoration_refs"] == binding.RESTORATION and receipt["task2_refs"] == binding.TASK2
    assert len(receipt["legacy_source_changes"]) == 4
    assert json.loads(json.dumps(receipt)) == receipt
    assert inputs["replan_shared_config"]["quality_response_budget"] == 320


def test_policy_case_excludes_scoring_and_future_recorded_states():
    inputs = binding.load_inputs()
    for case in inputs["cases"]:
        policy = binding.policy_case(case)
        selected = switch.selected_case(case, "new")
        assert set(policy) == {
            "spec",
            "intervention_spec",
            "targets",
            "anchor",
            "tool",
            "controller",
            "previous_evidence",
            "completed_history",
        }
        assert policy["spec"] == {"prefix_steps": 16} and policy["intervention_spec"] == {"cursor": 0}
        np.testing.assert_array_equal(policy["targets"], selected["targets"])
        assert not {"scoring_targets", "fixed_targets", "reference_chunks", "saved", "record", "commands"} & set(policy)


def test_wrong_completed_history_and_prefix_are_rejected():
    inputs = binding.load_inputs()
    wrong = copy.deepcopy(inputs["cases"][0])
    wrong["spec"]["prefix_steps"] = 15
    with pytest.raises(ValueError, match="Causal step-16"):
        binding.policy_case(wrong)
    wrong = copy.deepcopy(inputs["cases"][0])
    wrong["completed_history"] = SimpleNamespace(
        completed_periods=16, last_plan=np.zeros((7, 6)), previous_command=np.zeros(7)
    )
    with pytest.raises(ValueError, match="Causal step-16"):
        binding.policy_case(wrong)


def test_unregistered_legacy_source_drift_is_rejected():
    path = str(binding.REPO / "src/unregistered.py")
    before = [{"path": path, "sha256": "a" * 64}]
    after = [{"path": path, "sha256": "b" * 64}]
    with pytest.raises(ValueError, match="explicitly recorded"):
        binding.validate_legacy_sources(before, after)


def test_clean_checker_input_binding_does_not_import_candidate_modules():
    code = (
        "import sys; from scripts.simulation.runtime import prepare_replan_shared_shift as b; "
        "b.load_inputs(); b.source_refs(); "
        "assert all(name not in sys.modules for name in b.task2.CHECKER_FORBIDDEN)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=binding.REPO,
        env=os.environ.copy(),
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
