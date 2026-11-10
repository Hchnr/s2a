"""Verify original third-task warning selection and causal prefix boundaries."""

from copy import deepcopy

import pytest

from scripts.simulation.runtime import replay_task2_warning_prefix as binding


@pytest.fixture(scope="module")
def inputs():
    return binding.load_inputs()


def test_original_task2_warnings_bind_only_completed_prefixes(inputs):
    assert [case["spec"]["case_index"] for case in inputs["cases"]] == [10, 11]
    assert [len(case["commands"]) for case in inputs["cases"]] == [34, 82]
    assert [len(case["saved"]["sim_states"]) for case in inputs["cases"]] == [35, 83]
    assert all(case["config"]["task_id"] == 6 and case["config"]["seed"] == 43 for case in inputs["cases"])
    assert all(
        case["warning"]["warning"]["alarm"] and case["warning"]["minimum_margin_rad"] >= 0.05
        for case in inputs["cases"]
    )


def test_wrong_warning_identity_is_rejected(inputs):
    case = inputs["cases"][0]
    spec = deepcopy(case["spec"])
    spec["case_id"] = "another-case"
    audit = binding.prefix.read(binding.WARNINGS)
    warning_case = binding.prefix.read(audit["case_refs"][10])
    with pytest.raises(ValueError, match="warning"):
        binding.check_warning(spec, warning_case, case["saved"])
