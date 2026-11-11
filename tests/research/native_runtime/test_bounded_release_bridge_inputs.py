"""The bridge must begin from a completed hold and its independent replay."""

from scripts.simulation.runtime import prepare_bounded_release_bridge as binding
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix


def test_bridge_binds_the_completed_hold_and_independent_replay():
    inputs = binding.load_inputs()
    assert inputs["config"]["continuation_steps"] == 24
    assert inputs["case"]["spec"]["case_id"] == "task1_init0_s42_terminal058"
    assert set(inputs["prior_hold_branch_refs"]) == {"producer", "checker"}
    producer = prefix.read(inputs["prior_hold_branch_refs"]["producer"])
    checker = prefix.read(inputs["prior_hold_branch_refs"]["checker"])
    assert len(producer["runtime"]["continuation"]["periods"]) == 16
    assert len(checker["runtime"]["continuation"]["periods"]) == 16
    assert producer["runtime"]["release_selection"]["state_after_execution"]["mode"] == "held"
