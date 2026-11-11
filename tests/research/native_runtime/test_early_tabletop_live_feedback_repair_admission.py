"""The repair must account for the actual failed physical work."""

import copy

import pytest

from scripts.simulation.runtime import prepare_early_tabletop_live_feedback_repair as admission


def test_repair_pins_failed_workers_and_cumulative_budget():
    item = admission.load_inputs()["early_tabletop_live_feedback_repair"]
    assert item["config"]["prior_failed_native_steps"] == 6000
    assert item["config"]["total_max_native_steps"] == 30800


def test_failure_cannot_hide_a_dispatched_new_action():
    inputs = admission.load_inputs()
    branch = copy.deepcopy(
        admission.prefix.read(inputs["early_tabletop_live_feedback_repair"]["refs"]["failed_original"])
    )
    branch["runtime"]["newly_selected_commands"] = 1
    with pytest.raises(ValueError):
        admission.validate_failed_branch(
            branch,
            mode="original",
            state_sha256=inputs["early_tabletop_live_feedback"]["branches"]["original"]["state_sha256"],
        )
