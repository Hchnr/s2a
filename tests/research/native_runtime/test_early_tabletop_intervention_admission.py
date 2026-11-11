"""The early intervention must use only a previously generated H40."""

import copy

import pytest

from scripts.simulation.runtime import prepare_early_tabletop_intervention as admission


def test_early_branch_uses_step112_reference_and_38_completed_periods():
    inputs = admission.load_inputs()["early_tabletop_intervention"]
    assert inputs["config"]["branch_step"] == 118 and inputs["config"]["source_reference_step"] == 112
    assert inputs["source_reference"].shape == (40, 4, 4)
    stream = inputs["prefix_stream"]
    assert stream["control_steps_completed"] == 38 and stream["stop"]["at_step"] == 118
    assert [p["step"] for p in stream["segments"][-1]["continuation"]["periods"]] == list(range(112, 118))


def test_early_branch_rejects_an_incomplete_source_segment():
    inputs = admission.load_inputs()
    stream = copy.deepcopy(inputs["tabletop_projection_feedback_probe"]["previous_producer"]["runtime"]["continuation"])
    stream["segments"][4]["continuation"]["periods"].pop()
    with pytest.raises(ValueError):
        admission.replay_prefix_stream(stream)
