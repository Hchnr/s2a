"""Recorded inputs must be available before any counterfactual action."""

import copy

import pytest

from scripts.simulation.runtime import prepare_tabletop_projection_feedback_probe as admission


def test_recorded_branch_and_generated_reference_are_source_pinned():
    inputs = admission.load_inputs()["tabletop_projection_feedback_probe"]
    assert inputs["config"]["branch_step"] == 120
    assert inputs["prefix_stream"]["control_steps_completed"] == 40
    assert [row["start_step"] for row in inputs["prefix_stream"]["segments"]] == [80, 88, 96, 104, 112]
    assert len(inputs["target_chunks"]) == 6 and inputs["target_chunks"][5].shape == (8, 4, 4)


def test_incomplete_recorded_segment_is_not_replayable():
    inputs = admission.load_inputs()["tabletop_projection_feedback_probe"]
    stream = copy.deepcopy(inputs["previous_producer"]["runtime"]["continuation"])
    stream["segments"][3]["continuation"]["periods"].pop()
    with pytest.raises(ValueError):
        admission.replay_prefix_stream(stream)
