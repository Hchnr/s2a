"""The release screen must bind original first interventions and current state."""

import json

import numpy as np
import pytest

from scripts.simulation.runtime import screen_causal_opening_events as screen


def test_current_row_exposes_only_pre_action_fields(tmp_path):
    rows = tmp_path / "steps.jsonl"
    row = {
        "step": 1,
        "qualified_forward_input": {"x_joint_position": list(range(7)), "x_future": "forbidden"},
        "measured_source": np.eye(4).tolist(),
        "opening_before": 0.2,
        "objects": {"privileged": True},
        "command": [1] * 7,
        "opening_after": 0.9,
    }
    rows.write_text("not decoded\n" + json.dumps(row) + "\nnot decoded\n")
    actual = screen.current_row(rows, 1)
    assert set(actual) == {"q", "source_pose", "opening"}
    np.testing.assert_array_equal(actual["q"], list(range(7)))
    assert actual["opening"] == 0.2
    with pytest.raises(ValueError):
        screen.current_row(rows, 3)


@pytest.fixture(scope="module")
def screened_cases():
    context = screen.load_context()
    return [(item, screen.screen_case(item, context["config"])) for item in context["cases"]]


def test_four_original_first_openings_are_bound_without_physics(screened_cases):
    assert [item["spec"]["case_index"] for item, _ in screened_cases] == [3, 7, 10, 11]
    for item, result in screened_cases:
        spec, admission = item["spec"], result["admission"]
        assert result["reference_sources"]["event_endpoint"] == spec["event_endpoint"]
        assert result["reference_sources"]["remembered_step"] < spec["prefix_steps"]
        assert not result["prior_inner_samples_checked"] and not result["physical_feedback_executed"]
        q, limits = admission["joint_positions_rad"], admission["joint_limits_rad"]
        independent = min(min(value - low, high - value) for value, (low, high) in zip(q, limits, strict=True))
        assert admission["minimum_margin_rad"] == pytest.approx(independent, abs=1e-15)
        assert admission["recorded_endpoint_admitted"] == bool(independent >= 0.05)
