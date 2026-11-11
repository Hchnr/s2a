"""Bind the shadow proposal to the original controller decision and preview."""

import copy

import numpy as np
import pytest

from s2a.research.gripper.defer_close_release import propose_deferred_close
from scripts.wam.tracking.preview import shadow_defer_close_release as shadow
from tests.research.gripper.test_defer_close_release import sample


def test_original_decision_and_preview_must_match():
    poses, openings, offset, measured, opening, motor, baseline, closing = sample()
    candidate = propose_deferred_close(poses, openings, offset, measured, opening, motor, baseline, closing)
    recorded = dict(
        command=baseline,
        latch=dict(closing_episode=closing, measured_opening=opening, motor_opening=candidate["motor_opening"]),
        preview=candidate["preview"],
    )
    result, difference = shadow.analyze_one(poses, openings, offset, measured, opening, motor, baseline, recorded)
    assert result["deferred"] and difference < 1e-10
    changed = copy.deepcopy(recorded)
    changed["preview"]["guards"]["position_span"] = True
    with pytest.raises(ValueError, match="preview decision"):
        shadow.analyze_one(poses, openings, offset, measured, opening, motor, baseline, changed)


def test_protocol_has_fixed_whole_cohorts_and_no_physics():
    _ref, config = shadow.read_config()
    assert config["correct_cases"] == 18 and config["correct_steps"] == 4907
    assert config["development_cases"] == 12 and config["development_steps_per_case"] == 600
    assert config["preview_steps"] == 6 and config["world_model_calls"] == config["physics_steps"] == 0
    assert np.isclose(config["motor_distance_m_per_period"] / config["opening_span_m"], 0.12658227848101267)
