"""Independent physical replay must detect changed forces or contact identity."""

import copy

import pytest

from scripts.simulation.runtime.run_tabletop_projection_feedback_probe import assert_observations_match, summarize


def test_independent_sample_comparison_rejects_contact_or_torque_changes():
    original = {
        "time_before": 6.0,
        "controller_raw_torque_nm": [1.0, -2.0],
        "contacts": [{"geom1": "table_collision", "geom2": "finger", "distance_m": -0.001, "active_constraint": True}],
    }
    same = copy.deepcopy(original)
    assert_observations_match(original, same)
    same["controller_raw_torque_nm"][0] += 0.01
    with pytest.raises(AssertionError):
        assert_observations_match(original, same)
    same = copy.deepcopy(original)
    same["contacts"][0]["geom1"] = "another_geom"
    with pytest.raises(ValueError):
        assert_observations_match(original, same)


def test_checker_summary_does_not_treat_missing_policy_deviation_as_zero():
    sample = {
        "step": 120,
        "contacts": [],
        "controller_raw_torque_nm": [0.0] * 7,
        "commanded_torque_nm": [0.0] * 7,
        "bounded_sample_conditions": {"max_material_vertex_displacement_tcp_m": 0.0, "all_sample_bounds_met": False},
    }
    period = {
        "step": 120,
        "decision": {},
        "pose_caps_satisfied": True,
        "completed_feedback": {"projected_pose_caps_satisfied": True},
    }
    runtime = {
        "complete": True,
        "passed": True,
        "cleanup_complete": True,
        "continuation": {
            "mode": "nominal",
            "probe": {"periods": [period], "stop": None},
            "inner_samples": [sample] * 25,
        },
        "control_steps_completed": 121,
        "counters": {"mj_step": {"completed": 3025}, "model_construct": {"completed": 3}, "context": {"completed": 3}},
        "limits": {"mj_step": 3175},
        "newly_selected_commands": 0,
        "world_model_calls": 0,
        "fresh_cases": 0,
    }
    branch = {"complete": True, "passed": True, "mode": "nominal", "phase": "checker", "runtime": runtime}
    result = summarize(branch, mode="nominal", phase="checker", config={"max_native_steps_per_phase": 3175})
    assert result["max_reference_translation_m"] is None
    assert result["stop_reason_evaluated_by_worker"] is False
    assert result["producer_stop_independently_rechecked"] is False
