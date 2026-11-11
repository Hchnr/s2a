"""The fallback matrix must preserve all original development identities."""

from scripts.simulation.runtime import prepare_model_safe_tracking_fallback as binding


def test_registration_keeps_three_tasks_and_the_original_acceptance_thresholds():
    _ref, config = binding.read_config()
    assert config["replan_case_indices"] == [2, 6] and config["task2_case_indices"] == [10, 11]
    assert config["position_cap_m"] == 0.01 and config["rotation_cap_rad"] == 0.05
    assert config["fallback_position_limit_m"] == 0.02 and config["fallback_rotation_limit_rad"] == 0.1
    assert config["world_model_calls"] == config["fresh_cases"] == 0


def test_registered_bounds_and_budget_match_candidate_implementation():
    from s2a.research.joint_control.preview import model_safe_tracking_fallback as candidate

    _ref, config = binding.read_config()
    assert config["fallback_position_limit_m"] == candidate.POSITION_LIMIT
    assert config["fallback_rotation_limit_rad"] == candidate.ROTATION_LIMIT
    assert config["quality_response_budget"] == candidate.MAX_RESPONSES
