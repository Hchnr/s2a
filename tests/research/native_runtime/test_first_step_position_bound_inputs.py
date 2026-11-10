"""The outer-bound study retains the exact prior development states."""

import numpy as np

from scripts.tracking.trajectory_search import audit_first_step_position_bound as study


def test_prior_four_states_and_saved_attempts_are_bound():
    inputs = study.load_inputs()
    assert [case["comparison_spec"]["case_index"] for case in inputs["cases"]] == [2, 6, 10, 11]
    assert [case["comparison_spec"]["step"] for case in inputs["cases"]] == [48, 57, 34, 82]
    for case in inputs["cases"]:
        prior = case["previous_data"]
        assert len(prior["comparison"]["input_sha256"]) == 64
        closest = prior["comparison"]["solvers"]["feasibility_first"]["solver_result"][
            "closest_linear_feasible_attempt"
        ]
        assert np.asarray(closest["first_tcp_error"]).shape == (6,)
        assert np.asarray(prior["past_inner_error_rad"]).shape == (25, 7)


def test_no_physics_or_new_model_budget_in_input_receipt():
    inputs = study.load_inputs()
    receipt = study.input_receipt(inputs)
    assert receipt["prior_ref"] == study.PRIOR
    assert (
        inputs["config"]["physics_steps"]
        == inputs["config"]["world_model_calls"]
        == inputs["config"]["fresh_cases"]
        == 0
    )
