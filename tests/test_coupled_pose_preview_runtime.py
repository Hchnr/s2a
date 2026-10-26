import numpy as np
from coupled_pose_preview_independent import verify_preview
from full_coupled_pose_independent import objective
from test_coupled_pose_preview import problem

from s2a.coupled_pose_preview_runtime import solve_preview


def test_chunk_tail_and_independent_goal_dependent_first_command_forecast():
    p = problem()
    for index in (0, 3):
        result = solve_preview(
            p["mass"],
            p["jacobian"],
            p["measured"],
            p["twist"],
            p["references"],
            p["measured"],
            index,
            p["previous_command"],
            p["contract"],
            innovation=p["innovation"],
        )
        assert result["preview"]["horizon"] == 4 - index
        verify_preview(
            result["preview"],
            p["measured"],
            p["twist"],
            result["references"],
            result["reference_twists"],
            p["previous_command"],
            p["mass"],
            p["jacobian"],
            p["contract"],
            innovation=p["innovation"],
        )
        checked = objective(
            result["command"],
            p["mass"],
            p["jacobian"],
            p["contract"],
            p["measured"],
            p["references"][index],
            p["twist"],
            result["reference_twists"][0],
            p["innovation"],
            p["measured"][:3, :3],
            "reset",
        )
        for key in ("physical_delta", "nominal_response", "predicted_pose", "predicted_twist"):
            np.testing.assert_allclose(result[key], checked[key], atol=1e-8, rtol=0)
        assert result["native_decode_and_actual_commit_required"]
