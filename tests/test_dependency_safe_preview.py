"""Keep complete successful forecasts and stale-goal reset behavior unchanged."""

import numpy as np
import pytest
from test_coupled_pose_preview import problem

from s2a.coupled_pose_preview_runtime import solve_preview as original_solve
from s2a.dependency_safe_preview_runtime import solve_preview


def same(actual, expected):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            same(actual[key], expected[key])
    else:
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("index", [0, 3])
@pytest.mark.parametrize("stationary", [False, True])
def test_success_path_preserves_every_returned_forecast(index, stationary):
    p = problem()
    if stationary:
        p["twist"] = np.zeros(6)
        p["previous_command"] = np.zeros(6)
        p["innovation"] = np.zeros(12)
        p["references"] = np.repeat(p["measured"][None], 4, axis=0)
    args = (
        p["mass"],
        p["jacobian"],
        p["measured"],
        p["twist"],
        p["references"],
        p["measured"],
        index,
        p["previous_command"],
        p["contract"],
    )
    same(solve_preview(*args, innovation=p["innovation"]), original_solve(*args, innovation=p["innovation"]))
