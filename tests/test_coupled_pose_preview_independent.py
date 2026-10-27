import copy

import numpy as np
import pytest
from coupled_pose_preview_independent import verify_preview
from test_coupled_pose_preview import problem

from s2a.coupled_pose_preview import plan_pose


@pytest.mark.parametrize("horizon", [1, 2, 3, 4])
def test_real_fixture_all_available_horizons_with_completed_innovation(horizon):
    p = problem()
    p["references"] = p["references"][:horizon]
    p["reference_twists"] = p["reference_twists"][:horizon]
    result = plan_pose(**p)
    proof = verify_preview(result, **p)
    assert proof["max_independent_stationarity"] < 2e-6


@pytest.mark.parametrize("corruption", ["first_command", "prediction", "objective", "innovation", "rate"])
def test_corrupt_executed_command_prediction_and_history_are_rejected(corruption):
    p = problem()
    result = copy.deepcopy(plan_pose(**p))
    if corruption == "first_command":
        result["command"][0] += 0.01
    elif corruption == "prediction":
        result["predicted_tangent_states"][1, 0] += 0.001
    elif corruption == "objective":
        result["objective"] += 0.1
    elif corruption == "innovation":
        result["innovation"][8] += 0.001
    else:
        result["planned_commands"][1] = np.ones(6) * 2
    with pytest.raises(AssertionError):
        verify_preview(result, **p)
