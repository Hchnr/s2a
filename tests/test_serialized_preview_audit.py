import copy
import json

import numpy as np
import pytest
from serialized_rate_preview_independent import verify_plan
from test_rate_preview_active_set import contract

from s2a.rate_constrained_preview_active_set import plan_translation


def example():
    c = contract()
    position, velocity, previous = np.zeros(3), np.zeros(3), np.zeros(3)
    references = np.tile([0.003, -0.001, 0.002], (6, 1))
    targets, innovation = np.zeros((6, 3)), np.zeros((2, 3))
    plan = plan_translation(position, velocity, references, targets, previous, c, innovation=innovation)
    saved = json.loads(json.dumps(plan, default=lambda value: value.tolist(), allow_nan=False))
    return saved, (position, velocity, references, targets, previous, c, innovation)


def test_saved_json_preview_passes_same_numeric_kkt_without_mutation():
    plan, args = example()
    before = copy.deepcopy(plan)
    result = verify_plan(plan, *args)
    assert result["convex_kkt_verified"]
    assert result["max_stationarity_residual"] < 2e-6
    assert plan == before


def test_corrupted_saved_future_commands_and_predictions_are_rejected():
    plan, args = example()
    for key in ("planned_commands", "predicted_positions", "predicted_velocities"):
        broken = copy.deepcopy(plan)
        broken[key][2][0] += 0.02
        with pytest.raises(AssertionError):
            verify_plan(broken, *args)
