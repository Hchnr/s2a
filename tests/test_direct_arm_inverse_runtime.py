import numpy as np
import pytest
from test_direct_arm_inverse import parameters
from test_direct_arm_inverse_inputs import prepared

from s2a.direct_arm_inverse_runtime import translation_command


def test_final_rate_limit_uses_actual_previous_command_and_decodes_final_action():
    data, desired = prepared()
    p = parameters()
    p["coefficients"][0] = [10, -10, 10]
    data["x_previous_command"][1, :3] = [0.9, -0.9, -0.6]
    result = translation_command(data, desired, gripper_command=np.zeros(2), parameters=p)
    np.testing.assert_allclose(result["command"], [[0.25, -0.25, 0.25], [1, -1, -0.35]])
    np.testing.assert_allclose(result["physical_delta"], 0.05 * result["command"], atol=1e-15)
    assert result["rate_limited"].all()
    # Replanning after a different actual execution must use that execution.
    data["x_previous_command"][1, :3] = [-0.9, 0.9, 0.6]
    changed = translation_command(data, desired, gripper_command=np.zeros(2), parameters=p)
    np.testing.assert_allclose(changed["command"][1], [-0.65, 0.65, 0.85])


def test_episode_reset_rejects_stale_history_and_invalid_prior_command():
    data, desired = prepared()
    p = parameters()
    data["x_previous_command"][0, 0] = 0.1
    with pytest.raises(ValueError, match="reset"):
        translation_command(data, desired, gripper_command=np.zeros(2), parameters=p)
    data["x_previous_command"][0, 0] = 0
    data["x_previous_command"][1, 0] = np.nan
    with pytest.raises(ValueError, match="previous"):
        translation_command(data, desired, gripper_command=np.zeros(2), parameters=p)
