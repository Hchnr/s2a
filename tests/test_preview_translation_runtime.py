import copy

import numpy as np
import pytest
from rate_preview_independent import verify_plan
from test_rate_preview_active_set import contract

from s2a.preview_translation_runtime import plan_current_translation


def sample():
    generated = np.repeat(np.eye(4)[None], 8, axis=0)
    generated[:, :3, 3] = np.arange(1, 9)[:, None] * np.array([0.0005, -0.0002, 0.0001])
    data = dict(
        x_site_pose=np.eye(4)[None],
        x_site_twist=np.zeros((1, 6)),
        x_previous_command=np.zeros((1, 7)),
        x_previous_command_available=np.array([False]),
        x_response_residual=np.zeros((1, 12)),
    )
    return data, generated


def test_initial_generated_preview_and_short_tail_have_correct_known_velocities():
    data, generated = sample()
    for index, horizon in ((0, 6), (6, 2), (7, 1)):
        result = plan_current_translation(data, generated, index, np.eye(4), contract())
        assert result["horizon"] == horizon
        assert result["generated_indices"] == list(range(index, index + horizon))
        np.testing.assert_allclose(result["reference_positions"], generated[index : index + horizon, :3, 3], atol=1e-15)
        np.testing.assert_allclose(
            result["reference_velocities"], np.tile([0.01, -0.004, 0.002], (horizon, 1)), atol=1e-15
        )
        np.testing.assert_allclose(result["physical_delta"], result["command"] * 0.05, atol=1e-15)
        verify_plan(
            result,
            np.zeros(3),
            np.zeros(3),
            result["reference_positions"],
            result["reference_velocities"],
            np.zeros(3),
            contract(),
            np.zeros((2, 3)),
        )


def test_teacher_actions_and_future_measured_labels_cannot_affect_current_plan():
    data, generated = sample()
    original = copy.deepcopy(data)
    first = plan_current_translation(data, generated, 0, np.eye(4), contract())
    data["action_command"] = np.full((1, 7), np.nan)
    data["action_goal_delta"] = np.full((1, 6), np.nan)
    data["y_motion"] = np.full((1, 6), np.nan)
    second = plan_current_translation(data, generated, 0, np.eye(4), contract())
    np.testing.assert_array_equal(first["planned_commands"], second["planned_commands"])
    for key in original:
        np.testing.assert_array_equal(data[key], original[key])


def test_unavailable_cursor_and_stale_episode_memory_are_rejected():
    data, generated = sample()
    with pytest.raises(ValueError, match="cursor"):
        plan_current_translation(data, generated, len(generated), np.eye(4), contract())
    data["x_response_residual"][0, 0] = 0.001
    with pytest.raises(ValueError, match="reset"):
        plan_current_translation(data, generated, 0, np.eye(4), contract())
