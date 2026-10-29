"""TCP offset/angular derivatives and influence of already generated future targets."""

import numpy as np
from scipy.spatial.transform import Rotation

from s2a.joint_limit_preview import build_joint_preview, rollout_joint_preview
from s2a.joint_limit_preview_objective import preview_objective


def problem():
    rng = np.random.default_rng(42)
    a = rng.normal(size=(7, 7))
    mass = np.eye(7) + 0.02 * a.T @ a
    j = np.c_[np.eye(6), rng.normal(size=6) * 0.1]
    model = build_joint_preview(
        mass, j, np.ones(6) * 100, np.ones(6) * 20, np.zeros(7), np.ones(7) * 0.02, null_kp=10.0, dt=0.002
    )
    commands = rng.normal(size=(2, 6)) * 0.03
    tool = np.eye(4)
    tool[:3, 3] = [0.1, 0.02, -0.03]
    tool[:3, :3] = Rotation.from_rotvec([0.2, -0.1, 0.3]).as_matrix()
    desired = np.tile(tool, (2, 1, 1))
    desired[:, :3, 3] += [0.01, -0.02, 0.01]
    desired_twists = rng.normal(size=(2, 6)) * 0.02
    return model, commands, tool, desired, desired_twists


def test_tcp_pose_velocity_objective_derivative_includes_tool_offset():
    model, commands, tool, desired, desired_twists = problem()

    def evaluate(c):
        rollout = rollout_joint_preview(model, np.zeros(7), c, np.eye(4), -np.ones(7), np.ones(7), enforce_limits=True)
        return preview_objective(rollout, tool, desired, desired_twists)

    base = evaluate(commands)
    epsilon = 1e-6
    for column in range(commands.size):
        plus, minus = commands.copy(), commands.copy()
        plus.flat[column] += epsilon
        minus.flat[column] -= epsilon
        a, b = evaluate(plus), evaluate(minus)
        np.testing.assert_allclose(
            (a["residual"] - b["residual"]) / (2 * epsilon), base["jacobian"][:, column], atol=2e-6, rtol=1e-5
        )
        np.testing.assert_allclose(
            (a["objective"] - b["objective"]) / (2 * epsilon), base["gradient"][column], atol=2e-6, rtol=1e-5
        )


def test_available_future_reference_changes_first_control_gradient():
    model, _, tool, _, _ = problem()
    commands = np.zeros((4, 6))
    commands[:, 3] = 1e-9
    rollout = rollout_joint_preview(
        model, np.zeros(7), commands, np.eye(4), -np.ones(7), np.ones(7), enforce_limits=False
    )
    desired = rollout["pose"] @ tool
    before = preview_objective(rollout, tool, desired, np.zeros((4, 6)))
    changed = desired.copy()
    changed[1:, 0, 3] += 0.02
    after = preview_objective(rollout, tool, changed, np.zeros((4, 6)))
    np.testing.assert_array_equal(before["residual"][:12], after["residual"][:12])
    assert np.linalg.norm(before["gradient"][:6] - after["gradient"][:6]) > 0.01
