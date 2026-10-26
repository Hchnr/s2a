from types import SimpleNamespace

import numpy as np

import s2a.current_model_state as module
from s2a.runtime_response_history import RuntimeResponseHistory


def test_explicit_joint_indexes_scalar_motor_and_readonly_candidate(monkeypatch):
    q = np.arange(15, dtype=float) * 0.01
    v = np.arange(15, dtype=float) * 0.02
    simdata = SimpleNamespace(
        qpos=q,
        qvel=v,
        ctrl=np.zeros(9),
        sensordata=np.arange(6.0),
        get_site_xmat=lambda _: np.eye(3),
        get_site_xpos=lambda _: np.array([0.1, 0.2, 0.3]),
        get_site_xvelp=lambda _: np.array([0.01, 0.02, 0.03]),
        get_site_xvelr=lambda _: np.array([0.04, 0.05, 0.06]),
    )
    state = np.r_[0.0, q, v]
    sim = SimpleNamespace(data=simdata, get_state=lambda: SimpleNamespace(flatten=lambda: state.copy()))
    controller = SimpleNamespace(
        use_delta=True,
        impedance_mode="fixed",
        position_limits=None,
        orientation_limits=None,
        eef_name="site",
        input_min=-np.ones(6),
        input_max=np.ones(6),
        output_min=-np.ones(6) * 0.05,
        output_max=np.ones(6) * 0.05,
        goal_pos=np.array([0.1, 0.2, 0.3]),
        goal_ori=np.eye(3),
    )
    robot = SimpleNamespace(
        controller=controller,
        gripper=SimpleNamespace(current_action=np.array([0.2])),
        _ref_joint_pos_indexes=[3, 4, 5, 6, 7, 8, 9],
        _ref_joint_vel_indexes=[4, 5, 6, 7, 8, 9, 10],
        _ref_gripper_joint_pos_indexes=[11, 13],
        _ref_gripper_joint_vel_indexes=[12, 14],
    )
    env = SimpleNamespace(robots=[robot], sim=sim)
    monkeypatch.setattr(module, "read_current_wrench", lambda _: dict(world_at_control_site=np.arange(6.0)))
    command = np.array([0.2, -0.4, 0.6, 0.0, 0.0, 0.0, 1.0])
    row, _ = module.current_model_state(
        env, command, np.zeros(7), previous_available=False, history=RuntimeResponseHistory()
    )
    np.testing.assert_array_equal(row["x_joint_position"][0], q[[3, 4, 5, 6, 7, 8, 9]])
    np.testing.assert_array_equal(row["x_joint_velocity"][0], v[[4, 5, 6, 7, 8, 9, 10]])
    np.testing.assert_array_equal(row["x_motor_before"], [[0.2, 0.2]])
    np.testing.assert_allclose(row["action_goal_delta"][0], 0.05 * command[:6], atol=1e-14)
    row["x_joint_position"][0, 0] = 999
    row["action_command"][0, 0] = 999
    assert q[3] == 0.03 and command[0] == 0.2
    from scipy.spatial.transform import Rotation

    controller.goal_ori = Rotation.from_rotvec([0.1, -0.2, 0.05]).as_matrix()
    sticky, _ = module.current_model_state(
        env, command, np.zeros(7), previous_available=False, history=RuntimeResponseHistory()
    )
    np.testing.assert_allclose(sticky["action_goal_delta"][0, 3:], [0.1, -0.2, 0.05], atol=1e-14)
