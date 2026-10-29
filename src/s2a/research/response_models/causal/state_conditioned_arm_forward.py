"""088 conditional residual forward model, affine only in translation goal delta."""

import numpy as np
from scipy.linalg import expm
from scipy.spatial.transform import Rotation


def raw_condition(data):
    q = np.asarray(data["x_joint_position"], dtype=float)
    if q.ndim != 2 or q.shape[1] != 7 or not len(q) or not np.isfinite(q).all():
        raise ValueError("Finite current arm joint positions required")
    n = len(q)

    def get(name, shape):
        value = np.asarray(data[name], dtype=float)
        if value.shape != (n, *shape) or not np.isfinite(value).all():
            raise ValueError("Invalid current condition: " + name)
        return value

    qd = get("x_joint_velocity", (7,))
    pose = get("x_site_pose", (4, 4))
    twist = get("x_site_twist", (6,))
    if not np.allclose(pose[:, 3, :], np.tile([0, 0, 0, 1], (n, 1)), rtol=0, atol=1e-10) or not np.allclose(
        pose[:, :3, :3].transpose(0, 2, 1) @ pose[:, :3, :3], np.tile(np.eye(3), (n, 1, 1)), rtol=0, atol=1e-8
    ):
        raise ValueError("Current rigid SITE transforms required")
    finger_q = get("x_finger_qpos", (2,))
    finger_v = get("x_finger_qvel", (2,))
    motor = get("x_motor_before", (2,))
    command = get("action_command", (7,))
    previous = get("x_previous_command", (7,))
    if np.any(np.abs(motor) > 1) or np.any(np.abs(command) > 1):
        raise ValueError("Known motor/command outside native interface")
    after = motor.copy()
    for _ in range(25):
        after = np.clip(after + np.sign(command[:, -1, None]) * np.array([-0.01, 0.01]), -1, 1)
    previous_position = get("x_previous_goal_pos", (3,)) - pose[:, :3, 3]
    previous_rotation = Rotation.from_matrix(
        get("x_previous_goal_ori", (3, 3)) @ pose[:, :3, :3].transpose(0, 2, 1)
    ).as_rotvec()
    rotation_goal = get("action_goal_delta", (6,))[:, 3:]
    theta = np.linalg.norm(rotation_goal, axis=1)
    spring = rotation_goal * np.sinc(theta / np.pi)[:, None]
    available = get("x_previous_command_available", ())[:, None]
    if not np.isin(available, [0, 1]).all():
        raise ValueError("Explicit previous command availability required")
    rotation_columns = np.c_[pose[:, :3, 0], pose[:, :3, 1]]
    condition = np.c_[
        q,
        np.sin(q),
        np.cos(q),
        qd,
        pose[:, :3, 3],
        rotation_columns,
        twist,
        finger_q,
        finger_v,
        motor,
        after,
        previous,
        previous_position,
        previous_rotation,
        rotation_goal,
        spring,
        available,
    ]
    assert condition.shape == (n, 71)
    return condition


def design(data, parameters):
    raw = raw_condition(data)
    state_mean = np.asarray(parameters["state_mean"])
    state_scale = np.asarray(parameters["state_scale"])
    mean = np.asarray(parameters["translation_mean"])
    scale = np.asarray(parameters["translation_scale"])
    if state_mean.shape != (71,) or state_scale.shape != (71,) or mean.shape != (3,) or scale.shape != (3,):
        raise ValueError("Fixed71/3 statistics required")
    if (
        not all(np.isfinite(x).all() for x in (state_mean, state_scale, mean, scale))
        or np.any(state_scale <= 0)
        or np.any(scale <= 0)
    ):
        raise ValueError("Finite positive fit scales required")
    translation = np.asarray(data["action_goal_delta"], dtype=float)[:, :3]
    phi = np.c_[np.ones(len(raw)), (raw - state_mean) / state_scale]
    action = (translation - mean) / scale
    result = np.c_[phi, *(phi * action[:, i, None] for i in range(3))]
    assert result.shape == (len(raw), 288)
    return result


def nominal(data, parameters):
    dt = float(parameters["dt"])
    kp = np.asarray(parameters["kp"], dtype=float)
    kd = np.asarray(parameters["kd"], dtype=float)
    if (
        dt != 0.05
        or kp.shape != (6,)
        or kd.shape != (6,)
        or not np.isfinite(kp).all()
        or not np.isfinite(kd).all()
        or np.any(kp <= 0)
        or np.any(kd <= 0)
    ):
        raise ValueError("Fixed20Hz positive PD contract required")
    transitions = np.array([expm(np.array([[0.0, 1.0], [-p, -d]]) * dt) for p, d in zip(kp, kd, strict=True)])
    goal = np.asarray(data["action_goal_delta"], dtype=float)
    velocity = np.asarray(data["x_site_twist"], dtype=float)
    if (
        goal.ndim != 2
        or goal.shape[1] != 6
        or goal.shape != velocity.shape
        or not np.isfinite(goal).all()
        or not np.isfinite(velocity).all()
    ):
        raise ValueError("Current applied goal and current measured twist required")
    motion = (1 - transitions[:, 0, 0]) * goal + transitions[:, 0, 1] * velocity
    next_velocity = -transitions[:, 1, 0] * goal + transitions[:, 1, 1] * velocity
    return np.c_[motion, dt * next_velocity]


def targets(data, dt=0.05):
    if dt != 0.05:
        raise ValueError("Fixed20Hz target units required")
    value = np.c_[data["y_motion"], data["y_rotation_increment"], dt * np.asarray(data["y_twist"])]
    if value.ndim != 2 or value.shape[1] != 12 or not np.isfinite(value).all():
        raise ValueError("Finite separate next-state labels required")
    return value


def predict(data, parameters):
    coefficients = np.asarray(parameters["coefficients"], dtype=float)
    if coefficients.shape != (288, 12) or not np.isfinite(coefficients).all():
        raise ValueError("Finite288x12 coefficients required")
    value = nominal(data, parameters) + design(data, parameters) @ coefficients
    if not np.isfinite(value).all():
        raise ValueError("Nonfinite forward prediction")
    return value
