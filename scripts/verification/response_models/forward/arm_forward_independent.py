"""088 independent scalar condition assembly and closed-form nominal response."""

import math

import numpy as np
from scipy.spatial.transform import Rotation

INPUTS = (
    "x_joint_position",
    "x_joint_velocity",
    "x_site_pose",
    "x_site_twist",
    "x_finger_qpos",
    "x_finger_qvel",
    "x_motor_before",
    "action_command",
    "x_previous_command",
    "x_previous_goal_pos",
    "x_previous_goal_ori",
    "action_goal_delta",
    "x_previous_command_available",
)


def conditions(data):
    output = []
    for i in range(len(data["x_joint_position"])):
        row = {k: np.asarray(data[k][i]) for k in INPUTS}
        q = row["x_joint_position"]
        pose = row["x_site_pose"]
        motor = list(row["x_motor_before"].astype(float))
        sign = float(np.sign(row["action_command"][-1]))
        for _ in range(25):
            motor = [min(1.0, max(-1.0, motor[0] - 0.01 * sign)), min(1.0, max(-1.0, motor[1] + 0.01 * sign))]
        previous_rotation = Rotation.from_matrix(row["x_previous_goal_ori"] @ pose[:3, :3].T).as_rotvec()
        rot = row["action_goal_delta"][3:]
        theta = math.sqrt(float(rot @ rot))
        spring = rot * (math.sin(theta) / theta if theta else 1.0)
        values = [
            *q,
            *np.sin(q),
            *np.cos(q),
            *row["x_joint_velocity"],
            *pose[:3, 3],
            *pose[:3, 0],
            *pose[:3, 1],
            *row["x_site_twist"],
            *row["x_finger_qpos"],
            *row["x_finger_qvel"],
            *row["x_motor_before"],
            *motor,
            *row["x_previous_command"],
            *(row["x_previous_goal_pos"] - pose[:3, 3]),
            *previous_rotation,
            *rot,
            *spring,
            float(row["x_previous_command_available"]),
        ]
        assert len(values) == 71
        output.append(values)
    value = np.array(output, dtype=np.float64)
    assert np.isfinite(value).all()
    return value


def feature_matrix(data, parameters):
    raw = conditions(data)
    phi = np.concatenate(
        [np.ones((len(raw), 1)), (raw - np.array(parameters["state_mean"])) / np.array(parameters["state_scale"])],
        axis=1,
    )
    action = (np.asarray(data["action_goal_delta"])[:, :3] - np.array(parameters["translation_mean"])) / np.array(
        parameters["translation_scale"]
    )
    return np.concatenate([phi, *[phi * action[:, j : j + 1] for j in range(3)]], axis=1)


def nominal_prediction(data, parameters):
    dt = parameters["dt"]
    r = []
    b = []
    for p, d in zip(parameters["kp"], parameters["kd"], strict=True):
        a = d / 2
        if math.isclose(a * a, p, rel_tol=1e-12, abs_tol=1e-12):
            f = math.exp(-a * dt)
            e00 = f * (1 + a * dt)
            e01 = f * dt
        elif p > a * a:
            omega = math.sqrt(p - a * a)
            f = math.exp(-a * dt)
            e00 = f * (math.cos(omega * dt) + a * math.sin(omega * dt) / omega)
            e01 = f * math.sin(omega * dt) / omega
        else:
            omega = math.sqrt(a * a - p)
            s1, s2 = -a + omega, -a - omega
            f1, f2 = math.exp(s1 * dt), math.exp(s2 * dt)
            e00 = (-s2 * f1 + s1 * f2) / (s1 - s2)
            e01 = (f1 - f2) / (s1 - s2)
        r.append(1 - e00)
        b.append(e01)
    r, b = np.array(r), np.array(b)
    kp = np.array(parameters["kp"])
    kd = np.array(parameters["kd"])
    goal = np.asarray(data["action_goal_delta"])
    v = np.asarray(data["x_site_twist"])
    return np.concatenate([r * goal + b * v, dt * (kp * b * goal + (1 - r - kd * b) * v)], axis=1)


def prediction(data, parameters):
    return nominal_prediction(data, parameters) + feature_matrix(data, parameters) @ np.array(
        parameters["coefficients"]
    )
