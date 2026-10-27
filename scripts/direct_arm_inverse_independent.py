"""209 independent current-state inverse inputs and scalar physical baseline."""

import math

import numpy as np
from arm_joint_wrench_independent import conditions
from pose_velocity_arm_independent_replay import PoseVelocityArmReplay
from scipy.spatial.transform import Rotation

from s2a.osc import OSCInterface


def inputs(data, desired, grip):
    count = len(data["x_site_pose"])
    row = dict(data)
    row["action_command"] = np.column_stack([np.zeros((count, 6)), grip])
    row["action_goal_delta"] = np.zeros((count, 6))
    current = conditions(row)
    poses = np.asarray(data["x_site_pose"])
    goals = np.asarray(desired)
    result = []
    for i in range(count):
        motion = goals[i, :3, 3] - poses[i, :3, 3]
        rotation = Rotation.from_matrix(goals[i, :3, :3] @ poses[i, :3, :3].T).as_rotvec()
        result.append([*current[i, :64], *current[i, 70:], *motion, *rotation])
    return np.asarray(result)


def nominal(data, desired, contract):
    interface = OSCInterface(*(contract[k] for k in ("input_min", "input_max", "output_min", "output_max")))
    response = PoseVelocityArmReplay(interface, contract["kp"], contract["kd"], 0.05)
    result = []
    for pose, target, velocity in zip(data["x_site_pose"], desired, data["x_site_twist"], strict=True):
        physical = (target[:3, 3] - pose[:3, 3] - response.momentum[:3] * velocity[:3]) / response.response[:3]
        value = interface.input_min[:3] + (physical - interface.output_min[:3]) * (
            interface.input_max[:3] - interface.input_min[:3]
        ) / (interface.output_max[:3] - interface.output_min[:3])
        result.append(np.minimum(interface.input_max[:3], np.maximum(interface.input_min[:3], value)))
    return np.asarray(result)


def features(values, parameters):
    result = []
    for row in np.asarray(values):
        z = [
            (float(v) - float(m)) / float(s)
            for v, m, s in zip(row, parameters["normalization_mean"], parameters["normalization_scale"], strict=True)
        ]
        phase = [
            sum(a * float(b) for a, b in zip(z, frequency, strict=True)) for frequency in parameters["frequencies"]
        ]
        result.append([1.0, *z, *[math.cos(p) / 8 for p in phase], *[math.sin(p) / 8 for p in phase]])
    return np.asarray(result)
