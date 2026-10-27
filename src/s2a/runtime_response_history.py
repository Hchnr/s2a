"""Completed-endpoint innovation identical to training with_history conventions."""

import numpy as np
from scipy.spatial.transform import Rotation

from s2a.joint_wrench_condition_forward import nominal


class RuntimeResponseHistory:
    def __init__(self):
        self.reset()

    def reset(self):
        self.innovation = np.zeros(12)
        self.expected_pose = None
        self.expected_twist = None
        self.completed_periods = 0

    @staticmethod
    def endpoint(pose, twist):
        pose, twist = np.asarray(pose, dtype=float), np.asarray(twist, dtype=float)
        if pose.shape != (4, 4) or twist.shape != (6,) or not np.isfinite(pose).all() or not np.isfinite(twist).all():
            raise ValueError("Finite measured SITE pose4x4 and world twist6 required")
        if (
            not np.allclose(pose[3], [0, 0, 0, 1], rtol=0, atol=1e-10)
            or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), rtol=0, atol=1e-8)
            or np.linalg.det(pose[:3, :3]) <= 0
        ):
            raise ValueError("Proper measured rigid transform required")
        return pose, twist

    def attach(self, data):
        pose, twist = self.endpoint(np.asarray(data["x_site_pose"])[0], np.asarray(data["x_site_twist"])[0])
        if len(data["x_site_pose"]) != 1 or len(data["x_site_twist"]) != 1:
            raise ValueError("One current measured endpoint required")
        if self.expected_pose is not None and (
            not np.allclose(pose, self.expected_pose, rtol=0, atol=1e-9)
            or not np.allclose(twist, self.expected_twist, rtol=0, atol=1e-9)
        ):
            raise ValueError("Current endpoint differs from last completed observation")
        result = {k: np.asarray(v).copy() for k, v in data.items()}
        result["x_response_residual"] = self.innovation[None, :].copy()
        return result

    def observe_completed(self, executed_data, next_pose, next_twist, parameters):
        data = self.attach(executed_data)
        next_pose, next_twist = self.endpoint(next_pose, next_twist)
        current = np.asarray(data["x_site_pose"])[0]
        prediction = nominal(data, parameters)[0]
        innovation = np.r_[
            next_pose[:3, 3] - current[:3, 3] - prediction[:3],
            (
                Rotation.from_matrix(next_pose[:3, :3] @ current[:3, :3].T)
                * Rotation.from_rotvec(prediction[3:6]).inv()
            ).as_rotvec(),
            0.05 * next_twist - prediction[6:],
        ]
        self.innovation = innovation
        self.expected_pose = next_pose.copy()
        self.expected_twist = next_twist.copy()
        self.completed_periods += 1
