"""Optional batched free-model inner rollout; no joint projection or dispatch."""

import numpy as np
from scipy.spatial.transform import Rotation

from s2a.geometry import finite, transform, validate_transform
from s2a.research.joint_control.inverse.full_coupled_pose_inverse import left_jacobian


class BatchedFreeJointRollout:
    """Cache transition powers for the original fixed 25-by-2 ms model.

    This candidate does not implement joint-limit projection and is not a
    control or safety qualification. Inputs are copied so later mutation of a
    caller's model cannot silently change an already prepared rollout.
    """

    def __init__(self, model):
        q = finite(model["q"])
        if q.ndim != 1 or not len(q):
            raise ValueError("A nonempty finite joint state is required")
        n = len(q)
        self.q = q.copy()
        self.jacobian = finite(model["jacobian"], (6, n)).copy()
        mass = finite(model["mass"], (n, n))
        np.linalg.inv(mass)  # Preserve the original rejection of singular mass matrices.
        self.transition = finite(model["transition"], (2 * n, 2 * n)).copy()
        self.input = finite(model["input"], (2 * n, 6)).copy()
        self.affine = finite(model["affine"], (2 * n,)).copy()
        self.anchor_response = finite(model["anchor_response"], (2 * n, n)).copy()
        if not np.isfinite(model["dt"]) or not np.isclose(model["dt"] * 25, 0.05, atol=1e-12, rtol=0):
            raise ValueError("The original fixed25x2ms model is required")
        power, cumulative = np.eye(2 * n), np.zeros((2 * n, 2 * n))
        powers, sums = [], []
        for _ in range(25):
            cumulative = cumulative + power
            power = power @ self.transition
            powers.append(power.copy())
            sums.append(cumulative.copy())
        self.powers, self.sums = np.asarray(powers), np.asarray(sums)
        if not np.isfinite(self.powers).all() or not np.isfinite(self.sums).all():
            raise ValueError("Finite fixed-step transition powers are required")

    def rollout(self, velocity, physical_deltas, site, lower, upper, *, enforce_limits):
        """Return the same free-model fields as the original sequential rollout."""
        n = len(self.q)
        velocity = finite(velocity, (n,))
        deltas = finite(physical_deltas)
        site = validate_transform(site)
        lower, upper = finite(lower, (n,)), finite(upper, (n,))
        if (
            deltas.ndim != 2
            or deltas.shape[1] != 6
            or not 1 <= len(deltas) <= 40
            or site.shape != (4, 4)
            or np.any(lower >= upper)
            or enforce_limits is not False
        ):
            raise ValueError("One to forty deltas, one SITE, strict ranges and unprojected mode required")
        if np.any(np.all(deltas[:, 3:] == 0, axis=1)):
            raise ValueError("Every planned period requires a nonzero rotation reset representative")

        width = 6 * len(deltas)
        q, p, rotation = self.q.copy(), site[:3, 3].copy(), site[:3, :3].copy()
        qjac, vjac = np.zeros((n, width)), np.zeros((n, width))
        pjac, rjac = np.zeros((3, width)), np.zeros((3, width))
        records, inner_positions, inner_jacobians = [], [], []
        for index, delta in enumerate(deltas):
            anchor, anchor_jac = q.copy(), qjac.copy()
            initial_state = np.r_[np.zeros(n), velocity]
            initial_derivative = np.vstack((np.zeros((n, width)), vjac))
            delta_jac = np.zeros((6, width))
            delta_jac[:, 6 * index : 6 * (index + 1)] = np.eye(6)
            affine = self.affine + self.anchor_response @ (anchor - self.q)
            affine_jac = self.anchor_response @ anchor_jac
            states = self.powers @ initial_state + self.sums @ (self.input @ delta + affine)
            derivatives = self.powers @ initial_derivative + self.sums @ (self.input @ delta_jac + affine_jac)
            inner_positions.extend(anchor + states[:, :n])
            inner_jacobians.extend(anchor_jac + derivatives[:, :n, :])
            state, derivative = states[-1], derivatives[-1]
            movement, movement_jac = self.jacobian @ state[:n], self.jacobian @ derivative[:n]
            increment = Rotation.from_rotvec(movement[3:]).as_matrix()
            rjac = left_jacobian(movement[3:]) @ movement_jac[3:] + increment @ rjac
            rotation = increment @ rotation
            p, pjac = p + movement[:3], pjac + movement_jac[:3]
            q, qjac = anchor + state[:n], anchor_jac + derivative[:n]
            velocity, vjac = state[n:].copy(), derivative[n:].copy()
            records.append(
                {
                    "pose": transform(p, rotation),
                    "twist": self.jacobian @ velocity,
                    "joint_position": q.copy(),
                    "joint_velocity": velocity.copy(),
                    "position_jacobian": pjac.copy(),
                    "rotation_jacobian": rjac.copy(),
                    "twist_jacobian": self.jacobian @ vjac,
                    "joint_position_jacobian": qjac.copy(),
                    "joint_velocity_jacobian": vjac.copy(),
                    "weak_projection_steps": 0,
                    "position_projected_steps": 0,
                }
            )
        result = {key: np.asarray([item[key] for item in records]) for key in records[0]}
        result["initial_max_penetration_rad"] = float(max(0, np.max(lower - self.q), np.max(self.q - upper)))
        result["inner_joint_position"] = np.asarray(inner_positions)
        result["inner_joint_position_jacobian"] = np.asarray(inner_jacobians)
        return result
