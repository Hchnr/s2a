"""Capture the last completed prefix period's actual inner joint samples."""

import numpy as np


class FinalPrefixInner:
    def __init__(self, prefix_steps):
        if type(prefix_steps) is not int or prefix_steps < 1:
            raise ValueError("A positive completed prefix is required")
        self.prefix_steps = prefix_steps
        self.samples = []
        self.limits = None

    def observe(self, env, completed_steps):
        if completed_steps != self.prefix_steps - 1:
            return
        if len(self.samples) >= 25:
            raise ValueError("The last prefix period cannot exceed 25 inner samples")
        robot, sim = env.robots[0], env.sim
        q = np.asarray(sim.data.qpos)[robot._ref_joint_pos_indexes].copy()
        velocity = np.asarray(sim.data.qvel)[robot._ref_joint_vel_indexes].copy()
        ids = [sim.model.joint_name2id(f"robot0_joint{i}") for i in range(1, 8)]
        limits = np.asarray(sim.model.jnt_range)[ids].copy()
        time = float(sim.data.time)
        if (
            q.shape != (7,)
            or velocity.shape != (7,)
            or limits.shape != (7, 2)
            or not np.isfinite(q).all()
            or not np.isfinite(velocity).all()
            or not np.isfinite(limits).all()
            or not np.isfinite(time)
            or np.any(limits[:, 0] >= limits[:, 1])
        ):
            raise ValueError("Finite inner robot state and joint limits required")
        if self.limits is not None and not np.array_equal(self.limits, limits):
            raise ValueError("Joint limits must remain fixed during the period")
        self.limits = limits
        self.samples.append({"time": time, "q": q, "qvel": velocity})

    def finish(self):
        if len(self.samples) != 25 or self.limits is None:
            raise ValueError("Exactly 25 actual final-prefix inner samples are required")
        np.testing.assert_allclose(np.diff([row["time"] for row in self.samples]), 0.002, atol=1e-12, rtol=0)
        q = np.asarray([row["q"] for row in self.samples])
        margins = np.minimum(q - self.limits[:, 0], self.limits[:, 1] - q)
        return {
            "prefix_step": self.prefix_steps - 1,
            "samples": self.samples,
            "limits": self.limits,
            "minimum_margin_rad": float(margins.min()),
            "below_buffer_samples": int(np.sum(np.any(margins < 0.05, axis=1))),
            "outside_samples": int(np.sum(np.any(margins < 0, axis=1))),
        }
