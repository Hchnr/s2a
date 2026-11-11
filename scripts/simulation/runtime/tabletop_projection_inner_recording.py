"""Observe pre-integration geometry and post-solver contact in the short probe."""

import numpy as np

from s2a.geometry import validate_transform
from s2a.research.joint_control.tracking.tabletop_projection_observation import observed_bound_conditions
from scripts.tracking.response_models.replay_contact_torque import ReplayRecorder


class ProjectionInnerRecorder:
    """Wrap the existing read-only torque/contact recorder for seven periods."""

    def __init__(self, env, sensor, mode, *, base=None):
        if mode not in ("original", "nominal", "pose_bounded"):
            raise ValueError("A registered comparison mode is required")
        self.env, self.sensor, self.mode = env, sensor, mode
        self.base = ReplayRecorder(env, 7) if base is None else base
        self.original, self.samples = self.base.original, self.base.samples
        self.context = None

    def set_context(self, step, selected_tcp, vertices_tcp):
        if type(step) is not int or step != 120 + len(self.samples) // 25 or len(self.samples) % 25:
            raise ValueError("A fresh completed decision is required at each physical period")
        target = validate_transform(np.asarray(selected_tcp, dtype=float)).copy()
        vertices = np.asarray(vertices_tcp, dtype=float).copy()
        if (
            target.shape != (4, 4)
            or vertices.ndim != 2
            or vertices.shape[1] != 3
            or not len(vertices)
            or not np.isfinite(vertices).all()
        ):
            raise ValueError("A finite selected target and corresponding measured vertices are required")
        self.context = {"step": step, "selected_tcp": target, "vertices_tcp": vertices}

    def step(self, *args, **kwargs):
        if self.context is None or self.context["step"] != 120 + len(self.samples) // 25:
            raise ValueError("No current decision has been registered for this inner step")
        before = self.sensor.capture()
        controller, gripper = self.env.robots[0].controller, self.env.robots[0].gripper
        goal_pos, goal_ori = np.asarray(controller.goal_pos).copy(), np.asarray(controller.goal_ori).copy()
        gripper_action = np.asarray(gripper.current_action).copy()
        position, rotation = (0.01, 0.05) if self.mode == "pose_bounded" else (0.0, 0.0)
        conditions = observed_bound_conditions(
            self.context["vertices_tcp"],
            before["vertices_tcp"],
            self.context["selected_tcp"],
            before["tcp"],
            position_bound=position,
            rotation_bound=rotation,
            shape_bound=0.0,
        )
        value = self.base.step(*args, **kwargs)
        row = self.samples[-1]
        if not np.isclose(row["time_before"], before["time_s"], atol=1e-12, rtol=0):
            raise ValueError("Geometry and control must be sampled before the same integration")
        row.update(
            step=self.context["step"],
            geometry_sample_phase="after_environment_forward_before_integration",
            contact_sample_phase="last_inner_step_solver_cache_after_integration",
            tcp_before=before["tcp"],
            controller_goal_position=goal_pos,
            controller_goal_rotation=goal_ori,
            gripper_motor_target=gripper_action,
            bounded_sample_conditions=conditions,
            projection_bounds_applicable=self.mode != "original",
        )
        return value
