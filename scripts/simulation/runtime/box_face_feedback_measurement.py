"""Read-only current box, finger and single-slide measurements for feedback."""

import numpy as np

from s2a.geometry import finite, transform
from s2a.research.joint_control.tracking.tabletop_projection_observation import observed_bound_conditions
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime.tabletop_projection_measurement import CurrentFingerGeometry
from scripts.tracking.response_models.replay_contact_torque import ReplayRecorder


def _body_path(model, body):
    path, seen = [], set()
    while body:
        if body < 0 or body >= model.nbody or body in seen:
            raise ValueError("Invalid native body ancestry")
        seen.add(body)
        path.append(body)
        body = int(model.body_parentid[body])
    return [*path, 0]


def _joints(model, bodies):
    return [
        index
        for body in bodies
        for index in range(int(model.body_jntadr[body]), int(model.body_jntadr[body] + model.body_jntnum[body]))
    ]


def checked_slide_topology(model, finger_body, site_body, joint, slide_type):
    """Verify the sensor's relative motion without importing a candidate policy."""
    finger, site = _body_path(model, finger_body), _body_path(model, site_body)
    common = next(body for body in finger if body in set(site))
    if (
        _joints(model, finger[: finger.index(common)]) != [joint]
        or _joints(model, site[: site.index(common)])
        or int(model.jnt_type[joint]) != slide_type
        or not bool(model.jnt_limited[joint])
        or not np.isclose(np.linalg.norm(model.jnt_axis[joint]), 1.0, atol=1e-12, rtol=0)
    ):
        raise ValueError("Finger-to-TCP kinematics must contain one limited unit slide")
    interval = finite(model.jnt_range[joint], (2,)).copy()
    if interval[0] > interval[1]:
        raise ValueError("The native slide interval must be ordered")
    return interval


class CurrentBoxFaceGeometry:
    """Read a verified current finger and declared fixed box calibrations."""

    def __init__(self, env, *, site_from_tcp, config):
        import mujoco

        self.env, self.config = env, config
        self.finger = CurrentFingerGeometry(
            env,
            site_from_tcp=site_from_tcp,
            table_pose=config["table_pose"],
            table_half_size=config["table_half_size_m"],
            table_name=config["table_geom"],
            finger_name=config["finger_geom"],
        )
        self.model = self.finger.model
        self.stove = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, config["stove_geom"])
        self.joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, config["finger_joint"])
        if (
            min(self.stove, self.joint) < 0
            or self.model.geom_type[self.stove] != mujoco.mjtGeom.mjGEOM_BOX
            or not (
                (self.model.geom_contype[self.stove] & self.model.geom_conaffinity[self.finger.finger])
                or (self.model.geom_contype[self.finger.finger] & self.model.geom_conaffinity[self.stove])
            )
            or _joints(self.model, _body_path(self.model, int(self.model.geom_bodyid[self.stove]))[:-1])
        ):
            raise ValueError("A fixed and enabled stove/finger collision pair is required")
        self.interval = checked_slide_topology(
            self.model,
            int(self.model.geom_bodyid[self.finger.finger]),
            int(self.model.site_bodyid[self.finger.site]),
            self.joint,
            int(mujoco.mjtJoint.mjJNT_SLIDE),
        )
        np.testing.assert_array_equal(self.interval, config["finger_joint_range_m"])
        np.testing.assert_allclose(self.model.geom_size[self.stove], config["stove_half_size_m"], atol=1e-12, rtol=0)
        self.boxes = [
            (
                "table",
                np.asarray(config["table_pose"], dtype=float).copy(),
                np.asarray(config["table_half_size_m"], dtype=float).copy(),
            ),
            (
                "stove",
                np.asarray(config["stove_pose"], dtype=float).copy(),
                np.asarray(config["stove_half_size_m"], dtype=float).copy(),
            ),
        ]
        self.capture()

    def capture(self):
        """Use already-forwarded caches without stepping or refreshing physics."""
        sim, data = self.env.sim, self.env.sim.data._data
        before = np.asarray(sim.get_state().flatten()).copy()
        current = self.finger.capture()
        stove = transform(data.geom_xpos[self.stove].copy(), data.geom_xmat[self.stove].reshape(3, 3).copy())
        np.testing.assert_allclose(stove, self.boxes[1][1], atol=1e-12, rtol=0)
        axis_world = np.asarray(data.xaxis[self.joint], dtype=float).copy()
        prefix.require(
            np.isclose(np.linalg.norm(axis_world), 1.0, atol=1e-12, rtol=0), "Native slide axis must be unit"
        )
        axis_tcp = current["tcp"][:3, :3].T @ axis_world
        current.update(
            joint_position_m=float(data.qpos[int(self.model.jnt_qposadr[self.joint])]), slide_axis_tcp=axis_tcp
        )
        np.testing.assert_array_equal(before, sim.get_state().flatten())
        return current


def slide_sample_conditions(reference, current, selected_tcp, joint_range):
    """Record measured bound conditions; numerical residuals are not certificates."""
    q0, q = reference["joint_position_m"], current["joint_position_m"]
    lower, upper = finite(joint_range, (2,))
    if not lower <= q0 <= upper:
        raise ValueError("The decision's current joint must lie in its declared interval")
    shape_bound = float(max(q0 - lower, upper - q0))
    measured = observed_bound_conditions(
        reference["vertices_tcp"],
        current["vertices_tcp"],
        selected_tcp,
        current["tcp"],
        position_bound=0.01,
        rotation_bound=0.05,
        shape_bound=shape_bound,
    )
    axis0, axis = finite(reference["slide_axis_tcp"], (3,)), finite(current["slide_axis_tcp"], (3,))
    residual = np.asarray(current["vertices_tcp"]) - np.asarray(reference["vertices_tcp"]) - (q - q0) * axis0
    return {
        **measured,
        "finger_joint_position_m": float(q),
        "finger_joint_range_met": bool(lower <= q <= upper),
        "slide_axis_change_tcp": float(np.linalg.norm(axis - axis0)),
        "slide_vertex_residual_m": float(np.max(np.linalg.norm(residual, axis=1))),
        "scope": "measured sample only; numerical residuals do not establish exact shape or path clearance",
    }


class BoxFaceInnerRecorder:
    """Record geometry before integration and contact/torque after the solver."""

    def __init__(self, env, sensor, *, first_step=120, periods=6, base=None):
        if type(first_step) is not int or first_step < 0 or type(periods) is not int or not 1 <= periods <= 8:
            raise ValueError("A nonnegative start and bounded feedback period count are required")
        self.env, self.sensor, self.first_step, self.periods = env, sensor, first_step, periods
        self.base = ReplayRecorder(env, periods) if base is None else base
        self.original, self.samples = self.base.original, self.base.samples
        self.context = None

    def set_context(self, step, selected_tcp, current):
        if (
            type(step) is not int
            or step != self.first_step + len(self.samples) // 25
            or len(self.samples) % 25
            or len(self.samples) >= 25 * self.periods
        ):
            raise ValueError("A fresh decision is required at each registered physical period")
        self.context = {"step": step, "selected_tcp": np.asarray(selected_tcp, dtype=float).copy(), "current": current}

    def step(self, *args, **kwargs):
        if self.context is None or self.context["step"] != self.first_step + len(self.samples) // 25:
            raise ValueError("No current decision has been registered")
        before = self.sensor.capture()
        controller, gripper = self.env.robots[0].controller, self.env.robots[0].gripper
        goal_pos, goal_ori = np.asarray(controller.goal_pos).copy(), np.asarray(controller.goal_ori).copy()
        gripper_action = np.asarray(gripper.current_action).copy()
        conditions = slide_sample_conditions(
            self.context["current"], before, self.context["selected_tcp"], self.sensor.interval
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
        )
        return value
