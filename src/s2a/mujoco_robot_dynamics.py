"""Isolated robot quantities from XML, without contacts or physics integration."""

import numpy as np

from s2a.geometry import finite, transform


class MujocoRobotDynamics:
    """Hold explicitly named gripper coordinates at their current values.

    Other bodies keep XML defaults; their observed or future states are never
    accepted. Only low-level robot kinematics/inertia/passive queries run, not
    mj_forward, a constraint solver or mj_step. The arm mass subblock does not
    model acceleration coupling to the omitted gripper DOFs. Fixed gripper
    velocities are a current-state approximation, not integrated predictions.
    """

    def __init__(self, xml_path, site_name, robot_joint_names, fixed_joint_state):
        import mujoco

        self.mj = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        if self.model.ntendon:
            raise ValueError("Tendon models require additional kinematic/velocity stages and are unsupported")
        self.data = mujoco.MjData(self.model)
        names = tuple(robot_joint_names)
        if len(set(names)) != len(names) or set(names) & set(fixed_joint_state):
            raise ValueError("Distinct moving and fixed joint names required")

        def joint(name):
            index = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if index < 0 or self.model.jnt_type[index] not in (
                mujoco.mjtJoint.mjJNT_HINGE,
                mujoco.mjtJoint.mjJNT_SLIDE,
            ):
                raise ValueError("Named scalar robot/gripper joint required")
            return index

        self.joints = np.asarray([joint(name) for name in names])
        self.qids, self.dofs = self.model.jnt_qposadr[self.joints], self.model.jnt_dofadr[self.joints]
        for name, state in fixed_joint_state.items():
            index = joint(name)
            position, velocity = finite(state, (2,))
            self.data.qpos[self.model.jnt_qposadr[index]] = position
            self.data.qvel[self.model.jnt_dofadr[index]] = velocity
        self.sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if self.sid < 0:
            raise ValueError("Named controller SITE required")
        self.queries = 0

    def __call__(self, q, velocity):
        mj, model, data = self.mj, self.model, self.data
        n = len(self.joints)
        data.qpos[self.qids], data.qvel[self.dofs] = finite(q, (n,)), finite(velocity, (n,))
        mj.mj_kinematics(model, data)
        mj.mj_comPos(model, data)
        mj.mj_crb(model, data)
        mj.mj_comVel(model, data)
        mj.mj_rne(model, data, 0, data.qfrc_bias)
        mj.mj_passive(model, data)
        mass = np.zeros((model.nv, model.nv))
        mj.mj_fullM(model, mass, data.qM)
        jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
        mj.mj_jacSite(model, data, jp, jr, self.sid)
        damping = model.dof_damping[self.dofs].copy()
        self.queries += 1
        return dict(
            mass=mass[np.ix_(self.dofs, self.dofs)],
            jacobian=np.vstack((jp, jr))[:, self.dofs],
            pose=transform(data.site_xpos[self.sid].copy(), data.site_xmat[self.sid].reshape(3, 3).copy()),
            bias=data.qfrc_bias[self.dofs].copy(),
            damping=damping,
            passive_extra=data.qfrc_passive[self.dofs].copy() + damping * data.qvel[self.dofs],
        )
