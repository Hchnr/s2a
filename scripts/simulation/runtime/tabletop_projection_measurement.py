"""Read current native finger geometry without forwarding or stepping physics."""

import numpy as np

from s2a.geometry import transform, validate_transform
from s2a.research.joint_control.tracking.tabletop_projection_observation import measured_vertices_in_tcp


class CurrentFingerGeometry:
    """Read a named finger mesh and fixed controller site from the current cache.

    The caller is responsible for obtaining an already refreshed endpoint or
    entering after the environment's own inner-step forward. The calibrated
    table is checked once against this offline scene, never supplied by a live
    object/task observer to the candidate policy.
    """

    def __init__(self, env, *, site_from_tcp, table_pose, table_half_size, table_name, finger_name):
        import mujoco

        self.env, self.model = env, env.sim.model._model
        self.site_from_tcp = validate_transform(np.asarray(site_from_tcp, dtype=float)).copy()
        table = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, table_name)
        self.finger = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, finger_name)
        self.site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, env.robots[0].controller.eef_name)
        if (
            self.site_from_tcp.shape != (4, 4)
            or min(table, self.finger, self.site) < 0
            or self.model.geom_type[table] != mujoco.mjtGeom.mjGEOM_BOX
            or self.model.geom_type[self.finger] != mujoco.mjtGeom.mjGEOM_MESH
            or not (
                (self.model.geom_contype[table] & self.model.geom_conaffinity[self.finger])
                or (self.model.geom_contype[self.finger] & self.model.geom_conaffinity[table])
            )
        ):
            raise ValueError("The fixed controller site and enabled table/finger collision pair are required")
        mesh = int(self.model.geom_dataid[self.finger])
        if mesh < 0:
            raise ValueError("A compiled finger collision mesh is required")
        start, count = int(self.model.mesh_vertadr[mesh]), int(self.model.mesh_vertnum[mesh])
        self.mesh_vertices = np.asarray(self.model.mesh_vert[start : start + count], dtype=float).copy()
        if self.mesh_vertices.shape != (count, 3) or count < 4 or not np.isfinite(self.mesh_vertices).all():
            raise ValueError("A finite compiled collision mesh is required")
        data = env.sim.data._data
        calibrated = validate_transform(np.asarray(table_pose, dtype=float))
        actual_table = transform(data.geom_xpos[table].copy(), data.geom_xmat[table].reshape(3, 3).copy())
        np.testing.assert_allclose(actual_table, calibrated, atol=1e-12, rtol=0)
        np.testing.assert_allclose(self.model.geom_size[table], table_half_size, atol=1e-12, rtol=0)

    def capture(self):
        """Return only measured TCP and finger vertices; leave state unchanged."""
        sim = self.env.sim
        before = np.asarray(sim.get_state().flatten()).copy()
        data = sim.data._data
        site = transform(data.site_xpos[self.site].copy(), data.site_xmat[self.site].reshape(3, 3).copy())
        tcp = site @ self.site_from_tcp
        finger = transform(data.geom_xpos[self.finger].copy(), data.geom_xmat[self.finger].reshape(3, 3).copy())
        vertices = measured_vertices_in_tcp(self.mesh_vertices, tcp, finger)
        np.testing.assert_array_equal(before, sim.get_state().flatten())
        return {"tcp": tcp, "vertices_tcp": vertices, "time_s": float(data.time)}
