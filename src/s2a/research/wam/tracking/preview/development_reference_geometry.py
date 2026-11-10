"""Saved-state, static contact geometry for development references."""

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from s2a.geometry import inverse, transform
from s2a.infrastructure.snapshots.xml_asset_snapshot import _encoded_path, _file_attributes, _replace


def migrate_xml(raw, asset_paths):
    """Replace only registered file attributes and prove an exact byte inverse."""
    attributes = _file_attributes(raw)
    declared = {value for _, _, value in attributes}
    if declared != set(asset_paths):
        raise ValueError("XML asset declarations differ from the fixed snapshot")
    migrated = _replace(raw, attributes, [_encoded_path(asset_paths[value]) for _, _, value in attributes])
    remapped = _file_attributes(migrated)
    if len(remapped) != len(attributes):
        raise ValueError("XML asset reference count changed")
    inverse_bytes = _replace(migrated, remapped, [_encoded_path(value) for _, _, value in attributes])
    if inverse_bytes != raw:
        raise ValueError("XML migration is not byte-reversible")
    return migrated


def first_closure(openings):
    values = np.asarray(openings, dtype=float)
    if values.shape != (40,) or not np.all(np.isfinite(values)) or np.any((values < 0) | (values > 1)):
        raise ValueError("A finite generated 40-point opening trajectory in [0, 1] is required")
    indices = np.flatnonzero(values < 0.35)
    return None if not len(indices) else int(indices[0])


def contact_flags(contacts, object_geoms, pads):
    """Contact records are (geom1, geom2, distance), not contact forces."""
    if len(pads) != 2 or pads[0] == pads[1] or not object_geoms:
        raise ValueError("Distinct pads and a nonempty target collision group are required")
    flags = [False, False]
    for left, right, distance in contacts:
        pair = {left, right}
        if distance <= 0 and pair & object_geoms:
            for index, pad in enumerate(pads):
                flags[index] |= pad in pair
    return flags


class StaticReferenceProbe:
    """Move a fixed base for an offline geometry query; never advances time."""

    def __init__(self, model, target, budget=1201):
        self.model = model
        self.data = mujoco.MjData(model)
        self.budget = budget
        self.queries = 0
        self.base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot0_base")
        self.site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripper0_grip_site")
        if self.base < 0 or self.site < 0 or model.body_parentid[self.base] != 0 or model.body_jntnum[self.base] != 0:
            raise ValueError("Expected a fixed robot base under world and the grip site")
        names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or "" for i in range(model.ngeom)]
        self.object_geoms = {
            i
            for i, name in enumerate(names)
            if name.startswith(target + "_") and (model.geom_contype[i] or model.geom_conaffinity[i])
        }
        self.pads = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"gripper0_finger{k}_pad_collision") for k in (1, 2)
        ]
        ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"gripper0_finger_joint{k}") for k in (1, 2)]
        if not self.object_geoms or min(self.pads) < 0 or min(ids) < 0 or self.pads[0] == self.pads[1]:
            raise ValueError("Target collisions, finger pads and finger joints must all exist")
        self.qids = [int(model.jnt_qposadr[i]) for i in ids]
        self.object_body = int(model.geom_bodyid[min(self.object_geoms)])
        self.base_position = model.body_pos[self.base].copy()
        self.base_quaternion = model.body_quat[self.base].copy()

    def forward(self):
        if self.queries >= self.budget:
            raise RuntimeError("The registered per-case query budget is exhausted")
        self.queries += 1
        mujoco.mj_forward(self.model, self.data)

    def restore(self, state):
        state = np.asarray(state, dtype=float)
        m, d = self.model, self.data
        if state.shape != (1 + m.nq + m.nv + m.na,) or not np.all(np.isfinite(state)):
            raise ValueError("Complete finite native state required")
        m.body_pos[self.base] = self.base_position
        m.body_quat[self.base] = self.base_quaternion
        d.time = state[0]
        d.qpos[:] = state[1 : 1 + m.nq]
        d.qvel[:] = state[1 + m.nq : 1 + m.nq + m.nv]
        if m.na:
            d.act[:] = state[-m.na :]
        self.forward()

    def contacts(self):
        return contact_flags(
            ((int(c.geom1), int(c.geom2), float(c.dist)) for c in self.data.contact[: self.data.ncon]),
            self.object_geoms,
            self.pads,
        )

    def site_transform(self):
        return transform(self.data.site_xpos[self.site].copy(), self.data.site_xmat[self.site].reshape(3, 3).copy())

    def place(self, desired):
        desired = np.asarray(desired, dtype=float)
        if desired.shape != (4, 4) or not np.all(np.isfinite(desired)):
            raise ValueError("Finite homogeneous target required")
        d, m = self.data, self.model
        old_base = transform(d.xpos[self.base].copy(), d.xmat[self.base].reshape(3, 3).copy())
        new_base = desired @ inverse(self.site_transform()) @ old_base
        old_object_position = d.xpos[self.object_body].copy()
        old_object_rotation = d.xmat[self.object_body].copy()
        m.body_pos[self.base] = new_base[:3, 3]
        xyzw = Rotation.from_matrix(new_base[:3, :3]).as_quat()
        m.body_quat[self.base] = xyzw[[3, 0, 1, 2]]
        self.forward()
        np.testing.assert_allclose(self.site_transform(), desired, atol=1e-9, rtol=0)
        np.testing.assert_array_equal(d.xpos[self.object_body], old_object_position)
        np.testing.assert_array_equal(d.xmat[self.object_body], old_object_rotation)

    def set_opening(self, opening):
        if not np.isfinite(opening) or not 0 <= opening <= 1:
            raise ValueError("Opening must be finite and within [0, 1]")
        gap = 0.001 + 0.079 * opening
        self.data.qpos[self.qids] = [gap / 2, -gap / 2]
        self.forward()
