"""Independent fixed-base hinge/slide site kinematics from compiled MuJoCo XML."""

import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation


def vector(node, name, default):
    return np.fromstring(node.get(name, default), sep=" ")


def orientation(node):
    if any(key in node.attrib for key in ("euler", "axisangle", "xyaxes", "zaxis")):
        raise ValueError("Compiled quaternion XML required")
    w, x, y, z = vector(node, "quat", "1 0 0 0")
    return Rotation.from_quat([x, y, z, w]).as_matrix()


class XMLSiteKinematics:
    def __init__(self, xml, site_name):
        root = ET.fromstring(xml)
        compiler = root.find("compiler")
        if compiler is None or compiler.get("angle") != "radian":
            raise ValueError("Explicit compiled radian XML required")
        world = root.find("worldbody")
        parents = {child: parent for parent in world.iter() for child in parent}
        sites = [n for n in world.iter("site") if n.get("name") == site_name]
        if len(sites) != 1:
            raise ValueError("Exactly one controller site required")
        self.site = sites[0]
        self.chain = []
        node = parents[self.site]
        while node is not world:
            if node.tag != "body":
                raise ValueError("Body ancestry required")
            self.chain.append(node)
            node = parents[node]
        self.chain.reverse()
        self.indices = {}
        self.nq = self.nv = 0
        for node in world.iter():
            if node.tag not in ("joint", "freejoint"):
                continue
            kind = "free" if node.tag == "freejoint" else node.get("type", "hinge")
            self.indices[node] = (self.nq, self.nv, kind)
            a, b = {"free": (7, 6), "ball": (4, 3), "hinge": (1, 1), "slide": (1, 1)}[kind]
            self.nq += a
            self.nv += b
        for body in self.chain:
            for node in body:
                if node.tag in ("joint", "freejoint") and self.indices[node][2] not in ("hinge", "slide"):
                    raise ValueError("Fixed-base scalar joint chain required")

    def endpoint(self, state):
        state = np.asarray(state, dtype=float)
        if state.shape != (1 + self.nq + self.nv,) or not np.isfinite(state).all():
            raise ValueError("Complete finite raw time/qpos/qvel state required")
        q = state[1 : 1 + self.nq]
        v = state[1 + self.nq :]
        p = np.zeros(3)
        r = np.eye(3)
        joints = []
        for body in self.chain:
            p = p + r @ vector(body, "pos", "0 0 0")
            r = r @ orientation(body)
            for node in body:
                if node.tag not in ("joint", "freejoint"):
                    continue
                qi, vi, kind = self.indices[node]
                local = vector(node, "axis", "0 0 1")
                local = local / np.linalg.norm(local)
                pivot = vector(node, "pos", "0 0 0")
                axis = r @ local
                origin = p + r @ pivot
                joints.append((axis, origin, v[vi], kind))
                coordinate = q[qi] - float(node.get("ref", "0"))
                if kind == "hinge":
                    change = Rotation.from_rotvec(local * coordinate).as_matrix()
                    p = p + r @ pivot - r @ change @ pivot
                    r = r @ change
                else:
                    p = p + axis * coordinate
        site_position = p + r @ vector(self.site, "pos", "0 0 0")
        site_rotation = r @ orientation(self.site)
        linear = np.zeros(3)
        angular = np.zeros(3)
        for axis, origin, speed, kind in joints:
            if kind == "hinge":
                linear += np.cross(axis, site_position - origin) * speed
                angular += axis * speed
            else:
                linear += axis * speed
        pose = np.eye(4)
        pose[:3, :3] = site_rotation
        pose[:3, 3] = site_position
        return pose, np.r_[linear, angular]
