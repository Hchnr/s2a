"""Nominal URDF forward kinematics for command diagnostics, without IK/control."""

import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from s2a.data import require
from s2a.geometry import finite, transform


class URDFChain:
    def __init__(self, path, base_link, target_link):
        root = ET.parse(path).getroot()
        parents = {}
        for joint in root.findall("joint"):
            child = joint.find("child").attrib["link"]
            require(child not in parents, "ambiguous URDF child")
            parents[child] = joint
        chain, link, visited = [], target_link, set()
        while link != base_link:
            require(link not in visited and link in parents, "URDF chain missing or cyclic")
            visited.add(link)
            joint = parents[link]
            origin = joint.find("origin")
            xyz = np.fromstring(origin.get("xyz", "0 0 0") if origin is not None else "0 0 0", sep=" ")
            rpy = np.fromstring(origin.get("rpy", "0 0 0") if origin is not None else "0 0 0", sep=" ")
            require(xyz.shape == rpy.shape == (3,), "invalid URDF origin")
            kind = joint.attrib["type"]
            require(kind in {"fixed", "revolute", "continuous", "prismatic"}, "unsupported URDF joint")
            axis_node = joint.find("axis")
            axis = np.fromstring(axis_node.get("xyz", "1 0 0") if axis_node is not None else "1 0 0", sep=" ")
            if kind == "fixed":
                axis = np.array([1.0, 0.0, 0.0])  # fixed-joint axes are unused, and may be zero in valid URDF
            require(axis.shape == (3,) and np.linalg.norm(axis) > 1e-8, "invalid URDF axis")
            chain.append(
                (
                    joint.attrib["name"],
                    kind,
                    transform(xyz, Rotation.from_euler("xyz", rpy).as_matrix()),
                    axis / np.linalg.norm(axis),
                )
            )
            link = joint.find("parent").attrib["link"]
        self.chain = list(reversed(chain))
        self.joint_names = [name for name, kind, _, _ in self.chain if kind != "fixed"]

    def forward(self, positions):
        q = finite(positions)
        require(q.ndim == 2 and q.shape[1] == len(self.joint_names), "joint position layout mismatch")
        result = np.broadcast_to(np.eye(4), (len(q), 4, 4)).copy()
        index = 0
        for _, kind, origin, axis in self.chain:
            result = result @ origin
            if kind == "fixed":
                continue
            motion = np.broadcast_to(np.eye(4), result.shape).copy()
            if kind == "prismatic":
                motion[:, :3, 3] = q[:, index, None] * axis
            else:
                motion[:, :3, :3] = Rotation.from_rotvec(q[:, index, None] * axis).as_matrix()
            result = result @ motion
            index += 1
        return result
