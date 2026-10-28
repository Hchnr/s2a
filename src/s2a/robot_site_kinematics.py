"""Robot-chain-only compiled XML FK and world geometric Jacobian."""

from functools import lru_cache

import numpy as np
from scipy.spatial.transform import Rotation

from s2a.geometry import finite, transform
from s2a.xml_site_kinematics import XMLSiteKinematics, orientation, vector


class RobotSiteKinematics:
    """Use only named robot q and fixed geometry; no object or future state input."""

    def __init__(self, xml, site_name, joint_names):
        source = XMLSiteKinematics(xml, site_name)
        names = tuple(joint_names)
        if not names or len(set(names)) != len(names):
            raise ValueError("Unique named robot joints required")
        self.joints = len(names)
        lookup = {name: index for index, name in enumerate(names)}
        self.segments = []
        found = []
        for body in source.chain:
            joints = []
            for node in body:
                if node.tag not in ("joint", "freejoint"):
                    continue
                name = node.get("name")
                if name not in lookup:
                    raise ValueError("Every moving ancestor must be an explicitly supplied robot joint")
                kind = source.indices[node][2]
                axis = vector(node, "axis", "0 0 1")
                axis = axis / np.linalg.norm(axis)
                joints.append((lookup[name], kind, axis, vector(node, "pos", "0 0 0"), float(node.get("ref", "0"))))
                found.append(name)
            self.segments.append((vector(body, "pos", "0 0 0"), orientation(body), joints))
        if set(found) != set(names) or len(found) != len(names):
            raise ValueError("Named robot joints must occur exactly once on SITE ancestry")
        self.site_position = vector(source.site, "pos", "0 0 0")
        self.site_rotation = orientation(source.site)
        self.queries = self.evaluations = 0

    @lru_cache(maxsize=32768)
    def _evaluate(self, coordinates):
        self.evaluations += 1
        p, r, axes = np.zeros(3), np.eye(3), []
        for position, rotation, joints in self.segments:
            p, r = p + r @ position, r @ rotation
            for index, kind, local, pivot, reference in joints:
                axis, origin = r @ local, p + r @ pivot
                axes.append((index, kind, axis, origin))
                q = coordinates[index] - reference
                if kind == "hinge":
                    change = Rotation.from_rotvec(local * q).as_matrix()
                    p, r = p + r @ pivot - r @ change @ pivot, r @ change
                else:
                    p = p + axis * q
        position = p + r @ self.site_position
        jac = np.zeros((6, self.joints))
        for index, kind, axis, origin in axes:
            jac[:3, index] = np.cross(axis, position - origin) if kind == "hinge" else axis
            if kind == "hinge":
                jac[3:, index] = axis
        return transform(position, r @ self.site_rotation), jac

    def __call__(self, q):
        self.queries += 1
        pose, jac = self._evaluate(tuple(finite(q, (self.joints,))))
        return pose.copy(), jac.copy()
