"""Named robot coordinates exclude unrelated free objects and preserve world J."""

import numpy as np

from s2a.geometry import pose_error
from s2a.robot_site_kinematics import RobotSiteKinematics
from s2a.xml_site_kinematics import XMLSiteKinematics

XML = """<mujoco><compiler angle="radian"/><worldbody>
<body name="object"><freejoint/></body>
<body pos=".5 .2 0" quat=".9238795325112867 0 0 .3826834323650898">
<joint name="j1" axis="0 0 1" pos=".1 0 0"/>
<body pos=".2 .1 0"><joint name="j2" type="slide" axis="1 0 0" ref=".2"/>
<body><joint name="j3" axis="0 1 0" pos="0 .1 0"/>
<site name="tcp" pos=".2 .1 .3" quat=".9800665778412416 .1986693307950612 0 0"/>
</body></body></body></worldbody></mujoco>"""


def test_robot_only_fk_and_jacobian_match_raw_state_velocity_and_pose_differences():
    names = ("j3", "j1", "j2")
    fk = RobotSiteKinematics(XML, "tcp", names)
    full = XMLSiteKinematics(XML, "tcp")
    indices = {node.get("name"): value for node, value in full.indices.items()}
    q = np.array([0.4, 0.3, 0.25])
    state = np.zeros(1 + full.nq + full.nv)
    state[1:8] = [100, -50, 12, 1, 0, 0, 0]
    for name, value in zip(names, q, strict=True):
        state[1 + indices[name][0]] = value
    pose, jac = fk(q)
    np.testing.assert_allclose(pose, full.endpoint(state)[0], atol=1e-15)
    for column, name in enumerate(names):
        moving = state.copy()
        moving[1 + full.nq + indices[name][1]] = 1
        np.testing.assert_allclose(jac[:, column], full.endpoint(moving)[1], atol=1e-15)
        plus, minus = q.copy(), q.copy()
        plus[column] += 1e-6
        minus[column] -= 1e-6
        numeric = pose_error(fk(plus)[0], fk(minus)[0]) / 2e-6
        np.testing.assert_allclose(jac[:, column], numeric, atol=2e-10, rtol=0)
    pose[:] = 0
    jac[:] = 0
    np.testing.assert_allclose(fk(q)[0], full.endpoint(state)[0], atol=1e-15)
