"""Current robot sensor wrench with explicit world frame and moment origin."""

import numpy as np

from s2a.geometry import finite, validate_transform


def world_wrench_at_point(local, sensor_pose, point):
    local = finite(local, (6,))
    sensor_pose = validate_transform(sensor_pose)
    if sensor_pose.shape != (4, 4):
        raise ValueError("Individual world sensor pose required")
    point = finite(point, (3,))
    force = sensor_pose[:3, :3] @ local[:3]
    torque = sensor_pose[:3, :3] @ local[3:] + np.cross(sensor_pose[:3, 3] - point, force)
    return np.r_[force, torque]


def read_current_wrench(env):
    """Read only already-refreshed sensors; caller must establish endpoint timing."""
    import mujoco

    robot, sim = env.robots[0], env.sim
    model, data = sim.model, sim.data
    before = sim.get_state().flatten().copy()
    ctrl, sensors = np.asarray(data.ctrl).copy(), np.asarray(data.sensordata).copy()
    names = [robot.gripper.important_sensors[k] for k in ("force_ee", "torque_ee")]
    ids = [model.sensor_name2id(n) for n in names]
    assert [int(model.sensor_type[i]) for i in ids] == [
        int(mujoco.mjtSensor.mjSENS_FORCE),
        int(mujoco.mjtSensor.mjSENS_TORQUE),
    ]
    assert all(int(model.sensor_dim[i]) == 3 for i in ids)
    site = int(model.sensor_objid[ids[0]])
    assert site == int(model.sensor_objid[ids[1]])
    expected = np.concatenate([sensors[int(model.sensor_adr[i]) : int(model.sensor_adr[i]) + 3] for i in ids])
    local = np.r_[np.asarray(robot.ee_force).copy(), np.asarray(robot.ee_torque).copy()]
    np.testing.assert_array_equal(local, expected)
    sensor_pose = np.eye(4)
    sensor_pose[:3, :3] = np.asarray(data.site_xmat[site]).reshape(3, 3)
    sensor_pose[:3, 3] = np.asarray(data.site_xpos[site])
    control_site = model.site_name2id(robot.controller.eef_name)
    point = np.asarray(data.site_xpos[control_site]).copy()
    world = world_wrench_at_point(local, sensor_pose, point)
    np.testing.assert_array_equal(before, sim.get_state().flatten())
    np.testing.assert_array_equal(ctrl, data.ctrl)
    np.testing.assert_array_equal(sensors, data.sensordata)
    return dict(
        local=local,
        world_at_control_site=world,
        sensor_pose=sensor_pose,
        control_site_position=point,
        time=float(data.time),
        sensor_names=names,
        sensor_site=model.site_id2name(site),
        control_site=robot.controller.eef_name,
        sensor_types=[int(model.sensor_type[i]) for i in ids],
        sensor_addresses=[int(model.sensor_adr[i]) for i in ids],
        readonly_verified=True,
    )
