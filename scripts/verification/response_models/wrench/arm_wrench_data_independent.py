"""104 independent local-sensor-to-world reconstruction on qualified native rows."""

import numpy as np


def attach_current_wrench(data, entry, config, e):
    link = config["wrench_closure_ref"]
    e.check(link["path"], link["sha256"])
    contract = e.json(link["path"])
    assert contract["complete"] and contract["data_qualified"] and contract["verified_trajectories"] == 30
    assert contract["owner_normal_wait_verified"] and contract["old_six_anchor_prefixes_matched"]
    summary_link = contract["summary_ref"]
    e.check(summary_link["path"], summary_link["sha256"])
    rows = e.json(summary_link["path"])["trajectories"]
    row = next(r for r in rows if r["source"]["sha256"] == entry["source"]["sha256"])
    assert row["index"] == entry["index"] and row["split"] == entry["split"]
    source = row["wrench_ref"]
    e.check(source["path"], source["sha256"])
    recorded = e.json(source["path"])
    assert recorded["timing"] == "after original endpoint refresh, before next command"
    points = recorded["samples"]
    count = len(data["x_site_pose"])
    assert len(points) == count + 1 and row["physics_steps"] == count
    values = []
    for k in range(count):
        item = points[k]
        assert item["readonly_verified"]
        np.testing.assert_allclose(item["time"], data["time"][k], rtol=0, atol=1e-10)
        r = np.asarray(item["sensor_pose"])[:3, :3]
        center = np.asarray(item["sensor_pose"])[:3, 3]
        local = np.asarray(item["local"])
        point = data["x_site_pose"][k, :3, 3]
        np.testing.assert_allclose(point, item["control_site_position"], rtol=0, atol=1e-9)
        force = np.array([sum(r[i, j] * local[j] for j in range(3)) for i in range(3)])
        moment = np.array([sum(r[i, j] * local[3 + j] for j in range(3)) for i in range(3)])
        value = np.r_[force, moment + np.cross(center - point, force)]
        np.testing.assert_allclose(value, item["world_at_control_site"], rtol=0, atol=1e-10)
        values.append(value)
    values = np.asarray(values)
    assert values.shape == (count, 6) and np.isfinite(values).all()
    output = dict(data)
    output["x_current_wrench_world"] = values
    return output
