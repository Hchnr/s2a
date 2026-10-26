"""104 bind current measured wrench only after complete103 data qualification."""

import numpy as np


def attach_current_wrench(data, entry, config, e):
    ref = config["wrench_closure_ref"]
    e.check(ref["path"], ref["sha256"])
    closed = e.json(ref["path"])
    assert closed["complete"] and closed["data_qualified"] and closed["owner_normal_wait_verified"]
    assert closed["verified_trajectories"] == 30 and closed["old_six_anchor_prefixes_matched"]
    ref = closed["summary_ref"]
    e.check(ref["path"], ref["sha256"])
    summary = e.json(ref["path"])
    row = next(r for r in summary["trajectories"] if r["index"] == entry["index"])
    assert row["source"] == entry["source"] and row["split"] == entry["split"]
    n = len(data["x_site_pose"])
    assert row["physics_steps"] == n
    ref = row["wrench_ref"]
    e.check(ref["path"], ref["sha256"])
    payload = e.json(ref["path"])
    assert payload["timing"] == "after original endpoint refresh, before next command"
    samples = payload["samples"]
    assert len(samples) == n + 1
    # Do not use the last future endpoint or any y label as a condition.
    current = samples[:n]
    np.testing.assert_allclose([s["time"] for s in current], data["time"], rtol=0, atol=1e-10)
    np.testing.assert_allclose(
        [s["control_site_position"] for s in current], data["x_site_pose"][:, :3, 3], rtol=0, atol=1e-9
    )
    value = np.asarray([s["world_at_control_site"] for s in current], dtype=float)
    assert value.shape == (n, 6) and np.isfinite(value).all() and all(s["readonly_verified"] for s in current)
    result = dict(data)
    result["x_current_wrench_world"] = value
    return result
