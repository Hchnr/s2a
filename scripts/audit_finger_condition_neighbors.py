"""188 actual training intervention condition neighbors for three unseen anchors."""

import numpy as np
from arm_joint_wrench_independent import INPUTS, conditions
from arm_wrench_data import attach_current_wrench
from audit_wam_position import Evidence
from scipy.spatial.transform import Rotation

from s2a.artifacts import REPO
from s2a.durable_stage_supervisor import publish
from s2a.joint_wrench_condition_forward import with_history


def main():
    out = REPO / "outputs/finger_condition_neighbors_20261025"
    if out.exists():
        raise FileExistsError(out)
    e = Evidence()
    e.read(__file__)
    e.read("plans/v0.0.1/188_finger_condition_neighbors_20261025.md")
    equation = e.json("outputs/left_table_forward_equations_20261025/result.json")
    label = e.json("outputs/arm_finger_response_labels_20261025/result.json")
    for ref in (equation["training_ref"], label["labels_ref"]):
        e.check(ref["path"], ref["sha256"])
    with np.load(equation["training_ref"]["path"], allow_pickle=False) as a:
        x = a["features"][4865:].copy()
    with np.load(label["labels_ref"]["path"], allow_pickle=False) as a:
        targets = a["fit_target"][4865:].copy()
    train = []
    slopes = []
    identities = []
    for group in range(69):
        phis = []
        for axis in range(3):
            row = x[3 * group + axis]
            carrier = row[282 * (axis + 1) : 282 * (axis + 2)]
            assert abs(carrier[0]) > 1e-12
            phi = carrier / carrier[0]
            phis.append(np.r_[phi[1:84], phi[212:218]])
        np.testing.assert_allclose(phis, np.tile(phis[0], (3, 1)), atol=1e-9, rtol=0)
        train.append(phis[0])
        slopes.append(targets[3 * group : 3 * group + 3])
        identities.append({k: equation["contrast_sources"][3 * group][k] for k in ("dataset", "anchor_group")})
    train = np.asarray(train)
    config = e.json("configs/left_table_forward_20261025.json")
    parameters = e.json("outputs/left_table_fit_20261025/parameters.json")
    source = e.json(config["data_result_ref"]["path"])
    paired = e.json(config["paired_closure_ref"]["path"])
    summary = e.json(paired["summary_ref"]["path"])
    registration = e.json("configs/paired_arm_response_20261025.json")
    measured = e.json("outputs/arm_finger_local_response_20261025/result.json")
    results = []
    for group in config["validation_anchor_groups"]:
        profile = registration["profiles"][group // 2]
        entry = next(p for p in source["trajectories"] if p["source"]["sha256"] == profile["trajectory_ref"]["sha256"])
        ref = entry["cache_ref"]
        e.check(ref["path"], ref["sha256"])
        with np.load(ref["path"], allow_pickle=False) as a:
            data = with_history({k: a[k].copy() for k in a.files}, parameters)
        data = attach_current_wrench(data, entry, config, e)
        anchor = registration["trials"][7 * group]["anchor_cycle"]
        row = {k: data[k][anchor : anchor + 1].copy() for k in INPUTS}
        ref = summary["cases"][7 * group]["arrays_ref"]
        e.check(ref["path"], ref["sha256"])
        with np.load(ref["path"], allow_pickle=False) as a:
            row["action_command"][0] = a["sent_commands"][-1]
            current = row["x_site_pose"][0]
            row["action_goal_delta"][0, :3] = a["controller_goal_pos"][-1] - current[:3, 3]
            row["action_goal_delta"][0, 3:] = Rotation.from_matrix(
                a["controller_goal_ori"][-1] @ current[:3, :3].T
            ).as_rotvec()
        z = (conditions(row)[0] - parameters["state_mean"]) / parameters["state_scale"]
        actual = np.asarray(measured["results"][group]["measured"])
        metrics = {}
        for name, columns in [("all89", np.arange(89)), ("finger_motor", np.arange(43, 51))]:
            distances = np.linalg.norm(train[:, columns] - z[columns], axis=1)
            nearest = np.argsort(distances, kind="stable")[:5]
            metrics[name] = [
                dict(
                    identities[int(i)],
                    distance=float(distances[i]),
                    measured_slope=np.asarray(slopes[i]).tolist(),
                    slope_disagreement=float(np.linalg.norm(slopes[i] - actual)),
                )
                for i in nearest
            ]
        results.append(dict(group=group, measured_slope=actual.tolist(), neighbors=metrics))
    e.recheck()
    publish(
        out / "verified.json",
        dict(complete=True, results=results, source_sha256=e.hashes, fits=0, physics_steps=0, control_qualified=False),
    )
    for r in results:
        print(r)


if __name__ == "__main__":
    main()
