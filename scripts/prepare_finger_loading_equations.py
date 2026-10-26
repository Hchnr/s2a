"""193 merge9 new aperture intervention equations only after full191 closure."""

from pathlib import Path

import numpy as np
from arm_joint_wrench_independent import INPUTS, feature_matrix
from arm_wrench_data import attach_current_wrench
from audit_wam_position import Evidence
from scipy.spatial.transform import Rotation
from verify_arm_finger_response_labels import aperture

from s2a.artifacts import REPO
from s2a.durable_stage_supervisor import publish, sha
from s2a.joint_wrench_condition_forward import design, with_history


def main():
    out = REPO / "outputs/finger_loading_equations_20261025"
    if out.exists():
        raise FileExistsError(out)
    e = Evidence()
    e.read(__file__)
    e.read("plans/v0.0.1/193_finger_loading_equations_20261025.md")
    closed = e.json("outputs/finger_loading_interventions_20261025/closure_verified.json")
    assert (
        closed["complete"]
        and closed["data_qualified"]
        and closed["owner_normal_wait_verified"]
        and closed["verified_cases"] == 21
    )
    for p, h in closed["source_sha256"].items():
        e.check(p, h)
    summary = e.json(closed["summary_ref"]["path"])
    assert summary["explicit_integration_substeps"] == 75075
    registration = e.json("configs/finger_loading_interventions_20261025.json")
    config = e.json("configs/left_table_forward_20261025.json")
    parameters = e.json("outputs/left_table_fit_20261025/parameters.json")
    native = e.json(config["data_result_ref"]["path"])
    previous = e.json("outputs/left_table_forward_equations_20261025/result.json")
    old_labels = e.json("outputs/arm_finger_response_labels_20261025/result.json")
    for ref in (previous["training_ref"], old_labels["labels_ref"]):
        e.check(ref["path"], ref["sha256"])
    with np.load(previous["training_ref"]["path"], allow_pickle=False) as a:
        old_x = a["features"].copy()
        old_w = a["weights"].copy()
    with np.load(old_labels["labels_ref"]["path"], allow_pickle=False) as a:
        old_y = a["fit_target"].copy()
    features = []
    target = []
    records = []
    for group, profile in enumerate(registration["profiles"]):
        entry = next(p for p in native["trajectories"] if p["source"] == profile["trajectory_ref"])
        assert entry["split"] == "fit" and entry["index"] not in (0, 10, 20)
        anchor = registration["trials"][7 * group]["anchor_cycle"]
        ref = entry["cache_ref"]
        e.check(ref["path"], ref["sha256"])
        with np.load(ref["path"], allow_pickle=False) as a:
            data = with_history({k: a[k].copy() for k in a.files}, parameters)
        data = attach_current_wrench(data, entry, config, e)
        condition = {k: data[k][anchor : anchor + 1].copy() for k in INPUTS}
        packets = []
        for branch in range(7):
            ref = summary["cases"][7 * group + branch]["arrays_ref"]
            e.check(ref["path"], ref["sha256"])
            xml = e.read(Path(ref["path"]).parent / "model.xml").decode()
            with np.load(ref["path"], allow_pickle=False) as a:
                q = a["gripper_qpos"][-2:].copy()
                np.testing.assert_allclose(q[0], condition["x_finger_qpos"][0], atol=1e-12, rtol=0)
                observed = (q[1, 0] - q[1, 1]) - (q[0, 0] - q[0, 1])
                raw = aperture(a["sim_states"][-2:], xml)
                np.testing.assert_allclose(observed, raw[1] - raw[0], atol=1e-12, rtol=0)
                row = {k: v.copy() for k, v in condition.items()}
                command = a["sent_commands"][-1].copy()
                row["action_command"][0] = command
                current = condition["x_site_pose"][0]
                row["action_goal_delta"][0, :3] = a["controller_goal_pos"][-1] - current[:3, 3]
                row["action_goal_delta"][0, 3:] = Rotation.from_matrix(
                    a["controller_goal_ori"][-1] @ current[:3, :3].T
                ).as_rotvec()
            x = design(row, parameters)[0]
            independent = feature_matrix(row, parameters)[0]
            np.testing.assert_allclose(x, independent, atol=1e-9, rtol=0)
            packets.append((command, x, float(observed)))
        for axis in range(3):
            minus, plus = 1 + 2 * axis, 2 + 2 * axis
            delta = packets[plus][0][axis] - packets[minus][0][axis]
            assert delta > 0
            row = (packets[plus][1] - packets[minus][1]) / delta
            assert np.max(np.abs(row[:282])) == 0
            features.append(row)
            target.append((packets[plus][2] - packets[minus][2]) / delta)
            records.append(
                dict(
                    dataset="191",
                    anchor_group=group,
                    axis=axis,
                    native_index=entry["index"],
                    anchor_cycle=anchor,
                    actual_normalized_delta=float(delta),
                )
            )
    x = np.vstack([old_x, features])
    y = np.r_[old_y, target]
    w = np.r_[old_w[:4865], np.full(216, 0.5 / 216)]
    assert x.shape == (5081, 1128) and y.shape == w.shape == (5081,)
    np.testing.assert_allclose([w[:4865].sum(), w[4865:].sum()], [0.5, 0.5], atol=1e-12, rtol=0)
    e.recheck()
    out.mkdir()
    path = out / "training.npz"
    with path.open("xb") as stream:
        np.savez_compressed(stream, features=x, target=y, weights=w)
    publish(
        out / "result.json",
        dict(
            complete=True,
            rows=5081,
            natural_rows=4865,
            contrast_equations=216,
            new_contrasts=records,
            training_ref=dict(path=str(path), sha256=sha(path)),
            source_sha256=e.hashes,
            fits=0,
            physics_steps=0,
            control_qualified=False,
        ),
    )
    print("193 qualified191 merged9 equations;5081rows;no fit/control claim")


if __name__ == "__main__":
    main()
