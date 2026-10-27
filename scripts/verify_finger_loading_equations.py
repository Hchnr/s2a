"""194 independently reconstruct all9 added equations after full191 qualification."""

from pathlib import Path

import numpy as np
from arm_joint_wrench_independent import INPUTS, augment, feature_matrix
from arm_wrench_data_independent import attach_current_wrench
from audit_wam_position import Evidence
from scipy.spatial.transform import Rotation
from verify_arm_finger_response_labels import aperture

from s2a.artifacts import REPO
from s2a.durable_stage_supervisor import publish


def main():
    output = REPO / "outputs/finger_loading_equations_20261025/independent_verified.json"
    if output.exists():
        raise FileExistsError(output)
    e = Evidence()
    e.read(__file__)
    e.read("plans/v0.0.1/194_finger_loading_equations_verification_20261025.md")
    merged = e.json("outputs/finger_loading_equations_20261025/result.json")
    assert merged["complete"] and merged["rows"] == 5081 and merged["contrast_equations"] == 216
    for p, h in merged["source_sha256"].items():
        e.check(p, h)
    ref = merged["training_ref"]
    e.check(ref["path"], ref["sha256"])
    with np.load(ref["path"], allow_pickle=False) as a:
        x, y, w = (a[k].copy() for k in ("features", "target", "weights"))
    old = e.json("outputs/left_table_forward_equations_20261025/result.json")
    labels = e.json("outputs/arm_finger_response_labels_20261025/result.json")
    for ref in (old["training_ref"], labels["labels_ref"]):
        e.check(ref["path"], ref["sha256"])
    with np.load(old["training_ref"]["path"], allow_pickle=False) as a:
        np.testing.assert_array_equal(x[:5072], a["features"])
        np.testing.assert_array_equal(w[:4865], a["weights"][:4865])
    with np.load(labels["labels_ref"]["path"], allow_pickle=False) as a:
        np.testing.assert_array_equal(y[:5072], a["fit_target"])
    np.testing.assert_allclose(w[4865:], 0.5 / 216, atol=1e-15, rtol=0)
    config = e.json("configs/left_table_forward_20261025.json")
    params = e.json("outputs/left_table_fit_20261025/parameters.json")
    native = e.json(config["data_result_ref"]["path"])
    closure = e.json("outputs/finger_loading_interventions_20261025/closure_verified.json")
    assert closure["data_qualified"] and closure["owner_normal_wait_verified"]
    summary = e.json(closure["summary_ref"]["path"])
    registration = e.json("configs/finger_loading_interventions_20261025.json")
    independent_x = []
    independent_y = []
    for group, profile in enumerate(registration["profiles"]):
        entry = next(p for p in native["trajectories"] if p["source"] == profile["trajectory_ref"])
        assert entry["split"] == "fit" and entry["index"] not in (0, 10, 20)
        ref = entry["cache_ref"]
        e.check(ref["path"], ref["sha256"])
        with np.load(ref["path"], allow_pickle=False) as a:
            data = augment({k: a[k].copy() for k in a.files}, params)
        full = attach_current_wrench(data, entry, config, e)
        anchor = registration["trials"][7 * group]["anchor_cycle"]
        condition = {k: full[k][anchor : anchor + 1].copy() for k in INPUTS}
        for axis in range(3):
            packets = []
            for branch in (1 + 2 * axis, 2 + 2 * axis):
                ref = summary["cases"][7 * group + branch]["arrays_ref"]
                e.check(ref["path"], ref["sha256"])
                xml = e.read(Path(ref["path"]).parent / "model.xml").decode()
                with np.load(ref["path"], allow_pickle=False) as a:
                    gaps = aperture(a["sim_states"][-2:], xml)
                    command = a["sent_commands"][-1].copy()
                    row = {k: v.copy() for k, v in condition.items()}
                    row["action_command"][0] = command
                    current = condition["x_site_pose"][0]
                    row["action_goal_delta"][0, :3] = a["controller_goal_pos"][-1] - current[:3, 3]
                    row["action_goal_delta"][0, 3:] = Rotation.from_matrix(
                        a["controller_goal_ori"][-1] @ current[:3, :3].T
                    ).as_rotvec()
                packets.append((command, feature_matrix(row, params)[0], gaps[1] - gaps[0]))
            delta = packets[1][0][axis] - packets[0][0][axis]
            assert delta > 0
            independent_x.append((packets[1][1] - packets[0][1]) / delta)
            independent_y.append((packets[1][2] - packets[0][2]) / delta)
    np.testing.assert_allclose(x[5072:], independent_x, atol=1e-8, rtol=0)
    np.testing.assert_allclose(y[5072:], independent_y, atol=1e-12, rtol=0)
    e.recheck()
    publish(
        output,
        dict(
            complete=True,
            independent_equations_verified=True,
            rows=5081,
            new_contrasts=9,
            source_sha256=e.hashes,
            physics_steps=0,
            fits=0,
            control_qualified=False,
        ),
    )
    print("194 old5072 preserved;new9 independently verified;no fit/control claim")


if __name__ == "__main__":
    main()
