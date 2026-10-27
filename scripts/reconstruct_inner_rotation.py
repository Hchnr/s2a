"""119 reconstruct existing probe inner rotation path using independent XML FK."""

from pathlib import Path

import numpy as np
from audit_wam_position import Evidence
from collect_wam_native_training import write_once
from scipy.spatial.transform import Rotation

from s2a.xml_site_kinematics import XMLSiteKinematics


def main():
    e = Evidence()
    e.read(__file__)
    e.read("src/s2a/xml_site_kinematics.py")
    e.read("plans/v0.0.1/119_inner_rotation_reconstruction_20261025.md")
    root = Path("outputs/contact_probe_trace_20261025")
    c = e.json(root / "closure_verified.json")
    assert c["complete"] and c["owner_normal_wait_verified"]
    original = Path("outputs/paired_arm_response_20261025")
    closed = e.json(original / "closure_verified.json")
    assert closed["complete"] and closed["data_qualified"]
    rows = []
    for i in range(6):
        path = root / f"group{i:02d}/collection_result.json"
        e.check(path, c["source_sha256"][str(path.resolve())])
        r = e.json(path)
        for key in ("trace_ref", "arrays_ref", "baseline_ref"):
            ref = r[key]
            e.check(ref["path"], ref["sha256"])
        source = Path(r["baseline_ref"]["path"]).parent
        p = source / "collection_result.json"
        e.check(p, closed["source_sha256"][str(p.resolve())])
        src = e.json(p)
        xml = source / "model.xml"
        e.check(xml, src["source_sha256"][str(xml.resolve())])
        kin = XMLSiteKinematics(xml.read_bytes(), src["controller_contract"]["controller_site"])
        joints = {n.get("name"): indices for n, indices in kin.indices.items()}
        indices = [joints[f"robot0_joint{j}"] for j in range(1, 8)]
        assert all(kind == "hinge" for _, _, kind in indices)
        records = e.json(r["trace_ref"]["path"])["records"]
        assert len(records) == 25
        with np.load(r["arrays_ref"]["path"], allow_pickle=False) as a:
            initial = a["sim_states"][-2].copy()
            start_pose, end_pose = a["controller_site_pose"][-2:].copy()
            start_twist, end_twist = a["source_twist"][-2:].copy()
            end_time = float(a["sim_states"][-1, 0])
        poses, twists, times = [], [], []
        for rec in records:
            state = initial.copy()
            state[0] = rec["time"]
            for j, (qi, vi, _) in enumerate(indices):
                state[1 + qi] = rec["joint_pos"][j]
                state[1 + kin.nq + vi] = rec["joint_vel"][j]
            pose, twist = kin.endpoint(state)
            poses.append(pose)
            twists.append(twist)
            times.append(rec["time"])
        np.testing.assert_allclose(poses[0], start_pose, rtol=0, atol=1e-9)
        np.testing.assert_allclose(twists[0], start_twist, rtol=0, atol=1e-9)
        np.testing.assert_allclose(times[0], initial[0], rtol=0, atol=1e-10)
        poses.append(end_pose)
        twists.append(end_twist)
        times.append(end_time)
        np.testing.assert_allclose(np.diff(times), 0.002, rtol=0, atol=1e-10)
        product = np.eye(3)
        integrated = start_pose[:3, :3].copy()
        for k, dt in enumerate(np.diff(times)):
            product = (poses[k + 1][:3, :3] @ poses[k][:3, :3].T) @ product
            omega = 0.5 * (twists[k][3:] + twists[k + 1][3:])
            integrated = Rotation.from_rotvec(dt * omega).as_matrix() @ integrated
        np.testing.assert_allclose(product, end_pose[:3, :3] @ start_pose[:3, :3].T, rtol=0, atol=1e-12)
        rows.append(
            dict(
                probe=i,
                original_case=source.name,
                records=25,
                duration=times[-1] - times[0],
                trapezoidal_rotation_error=float(Rotation.from_matrix(integrated @ end_pose[:3, :3].T).magnitude()),
                maximum_inner_angular_speed=float(np.max(np.linalg.norm(np.asarray(twists)[:, 3:], axis=1))),
            )
        )
    e.recheck()
    output = Path("outputs/inner_rotation_reconstruction_20261025")
    output.mkdir(exist_ok=False)
    write_once(
        output / "result.json",
        dict(
            complete=True,
            diagnostic_only=True,
            probes=rows,
            source_sha256=e.hashes,
            new_fits=0,
            physics_steps=0,
            model_calls=0,
            fresh_cases=0,
        ),
    )
    print(rows)


if __name__ == "__main__":
    main()
