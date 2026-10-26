"""087 independent source/label/API-goal and numerical-FK derivative verification."""

import json
from pathlib import Path

import numpy as np
from audit_wam_position import Evidence, native_delta_rotation
from scipy.spatial.transform import Rotation

from s2a.owned_receipt_contract import normal_receipts
from s2a.xml_site_kinematics import XMLSiteKinematics

ROOT = Path("outputs/native_arm_response_data_20261025")


def main():
    e = Evidence()
    for path in (__file__, "src/s2a/xml_site_kinematics.py", "scripts/audit_wam_position.py"):
        e.read(path)
    config = e.json("configs/native_arm_response_data_20261025.json")
    for path, signature in e.json(ROOT / "source_sha256.json").items():
        e.check(path, signature)
    folder = ROOT / "orchestration/reconstruct"
    normal_receipts(e, folder / "launch.json", folder / "exit.json", folder / "release.json", format="durable")
    reconstructed = e.json(ROOT / "reconstruction_result.json")
    assert reconstructed["complete"] and reconstructed["reconstruction_passed"]
    for path, signature in reconstructed["source_sha256"].items():
        e.check(path, signature)
    assert len(reconstructed["trajectories"]) == len(config["trajectories"]) == 30
    epsilon = config["finite_difference_epsilon_s"]
    tolerance = config["finite_difference_absolute_tolerance"]
    assert epsilon == 1e-6 and tolerance == 1e-7
    metrics = []
    for i, (entry, cached) in enumerate(zip(config["trajectories"], reconstructed["trajectories"], strict=True)):
        assert cached["index"] == i and cached["split"] == entry["split"] and cached["task"] == entry["task"]
        for key in ("ref", "result_ref", "goal_log_ref"):
            ref = entry[key]
            e.check(ref["path"], ref["sha256"])
        ref = cached["cache_ref"]
        e.check(ref["path"], ref["sha256"])
        result = e.json(entry["result_ref"]["path"])
        goals = e.json(entry["goal_log_ref"]["path"])
        contract = result["controller_contract"]
        kin = XMLSiteKinematics(result["record"]["model_xml"], contract["controller_site"])
        by_name = {node.get("name"): (qi, vi) for node, (qi, vi, _) in kin.indices.items()}
        arm = [by_name[f"robot0_joint{k}"] for k in range(1, 8)]
        fingers = [by_name[f"gripper0_finger_joint{k}"] for k in (1, 2)]
        largest_fd = 0.0
        with (
            np.load(entry["ref"]["path"], allow_pickle=False) as source,
            np.load(ref["path"], allow_pickle=False) as data,
        ):
            raw = source["sim_states"]
            n = len(source["sent_commands"])
            native = result["native_steps"]
            assert len(goals) == n and cached["metrics"]["rows"] == n and len(raw) == n + 1
            q = raw[:, 1 : 1 + kin.nq]
            v = raw[:, 1 + kin.nq :]
            analytic = []
            fd = []
            for endpoint_index, state in enumerate(raw):
                pose, twist = kin.endpoint(state)
                analytic.append(twist)
                np.testing.assert_allclose(pose, source["controller_site_pose"][endpoint_index], rtol=0, atol=1e-9)
                left, right = state.copy(), state.copy()
                for qi, vi in arm:
                    perturb = epsilon * state[1 + kin.nq + vi]
                    left[1 + qi] -= perturb
                    right[1 + qi] += perturb
                minus, _ = kin.endpoint(left)
                plus, _ = kin.endpoint(right)
                derivative = np.r_[
                    (plus[:3, 3] - minus[:3, 3]) / (2 * epsilon),
                    Rotation.from_matrix(plus[:3, :3] @ minus[:3, :3].T).as_rotvec() / (2 * epsilon),
                ]
                np.testing.assert_allclose(twist, derivative, rtol=0, atol=tolerance)
                largest_fd = max(largest_fd, float(np.max(np.abs(twist - derivative))))
                fd.append(derivative)
            analytic, fd = np.array(analytic), np.array(fd)
            np.testing.assert_allclose(data["x_site_twist"], fd[:-1], rtol=0, atol=tolerance)
            np.testing.assert_allclose(data["y_twist"], fd[1:], rtol=0, atol=tolerance)
            np.testing.assert_allclose(data["x_site_pose"], source["controller_site_pose"][:-1], rtol=0, atol=1e-9)
            for field, indices in [("joint_position", [a[0] for a in arm]), ("finger_qpos", [a[0] for a in fingers])]:
                np.testing.assert_array_equal(data["x_" + field], q[:-1, indices])
            for field, indices in [("joint_velocity", [a[1] for a in arm]), ("finger_qvel", [a[1] for a in fingers])]:
                np.testing.assert_array_equal(data["x_" + field], v[:-1, indices])
            np.testing.assert_array_equal(data["y_joint_position"], q[1:, [a[0] for a in arm]])
            np.testing.assert_array_equal(data["y_joint_velocity"], v[1:, [a[1] for a in arm]])
            np.testing.assert_array_equal(data["action_command"], source["sent_commands"])
            np.testing.assert_array_equal(data["x_previous_command_available"], np.arange(n) > 0)
            np.testing.assert_array_equal(data["x_previous_command"][0], np.zeros(7))
            np.testing.assert_array_equal(data["x_previous_command"][1:], source["sent_commands"][:-1])
            np.testing.assert_array_equal(data["step"], np.arange(n))
            np.testing.assert_array_equal(data["time"], source["sim_time"][:-1])
            np.testing.assert_allclose(np.diff(source["sim_time"]), 0.05, rtol=0, atol=1e-10)
            np.testing.assert_array_equal(data["protocol_native_goal_override"], np.arange(n) >= native)
            lo, hi = np.array(contract["input_min"]), np.array(contract["input_max"])
            outlo, outhi = np.array(contract["output_min"]), np.array(contract["output_max"])
            for k, goal in enumerate(goals):
                site = source["controller_site_pose"][k]
                command = source["sent_commands"][k]
                np.testing.assert_array_equal(goal["action"], command[:6])
                assert goal["step_index"] == k
                np.testing.assert_array_equal(data["x_previous_goal_pos"][k], source["controller_goal_pos"][k])
                np.testing.assert_array_equal(data["x_previous_goal_ori"][k], source["controller_goal_ori"][k])
                if k < native:
                    assert goal["overridden"] is False and goal["set_pos"] is None and goal["set_ori"] is None
                    delta = outlo + (command[:6] - lo) * (outhi - outlo) / (hi - lo)
                    expected_pos = site[:3, 3] + delta[:3]
                    expected_ori = (
                        native_delta_rotation(delta[3:]) @ site[:3, :3]
                        if np.any(delta[3:])
                        else source["controller_goal_ori"][k]
                    )
                else:
                    assert goal["overridden"] is True
                    expected_pos = source["controller_site_pose"][native, :3, 3]
                    expected_ori = source["controller_site_pose"][native, :3, :3]
                    np.testing.assert_allclose(goal["set_pos"], expected_pos, rtol=0, atol=1e-9)
                    np.testing.assert_allclose(goal["set_ori"], expected_ori, rtol=0, atol=1e-9)
                np.testing.assert_allclose(data["action_goal_pos"][k], expected_pos, rtol=0, atol=1e-9)
                np.testing.assert_allclose(data["action_goal_ori"][k], expected_ori, rtol=0, atol=1e-6)
                np.testing.assert_allclose(data["action_goal_pos"][k], goal["actual_goal_pos"], rtol=0, atol=1e-12)
                np.testing.assert_allclose(data["action_goal_ori"][k], goal["actual_goal_ori"], rtol=0, atol=1e-12)
                expected_delta = np.r_[
                    np.array(goal["actual_goal_pos"]) - site[:3, 3],
                    Rotation.from_matrix(np.array(goal["actual_goal_ori"]) @ site[:3, :3].T).as_rotvec(),
                ]
                np.testing.assert_allclose(data["action_goal_delta"][k], expected_delta, rtol=0, atol=1e-10)
                offsets = source["gripper_current_action_offsets"]
                m = source["gripper_current_action_flat"][offsets[k] : offsets[k + 1]]
                np.testing.assert_array_equal(data["x_motor_before"][k], np.repeat(m, 2) if len(m) == 1 else m)
                assert data["protocol_original_motor_size"][k] == len(m)
            poses = source["controller_site_pose"]
            np.testing.assert_allclose(data["y_motion"], poses[1:, :3, 3] - poses[:-1, :3, 3], rtol=0, atol=1e-10)
            expected_rotation = Rotation.from_matrix(
                poses[1:, :3, :3] @ poses[:-1, :3, :3].transpose(0, 2, 1)
            ).as_rotvec()
            np.testing.assert_allclose(data["y_rotation_increment"], expected_rotation, rtol=0, atol=1e-9)
            assert all(len(data[key]) == n and np.isfinite(data[key]).all() for key in data.files)
        metrics.append(dict(index=i, split=entry["split"], rows=n, finite_difference_max_abs=largest_fd))
        print("independent verified", i, entry["split"], n, "rows", flush=True)
    assert sum(x["split"] == "fit" for x in metrics) == 24 and sum(x["split"] == "holdout" for x in metrics) == 6
    counts = {split: sum(x["rows"] for x in metrics if x["split"] == split) for split in ("fit", "holdout")}
    assert counts == reconstructed["rows_by_split"]
    e.recheck()
    output = dict(
        complete=True,
        mechanical_data_verified=True,
        data_qualification_pending_owner_close=True,
        trajectories=metrics,
        rows_by_split=counts,
        source_sha256=e.hashes,
        physics_steps=0,
        new_fits=0,
        model_calls=0,
        fresh_cases=0,
    )
    with (ROOT / "independent_verified.json").open("x") as stream:
        json.dump(output, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
