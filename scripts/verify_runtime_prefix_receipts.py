"""152 independent command/prefix/XML-kinematics/image audit."""

import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from audit_wam_position import Evidence
from verify_settled_native_training import verify_canvas
from verify_training_recovery_labels import check_labels

from s2a.owned_receipt_contract import normal_receipts
from s2a.xml_site_kinematics import XMLSiteKinematics

ROOT = Path("outputs/runtime_prefix_parity_corrected_20261025")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-index", type=int, choices=range(21), required=True)
    args = parser.parse_args()
    e = Evidence()
    e.read(__file__)
    e.read("src/s2a/xml_site_kinematics.py")
    e.read("scripts/verify_settled_native_training.py")
    e.read("scripts/verify_training_recovery_labels.py")
    folder = ROOT / f"case{args.case_index:02d}"
    registration = e.json("configs/left_table_interventions_20261025.json")
    trial = registration["trials"][args.case_index]
    entry = dict(registration["profiles"][trial["profile_index"]], **trial)
    for p, s in e.json(ROOT / "source_sha256.json").items():
        e.check(p, s)
    receipts = ROOT / "collection"
    normal_receipts(e, receipts / "launch.json", receipts / "exit.json", receipts / "release.json", format="durable")
    result = e.json(folder / "collection_result.json")
    assert result["complete"] and result["profile"] == entry and result["physics_steps"] == entry["physics_steps"]
    for p, s in result["source_sha256"].items():
        e.check(p, s)
    e.check(result["arrays_ref"]["path"], result["arrays_ref"]["sha256"])
    for ref in (entry["trajectory_ref"], entry["case_ref"]):
        e.check(ref["path"], ref["sha256"])
    prior = e.json(entry["case_ref"]["path"])
    pref = prior["prepared_ref"]
    e.check(pref["path"], pref["sha256"])
    prepared = e.json(pref["path"])["original_prepared"]
    h = np.array(prepared["tool_source_from_tcp"])
    n = entry["physics_steps"]
    xml = ET.fromstring(e.read(folder / "model.xml"))
    nq = nv = 0
    indices = {}
    for node in xml.find("worldbody").iter():
        if node.tag not in ("joint", "freejoint"):
            continue
        kind = "free" if node.tag == "freejoint" else node.get("type", "hinge")
        if node.get("name") in ("gripper0_finger_joint1", "gripper0_finger_joint2"):
            indices[node.get("name")] = (nq, nv)
        a, b = {"free": (7, 6), "ball": (4, 3), "slide": (1, 1), "hinge": (1, 1)}[kind]
        nq += a
        nv += b
    logs = [json.loads(line) for line in e.read(folder / "endpoints.jsonl").splitlines()]
    with (
        np.load(result["arrays_ref"]["path"], allow_pickle=False) as data,
        np.load(entry["trajectory_ref"]["path"], allow_pickle=False) as source,
    ):
        assert data["sim_states"].shape == (n + 1, 1 + nq + nv) and len(logs) == n + 1
        check_labels(data, h)
        expected = source["sent_commands"][:n].copy()
        if entry["translation_axis"] is not None:
            axis = entry["translation_axis"]
            assert type(axis) is int and axis in (0, 1, 2)
            assert entry["normalized_command_offset"] in (-0.1, 0.1)
            expected[-1, axis] = min(1.0, max(-1.0, expected[-1, axis] + entry["normalized_command_offset"]))
        else:
            assert entry["branch"] == "baseline" and entry["normalized_command_offset"] == 0
        assert n == entry["anchor_cycle"] + 1
        np.testing.assert_array_equal(expected[:, 3:], source["sent_commands"][:n, 3:])
        np.testing.assert_array_equal(expected[:-1], source["sent_commands"][: n - 1])
        np.testing.assert_array_equal(result["nominal_anchor_command"], source["sent_commands"][n - 1])
        np.testing.assert_array_equal(result["actual_anchor_command"], expected[-1])
        np.testing.assert_array_equal(
            result["realized_command_difference"], expected[-1] - source["sent_commands"][n - 1]
        )
        np.testing.assert_array_equal(data["sent_commands"], expected)
        np.testing.assert_allclose(np.diff(data["sim_time"]), 0.05, rtol=0, atol=1e-10)
        pos = [indices[f"gripper0_finger_joint{i}"][0] for i in (1, 2)]
        vel = [indices[f"gripper0_finger_joint{i}"][1] for i in (1, 2)]
        np.testing.assert_array_equal(data["gripper_qpos"], data["sim_states"][:, [1 + i for i in pos]])
        np.testing.assert_array_equal(data["gripper_qvel"], data["sim_states"][:, [1 + nq + i for i in vel]])
        flat = data["gripper_current_action_flat"]
        off = data["gripper_current_action_offsets"]
        motors = [flat[off[k] : off[k + 1]] for k in range(n + 1)]
        for k, command in enumerate(expected):
            m = np.array(motors[k], copy=True)
            if len(m) == 1:
                m = np.repeat(m, 2)
            for _ in range(25):
                m = np.clip(m + np.array([-1.0, 1.0]) * 0.01 * np.sign(command[-1]), -1, 1)
            np.testing.assert_allclose(m, motors[k + 1], rtol=0, atol=1e-12)
        for k, receipt in enumerate(logs):
            assert receipt["endpoint_step"] == k
            assert hashlib.sha256(data["sim_states"][k].tobytes()).hexdigest() == receipt["sim_state_sha256"]
        matched = n + 1 if entry["branch"] == "baseline" else entry["anchor_cycle"] + 1
        for key in (
            "sim_states",
            "source_pose",
            "opening",
            "gripper_qpos",
            "controller_goal_pos",
            "controller_goal_ori",
            "controller_site_pose",
        ):
            np.testing.assert_allclose(data[key][:matched], source[key][:matched], rtol=0, atol=1e-10, err_msg=key)
        for k in range(matched):
            old = source["gripper_current_action_flat"][
                source["gripper_current_action_offsets"][k] : source["gripper_current_action_offsets"][k + 1]
            ]
            np.testing.assert_allclose(motors[k], old, rtol=0, atol=1e-12)
        contract = result["controller_contract"]
        assert contract["control_freq"] == 20 and contract["model_timestep"] == 0.002
        np.testing.assert_array_equal(contract["input_min"], -np.ones(6))
        np.testing.assert_array_equal(contract["input_max"], np.ones(6))
        assert np.all(np.abs(data["sent_commands"][:, :6]) <= 1)
        kinematics = XMLSiteKinematics(e.read(folder / "model.xml"), contract["controller_site"])
        assert data["source_twist"].shape == (n + 1, 6)
        for k in range(n + 1):
            pose, twist = kinematics.endpoint(data["sim_states"][k])
            np.testing.assert_allclose(pose, data["controller_site_pose"][k], rtol=0, atol=1e-9)
            np.testing.assert_allclose(twist, data["source_twist"][k], rtol=0, atol=1e-9)
        assert [anchor["frame_index"] for anchor in result["images"]] == [entry["anchor_cycle"], n]
        for anchor in result["images"]:
            for key in ("raw_cameras", "canvas_png"):
                ref = anchor[key]
                e.check(ref["path"], ref["sha256"])
            verify_canvas(anchor, data, logs[anchor["frame_index"]])
    e.recheck()
    output = dict(
        complete=True,
        mechanical_verified=True,
        baseline_native_match=entry["branch"] == "baseline",
        prefix_native_match=True,
        source_sha256=e.hashes,
        physics_steps=n,
        model_calls=0,
        fresh_cases=0,
        task_success_qualification=False,
        source_twist_independent_xml_verified=True,
        current_images_verified=True,
        anchor_group=entry["anchor_group"],
    )
    with (folder / "independent_verified.json").open("x") as stream:
        json.dump(output, stream, indent=2)
        stream.write("\n")
    print(
        json.dumps(
            dict(case=args.case_index, baseline_native_match=output["baseline_native_match"], mechanical_verified=True)
        )
    )


if __name__ == "__main__":
    main()
