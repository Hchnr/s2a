"""076 independent actual-state, known-command and baseline equivalence audit."""

import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from audit_wam_position import Evidence
from verify_training_recovery_labels import check_labels

from s2a.owned_receipt_contract import normal_receipts

ROOT = Path("outputs/native_gripper_response_interventions_20261025_v3")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-index", type=int, choices=range(12), required=True)
    args = parser.parse_args()
    e = Evidence()
    e.read(__file__)
    folder = ROOT / f"case{args.case_index:02d}"
    registration = e.json("configs/native_gripper_response_interventions_20261025_v3.json")
    entry = registration["profiles"][args.case_index]
    for p, s in e.json(ROOT / "source_sha256.json").items():
        e.check(p, s)
    receipts = ROOT / "orchestration" / f"case{args.case_index:02d}"
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
        if entry["gripper_override"] is not None:
            expected[entry["anchor_cycle"] : entry["anchor_cycle"] + 8, -1] = entry["gripper_override"]
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
        matched = n + 1 if entry["branch"] == "native_baseline" else entry["anchor_cycle"] + 1
        for key in ("sim_states", "source_pose", "opening", "gripper_qpos"):
            np.testing.assert_allclose(data[key][:matched], source[key][:matched], rtol=0, atol=1e-10, err_msg=key)
        for k in range(matched):
            old = source["gripper_current_action_flat"][
                source["gripper_current_action_offsets"][k] : source["gripper_current_action_offsets"][k + 1]
            ]
            np.testing.assert_allclose(motors[k], old, rtol=0, atol=1e-12)
    e.recheck()
    output = dict(
        complete=True,
        mechanical_verified=True,
        baseline_native_match=entry["branch"] == "native_baseline",
        prefix_native_match=True,
        source_sha256=e.hashes,
        physics_steps=n,
        model_calls=0,
        fresh_cases=0,
        task_success_qualification=False,
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
