"""Measured endpoint/label/image checks for600-step preview WAM cases."""

import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from qualified_input_state_audit import verify_inputs as verify_actual_inputs
from verify_settled_native_training import verify_canvas
from verify_training_recovery_labels import check_labels


def verify_measured_endpoints(evidence, config, collection, case):
    root = Path(config["output_dir"]).resolve()
    array_ref = collection["arrays_ref"]
    evidence.check(array_ref["path"], array_ref["sha256"])
    receipt_ref = collection["endpoint_receipts_ref"]
    evidence.check(receipt_ref["path"], receipt_ref["sha256"])
    assert Path(receipt_ref["path"]) == root / "endpoints.jsonl"
    receipts = [json.loads(line) for line in evidence.read(root / "endpoints.jsonl").splitlines()]
    template = evidence.json(config["policy_template"])
    tool = np.asarray(evidence.json(template["tool_contract"])["source_from_tcp"])
    with np.load(array_ref["path"], allow_pickle=False) as data:
        assert len(receipts) == len(data["source_pose"]) == 601
        check_labels(data, tool)
        # Independently bind every planner input to its actual endpoint state.
        model = ET.fromstring(evidence.read(case / "model.xml"))
        nq = nv = 0
        fingers = {}
        for node in model.find("worldbody").iter():
            if node.tag not in ("joint", "freejoint"):
                continue
            kind = "free" if node.tag == "freejoint" else node.get("type", "hinge")
            if node.get("name") in ("gripper0_finger_joint1", "gripper0_finger_joint2"):
                fingers[node.get("name")] = (nq, nv)
            a, b = {"free": (7, 6), "ball": (4, 3), "slide": (1, 1), "hinge": (1, 1)}[kind]
            nq += a
            nv += b
        assert data["sim_states"].shape == (601, 1 + nq + nv)
        vids = [1 + nq + fingers[f"gripper0_finger_joint{i}"][1] for i in (1, 2)]
        rows = [json.loads(line) for line in evidence.read(case / "steps.jsonl").splitlines()]
        assert len(rows) == 600
        verify_actual_inputs(evidence.read(case / "model.xml"), data, rows)
        for k, row in enumerate(rows):
            np.testing.assert_allclose(row["measured_gripper_qvel"], data["sim_states"][k, vids], rtol=0, atol=1e-12)
            for label, endpoint in (("before", k), ("after", k + 1)):
                np.testing.assert_allclose(row[f"opening_{label}"], data["opening"][endpoint], rtol=0, atol=1e-12)
                offsets, flat = data["gripper_current_action_offsets"], data["gripper_current_action_flat"]
                np.testing.assert_allclose(
                    row[f"motor_target_{label}"], flat[offsets[endpoint] : offsets[endpoint + 1]], rtol=0, atol=1e-12
                )
            np.testing.assert_allclose(row["measured_source"], data["source_pose"][k], rtol=0, atol=1e-12)
            np.testing.assert_allclose(row["after_source"], data["source_pose"][k + 1], rtol=0, atol=1e-12)
        for i, receipt in enumerate(receipts):
            assert receipt["endpoint_step"] == i
            assert hashlib.sha256(data["sim_states"][i].tobytes()).hexdigest() == receipt["sim_state_sha256"]
        images = [verify_canvas(anchor, data, receipts[anchor["frame_index"]]) for anchor in collection["images"]]
    for anchor in collection["images"]:
        for key in ("raw_cameras", "canvas_png"):
            evidence.check(anchor[key]["path"], anchor[key]["sha256"])
    assert len(images) == 3
    return dict(endpoints=601, periods=600, images=images, measured_labels_verified=True)
