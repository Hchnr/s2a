"""Independent full064 saved-log, measured-label and image verification."""

import argparse
import ctypes
import hashlib
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from audit_wam_position import Evidence
from full_coupled_sqp_acceptance_checker import verify_case
from qualified_input_state_audit import verify_inputs as verify_actual_inputs
from run_matched_oracle_case import CONTROL, CONTROL_SHA
from verify_settled_native_training import verify_canvas
from verify_training_recovery_labels import check_labels

from s2a.artifacts import sha256
from s2a.native_environment import restore_cv2_environment, strict_native_environment
from s2a.owned_receipt_contract import normal_receipts

ROOT = Path("outputs/full_coupled_sqp_control_20261026")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-index", type=int, choices=range(18), required=True)
    args = parser.parse_args()
    strict_native_environment()
    assert sha256(CONTROL) == CONTROL_SHA
    guard = json.loads(CONTROL.read_text())["cv2_guard_registration"]
    restore_cv2_environment(guard["exact_probe_ref"], guard["runtime_source_refs"])
    label = f"case{args.case_index:02d}"
    evidence = Evidence()
    for path in (
        __file__,
        "scripts/full_coupled_sqp_acceptance_checker.py",
        "scripts/current_coupled_independent.py",
        "scripts/full_coupled_pose_independent.py",
        "scripts/velocity_plan_independent_audit.py",
        "scripts/loaded_velocity_independent_replay.py",
        "scripts/pose_velocity_arm_independent_replay.py",
        "scripts/full_coupled_arm_log_replay.py",
        "scripts/qualified_inverse_independent.py",
        "scripts/qualified_input_state_audit.py",
        "scripts/verify_gripper_position_hold.py",
        "scripts/verify_training_recovery_labels.py",
        "scripts/verify_settled_native_training.py",
    ):
        evidence.read(path)
    for path, sha in evidence.json(ROOT / "source_sha256.json").items():
        evidence.check(path, sha)
    directory = ROOT / "orchestration" / label
    normal_receipts(
        evidence, directory / "launch.json", directory / "exit.json", directory / "release.json", format="durable"
    )
    registration = evidence.json("configs/full_coupled_sqp_control_registered_20261026.json")
    profile = registration["profiles"][args.case_index]
    ref = profile["trajectory_ref"]
    evidence.check(ref["path"], ref["sha256"])
    with np.load(ref["path"], allow_pickle=False) as native:
        if profile["format"] == "native_endpoints":
            poses, opening = native["source_pose"][1:].copy(), native["opening"][1:].copy()
        elif profile["format"] == "world_reference":
            poses, opening = native["poses"].copy(), native["openings"].copy()
        else:
            raise ValueError("Unregistered reference format")
    assert len(poses) == profile["physics_steps"]
    root = ROOT / label
    config = evidence.json(root / "execution_config.json")
    assert config["gripper_preview_steps"] == profile["preview_steps"]
    assert (
        config["arm_pose_objective"] == "full_pose_geodesic_plus_dt_world_twist_squared"
        and config["source_provisional_inverse_weight"] == 0.5
    )
    assert config["arm_residual_policy"] == registration["arm_residual_policy"]
    case = next(root.glob("simulation/*/*"))
    manifest = evidence.json(case / "manifest.json")
    cache = root / "verification_cache"
    cache.mkdir(exist_ok=False)
    os.environ.update(NUMBA_CACHE_DIR=str(cache / "numba"), MPLCONFIGDIR=str(cache / "matplotlib"))
    ctypes.CDLL(manifest["runtime_config"]["osmesa_library"], mode=ctypes.RTLD_GLOBAL)
    control = verify_case(
        config,
        manifest["record"],
        config["variants"][0],
        case,
        evidence,
        source="oracle",
        checkpoint=None,
        full=True,
        sampling_seed=42,
        first_query_validator=None,
        oracle_reference=(poses, opening),
    )
    collection = evidence.json(root / "collection_result.json")
    array_ref = collection["arrays_ref"]
    evidence.check(array_ref["path"], array_ref["sha256"])
    receipts = [json.loads(line) for line in evidence.read(root / "endpoints.jsonl").splitlines()]
    template = evidence.json(config["policy_template"])
    tool = np.asarray(evidence.json(template["tool_contract"])["source_from_tcp"])
    with np.load(array_ref["path"], allow_pickle=False) as data:
        assert len(receipts) == len(data["source_pose"]) == len(poses) + 1
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
        assert data["sim_states"].shape == (len(poses) + 1, 1 + nq + nv)
        vids = [1 + nq + fingers[f"gripper0_finger_joint{i}"][1] for i in (1, 2)]
        rows = [json.loads(line) for line in evidence.read(case / "steps.jsonl").splitlines()]
        assert len(rows) == len(poses)
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
    evidence.recheck()
    qualified = (
        control["outcome"]["stable"]
        and control["tracking"]["admitted"]["position_rmse_m"] <= 0.01
        and control["tracking"]["admitted"]["so3_rmse_rad"] <= 0.05
    )
    result = dict(
        complete=True,
        mechanical_labels_images_passed=True,
        performance_qualified=qualified,
        control=control,
        routing_summary=dict(
            native_load_cycles=sum(r["gripper"]["route"] == "native_close_load" for r in rows),
            tracking_cycles=sum(r["gripper"]["route"] == "model_tracking" for r in rows),
            executed_predictions_outside_velocity_envelope=sum(
                not r["gripper"]["executed_prediction_in_velocity_envelope"] for r in rows
            ),
        ),
        images=images,
        input_sha256=evidence.hashes,
        new_model_calls=0,
        new_physics_steps=0,
        fresh_cases=0,
    )
    with (root / "independent_verified.json").open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(dict(complete=True, performance_qualified=qualified, outcome=control["outcome"])))


if __name__ == "__main__":
    main()
