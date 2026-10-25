"""Independent full064 saved-log, measured-label and image verification."""

import hashlib
import json
from pathlib import Path

import numpy as np
from audit_wam_position import Evidence
from training_recovery_acceptance_checker import verify_case
from verify_settled_native_training import verify_canvas
from verify_training_recovery_labels import check_labels

from s2a.owned_receipt_contract import normal_receipts

ROOT = Path("outputs/training_recovery_delay_isolation_20261025")


def main():
    evidence = Evidence()
    for path in (
        __file__,
        "scripts/training_recovery_acceptance_checker.py",
        "scripts/verify_training_recovery_labels.py",
        "scripts/verify_settled_native_training.py",
    ):
        evidence.read(path)
    execution = evidence.json(ROOT / "execution_result.json")
    if not execution["complete"] or not execution["passed"] or len(execution["stages"]) != 1:
        raise ValueError("Requires completed single064 physical trial")
    for path, sha in evidence.json(ROOT / "source_sha256.json").items():
        evidence.check(path, sha)
    directory = ROOT / "orchestration/case03"
    normal_receipts(
        evidence, directory / "launch.json", directory / "exit.json", directory / "release.json", format="durable"
    )
    registration = evidence.json("configs/training_state_recovery_qualification_20261025.json")
    ref = registration["records"][1]["trajectory_ref"]
    evidence.check(ref["path"], ref["sha256"])
    with np.load(ref["path"], allow_pickle=False) as native:
        poses, opening = native["source_pose"][1:].copy(), native["opening"][1:].copy()
    assert len(poses) == 254
    root = ROOT / "case03"
    config = evidence.json(root / "execution_config.json")
    case = next(root.glob("simulation/*/*"))
    manifest = evidence.json(case / "manifest.json")
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
        assert len(receipts) == len(data["source_pose"]) == 255
        check_labels(data, tool)
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
        images=images,
        input_sha256=evidence.hashes,
        new_model_calls=0,
        new_physics_steps=0,
        fresh_cases=0,
    )
    with (ROOT / "independent_verified.json").open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(dict(complete=True, performance_qualified=qualified, outcome=control["outcome"])))


if __name__ == "__main__":
    main()
