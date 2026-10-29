"""Check all saved measured endpoints and q7 limits without advancing physics."""

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from audit_wam_position import Evidence
from coupled_preview_wam_measured_audit import verify_measured_endpoints

from s2a.artifacts import REPO
from s2a.clean_training_runtime import publish, reference, require


def main():
    root = REPO / "outputs/tracking_preview_wam_development_v5_20261028"
    ev = Evidence()
    worker_path = root / "development/cases/task0_init0_s42_terminal058/worker_result.json"
    worker = ev.json(worker_path)
    require(worker["complete"] is worker["passed"] is True and worker["physics_steps"] == 600, "Closed600")
    ev.check(worker["config_ref"]["path"], worker["config_ref"]["sha256"])
    config = ev.json(worker["config_ref"]["path"])
    case = Path(worker["physical_case_dir"])
    measured = verify_measured_endpoints(ev, config, worker, case)
    model = ET.fromstring(ev.read(case / "model.xml"))
    require(model.find("compiler").get("angle") == "radian", "Radian joint limits")
    offset, q7_offset, limits = 0, None, None
    for node in model.find("worldbody").iter():
        if node.tag not in ("joint", "freejoint"):
            continue
        kind = "free" if node.tag == "freejoint" else node.get("type", "hinge")
        if node.get("name") == "robot0_joint7":
            require(kind == "hinge", "Native hinge q7")
            q7_offset = offset
            limits = np.asarray([float(x) for x in node.get("range").split()])
        offset += {"free": 7, "ball": 4, "slide": 1, "hinge": 1}[kind]
    require(q7_offset is not None and limits.shape == (2,), "Explicit q7 address and limits")
    with np.load(worker["arrays_ref"]["path"], allow_pickle=False) as archive:
        q7 = archive["sim_states"][:, 1 + q7_offset].copy()
    rows = [json.loads(line) for line in ev.read(case / "steps.jsonl").splitlines()]
    logged = np.asarray([row["coupled_dynamics"]["robot_joint_position"][6] for row in rows])
    np.testing.assert_allclose(q7[:-1], logged, rtol=0, atol=1e-12)
    margin = np.minimum(q7 - limits[0], limits[1] - q7)
    outside = np.flatnonzero(margin < 0)
    ev.recheck()
    result = dict(
        complete=True,
        passed=True,
        measured_endpoint_audit=measured,
        q7_limits_rad=limits.tolist(),
        minimum_q7_margin_rad=float(margin.min()),
        outside_endpoints=outside.tolist(),
        outside_before_periods=int(np.sum(margin[:-1] < 0)),
        source_and_artifact_sha256=ev.hashes,
        source_refs=[
            reference(Path(__file__)),
            reference(REPO / "scripts/coupled_preview_wam_measured_audit.py"),
            reference(REPO / "scripts/qualified_input_state_audit.py"),
        ],
        physics_steps=0,
        world_model_calls=0,
        fresh_cases=0,
        scope="Saved601 state/label/image integrity and joint-limit measurement; passed does not mean safe limits or task success",
    )
    print(json.dumps(publish(REPO / "outputs/tracking_v5_case01_endpoints_20261028/verified.json", result)))


if __name__ == "__main__":
    main()
