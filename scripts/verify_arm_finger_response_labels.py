"""180 independently recover aperture changes from raw states and XML joints."""

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from audit_wam_position import Evidence

from s2a.artifacts import REPO
from s2a.durable_stage_supervisor import publish


def aperture(states, xml):
    nodes = ET.fromstring(xml).find("worldbody")
    qindex = 0
    selected = {}
    for node in nodes.iter():
        if node.tag not in ("joint", "freejoint"):
            continue
        kind = "free" if node.tag == "freejoint" else node.get("type", "hinge")
        if node.get("name") in ("gripper0_finger_joint1", "gripper0_finger_joint2"):
            assert kind == "slide"
            selected[node.get("name")] = qindex
        qindex += dict(free=7, ball=4, hinge=1, slide=1)[kind]
    assert len(selected) == 2
    states = np.asarray(states, dtype=float)
    assert states.ndim == 2 and np.isfinite(states).all()
    return states[:, 1 + selected["gripper0_finger_joint1"]] - states[:, 1 + selected["gripper0_finger_joint2"]]


def main():
    out = REPO / "outputs/arm_finger_labels_independent_20261025"
    if out.exists():
        raise FileExistsError(out)
    e = Evidence()
    e.read(__file__)
    produced = e.json("outputs/arm_finger_response_labels_20261025/result.json")
    source = e.json("outputs/native_arm_response_data_20261025/reconstruction_result.json")
    registration = e.json("configs/native_arm_response_data_20261025.json")
    equations = e.json("outputs/left_table_forward_equations_20261025/result.json")
    fit, heldout = [], []
    for entry, profile in zip(source["trajectories"], registration["trajectories"], strict=True):
        assert entry["source"] == profile["ref"]
        ref = profile["result_ref"]
        e.check(ref["path"], ref["sha256"])
        metadata = e.json(ref["path"])
        ref = entry["source"]
        e.check(ref["path"], ref["sha256"])
        with np.load(ref["path"], allow_pickle=False) as data:
            gaps = aperture(data["sim_states"], metadata["record"]["model_xml"])
        labels = gaps[1:] - gaps[:-1]
        (fit if entry["split"] == "fit" else heldout).extend(labels.tolist())
    natural_count = len(fit)
    for row in equations["contrast_sources"]:
        changes = []
        for key in ("minus_ref", "plus_ref"):
            ref = row[key]
            e.check(ref["path"], ref["sha256"])
            xml = Path(ref["path"]).parent / "model.xml"
            text = e.read(xml).decode()
            with np.load(ref["path"], allow_pickle=False) as data:
                gaps = aperture(data["sim_states"][-2:], text)
                changes.append(gaps[1] - gaps[0])
        fit.append((changes[1] - changes[0]) / row["actual_normalized_delta"])
    ref = produced["labels_ref"]
    e.check(ref["path"], ref["sha256"])
    with np.load(ref["path"], allow_pickle=False) as target:
        np.testing.assert_allclose(fit, target["fit_target"], atol=1e-12, rtol=0)
        np.testing.assert_allclose(heldout, target["holdout_target"], atol=1e-12, rtol=0)
    assert natural_count == 4865 and len(fit) == 5072 and len(heldout) == 1315
    e.recheck()
    publish(
        out / "verified.json",
        dict(
            complete=True,
            independent_raw_state_xml_verified=True,
            fit_rows=len(fit),
            holdout_rows=len(heldout),
            source_sha256=e.hashes,
            physics_steps=0,
            fits=0,
            control_qualified=False,
        ),
    )
    print("180 all5072fit/1315holdout labels match raw-state/XML recovery")


if __name__ == "__main__":
    main()
