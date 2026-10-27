"""142 original fit-prefix current table-contact masks, no integration."""

import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from audit_wam_position import Evidence
from collect_wam_native_training import write_once


def main():
    e = Evidence()
    e.read(__file__)
    e.read("plans/v0.0.1/142_native_training_table_coverage_20261025.md")
    config = e.json("configs/condition_span_forward_20261025.json")
    ref = config["data_closure_ref"]
    e.check(ref["path"], ref["sha256"])
    closed = e.json(ref["path"])
    assert closed["complete"] and closed["data_qualified"] and closed["owner_normal_wait_verified"]
    ref = config["data_result_ref"]
    e.check(ref["path"], ref["sha256"])
    source = e.json(ref["path"])
    entries = [r for r in source["trajectories"] if r["split"] == "fit"]
    assert len(entries) == 24
    rows = []
    for entry in entries:
        ref = entry["cache_ref"]
        e.check(ref["path"], ref["sha256"])
        with np.load(ref["path"], allow_pickle=False) as a:
            poses = a["x_site_pose"].copy()
            twists = a["x_site_twist"].copy()
            override = a["protocol_native_goal_override"].copy()
        ref = entry["source"]
        e.check(ref["path"], ref["sha256"])
        meta = Path(ref["path"]).parent / "result.json"
        e.check(meta, closed["source_sha256"][str(meta.resolve())])
        record = e.json(meta)
        with np.load(ref["path"], allow_pickle=False) as a:
            states = a["sim_states"][: len(poses)].copy()
        xml_root = ET.fromstring(record["record"]["model_xml"])
        # Resolve historical asset locations by exact asset tag/name in an audited snapshot.
        paired = e.json(config["paired_closure_ref"]["path"])
        snapshot_path = Path("outputs/paired_arm_response_20261025/case00/model.xml")
        snapshot_meta = snapshot_path.parent / "collection_result.json"
        e.check(snapshot_meta, paired["source_sha256"][str(snapshot_meta.resolve())])
        snapshot = e.json(snapshot_meta)
        e.check(snapshot_path, snapshot["source_sha256"][str(snapshot_path.resolve())])
        known = {
            (n.tag, n.get("name")): n.get("file")
            for n in ET.fromstring(snapshot_path.read_bytes()).iter()
            if n.get("file")
        }
        asset_mapping = []
        for node in xml_root.iter():
            file = node.get("file")
            if file:
                replacement = known[(node.tag, node.get("name"))]
                assert Path(replacement).is_absolute()
                e.read(replacement)
                asset_mapping.append(
                    dict(tag=node.tag, name=node.get("name"), historical_path=file, resolved_path=replacement)
                )
                node.set("file", replacement)
        xml = ET.tostring(xml_root, encoding="unicode")
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, record["controller_contract"]["controller_site"])
        assert sid >= 0
        masks = {name: [] for name in ("left", "right", "other")}
        for k, state in enumerate(states):
            assert len(state) == 1 + model.nq + model.nv
            data.time = state[0]
            data.qpos[:] = state[1 : 1 + model.nq]
            data.qvel[:] = state[1 + model.nq :]
            mujoco.mj_forward(model, data)
            np.testing.assert_allclose(data.site_xpos[sid], poses[k, :3, 3], rtol=0, atol=1e-9)
            np.testing.assert_allclose(data.site_xmat[sid].reshape(3, 3), poses[k, :3, :3], rtol=0, atol=1e-9)
            jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
            mujoco.mj_jacSite(model, data, jp, jr, sid)
            np.testing.assert_allclose(np.vstack([jp, jr]) @ data.qvel, twists[k], rtol=0, atol=1e-8)
            contacts = set()
            for contact in data.contact[: data.ncon]:
                bodies = [
                    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g])) or ""
                    for g in (contact.geom1, contact.geom2)
                ]
                if "table" not in bodies:
                    continue
                for b in bodies:
                    if b.startswith(("robot0_", "gripper0_")):
                        contacts.add(
                            "left"
                            if "leftfinger" in b or "finger_joint1" in b
                            else "right"
                            if "rightfinger" in b or "finger_joint2" in b
                            else "other"
                        )
            for name in masks:
                masks[name].append(k if name in contacts else -1)
        indices = {name: [v for v in values if v >= 0] for name, values in masks.items()}
        rows.append(
            dict(
                index=entry["index"],
                asset_path_mapping=asset_mapping,
                rows=len(poses),
                table_contact_cycles=indices,
                protocol_override_cycles=np.flatnonzero(override).tolist(),
            )
        )
        print(
            "trajectory",
            entry["index"],
            "rows",
            len(poses),
            "table cycles",
            {k: len(v) for k, v in indices.items()},
            flush=True,
        )
    assert sum(r["rows"] for r in rows) == 4865
    e.recheck()
    out = Path("outputs/native_training_table_coverage_20261025")
    out.mkdir(exist_ok=False)
    write_once(
        out / "result.json",
        dict(
            complete=True,
            training_only=True,
            trajectories=rows,
            source_sha256=e.hashes,
            new_fits=0,
            physics_steps=0,
            model_calls=0,
            fresh_cases=0,
        ),
    )


if __name__ == "__main__":
    main()
