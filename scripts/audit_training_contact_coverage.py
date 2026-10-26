"""141 current training contact geometry only; no integration or force claim."""

from pathlib import Path

import mujoco
import numpy as np
from audit_wam_position import Evidence
from collect_wam_native_training import write_once


def main():
    e = Evidence()
    e.read(__file__)
    e.read("plans/v0.0.1/141_training_contact_coverage_20261025.md")
    root = Path("outputs/condition_span_forward_20261025")
    c = e.json(root / "closure_verified.json")
    assert c["complete"] and c["numerically_verified"] and c["owner_normal_wait_verified"]
    p = root / "fit_result.json"
    e.check(p, c["source_sha256"][str(p.resolve())])
    fit = e.json(p)
    seen = set()
    rows = []
    for r in fit["contrast_sources"]:
        identity = (r["dataset"], r["anchor_group"])
        if identity in seen:
            continue
        seen.add(identity)
        ref = r["baseline_ref"]
        e.check(ref["path"], ref["sha256"])
        folder = Path(ref["path"]).parent
        meta = folder / "collection_result.json"
        e.check(meta, fit["source_sha256"][str(meta.resolve())])
        record = e.json(meta)
        xml = folder / "model.xml"
        e.check(xml, record["source_sha256"][str(xml.resolve())])
        with np.load(ref["path"], allow_pickle=False) as a:
            state = a["sim_states"][-2].copy()
            pose = a["controller_site_pose"][-2].copy()
            twist = a["source_twist"][-2].copy()
        model = mujoco.MjModel.from_xml_path(str(xml.resolve()))
        data = mujoco.MjData(model)
        assert len(state) == 1 + model.nq + model.nv
        data.time = state[0]
        data.qpos[:] = state[1 : 1 + model.nq]
        data.qvel[:] = state[1 + model.nq :]
        mujoco.mj_forward(model, data)
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, record["controller_contract"]["controller_site"])
        assert sid >= 0
        np.testing.assert_allclose(data.site_xpos[sid], pose[:3, 3], rtol=0, atol=1e-9)
        np.testing.assert_allclose(data.site_xmat[sid].reshape(3, 3), pose[:3, :3], rtol=0, atol=1e-9)
        jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jp, jr, sid)
        np.testing.assert_allclose(np.vstack([jp, jr]) @ data.qvel, twist, rtol=0, atol=1e-8)
        contacts = []
        for contact in data.contact[: data.ncon]:
            bodies = [
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g])) or ""
                for g in (contact.geom1, contact.geom2)
            ]
            if any(b.startswith(("robot0_", "gripper0_")) for b in bodies):
                contacts.append(dict(bodies=bodies, distance=float(contact.dist)))
        rows.append(
            dict(
                dataset=r["dataset"],
                anchor_group=r["anchor_group"],
                robot_contacts=contacts,
                robot_table_contact=any("table" in v["bodies"] for v in contacts),
            )
        )
    assert len(rows) == 66
    e.recheck()
    out = Path("outputs/training_contact_coverage_20261025")
    out.mkdir(exist_ok=False)
    write_once(
        out / "result.json",
        dict(
            complete=True,
            training_only=True,
            anchors=rows,
            source_sha256=e.hashes,
            contact_forces_measured=False,
            new_fits=0,
            physics_steps=0,
            model_calls=0,
            fresh_cases=0,
        ),
    )
    for dataset in ("086", "097", "106", "115"):
        subset = [r for r in rows if r["dataset"] == dataset]
        print(
            dataset,
            "anchors",
            len(subset),
            "robot contact",
            sum(bool(r["robot_contacts"]) for r in subset),
            "robot-table",
            sum(r["robot_table_contact"] for r in subset),
        )


if __name__ == "__main__":
    main()
