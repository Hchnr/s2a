"""172 saved endpoint contact geometry only; no integration or force reconstruction."""

import mujoco
import numpy as np
from audit_wam_position import Evidence

from s2a.artifacts import REPO
from s2a.durable_stage_supervisor import publish


def main():
    root = REPO / "outputs/departure_contact_geometry_20261025"
    if root.exists():
        raise FileExistsError(root)
    e = Evidence()
    e.read(__file__)
    results = []
    for name in ("causal_residual_loaded_control", "qualified_inverse_enumerated", "qualified_inverse_supported"):
        case = REPO / f"outputs/{name}_20261025/case03"
        assert e.json(case / "independent_verified.json")["mechanical_labels_images_passed"]
        collection = e.json(case / "collection_result.json")
        ref = collection["arrays_ref"]
        e.check(ref["path"], ref["sha256"])
        with np.load(ref["path"], allow_pickle=False) as arrays:
            states = arrays["sim_states"].copy()
            poses = arrays["controller_site_pose"].copy()
        xml = next(case.rglob("model.xml"))
        e.read(xml)
        contract = e.json(xml.with_name("controller.json"))["contract"]
        model = mujoco.MjModel.from_xml_path(str(xml))
        data = mujoco.MjData(model)
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, contract["controller_site"])
        assert sid >= 0
        rows = []
        for endpoint in range(50, 62):
            state = states[endpoint]
            assert len(state) == 1 + model.nq + model.nv
            data.time = state[0]
            data.qpos[:] = state[1 : 1 + model.nq]
            data.qvel[:] = state[1 + model.nq :]
            mujoco.mj_forward(model, data)
            np.testing.assert_allclose(data.site_xpos[sid], poses[endpoint, :3, 3], atol=1e-9, rtol=0)
            np.testing.assert_allclose(data.site_xmat[sid].reshape(3, 3), poses[endpoint, :3, :3], atol=1e-9, rtol=0)
            contacts = []
            for contact in data.contact[: data.ncon]:
                bodies = [
                    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g])) or ""
                    for g in (contact.geom1, contact.geom2)
                ]
                if any(b.startswith(("robot0_", "gripper0_")) for b in bodies):
                    contacts.append(dict(bodies=bodies, distance=float(contact.dist), position=contact.pos.tolist()))
            rows.append(dict(endpoint=endpoint, time=float(data.time), contacts=contacts))
        results.append(dict(name=name, endpoints=rows))
    e.recheck()
    publish(
        root / "verified.json",
        dict(
            complete=True,
            cases=results,
            source_sha256=e.hashes,
            physics_steps=0,
            fits=0,
            reconstructed_forces=False,
            limitation="Saved endpoint collision geometry, mj_forward without mj_step; no historical warmstart/controller/force replay or causal proof",
        ),
    )
    for case in results:
        print(case["name"], [(r["endpoint"], [c["bodies"] for c in r["contacts"]]) for r in case["endpoints"]])


if __name__ == "__main__":
    main()
