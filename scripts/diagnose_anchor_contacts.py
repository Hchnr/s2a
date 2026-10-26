"""092 offline geometry and torque-derivative audit; no physical integration."""

import json
from pathlib import Path

import mujoco
import numpy as np
from audit_wam_position import Evidence
from scipy.linalg import block_diag

ROOT = Path("outputs/anchor_contact_diagnosis_20261025")


def name(model, kind, index):
    return mujoco.mj_id2name(model, kind, index) or f"unnamed:{index}"


def main():
    e = Evidence()
    e.read(__file__)
    e.read("plans/v0.0.1/092_anchor_contact_diagnosis_20261025.md")
    closed = e.json("outputs/paired_arm_response_20261025/closure_verified.json")
    assert closed["complete"] and closed["data_qualified"] and closed["owner_normal_wait_verified"]
    groups = []
    for group in range(6):
        folder = Path("outputs/paired_arm_response_20261025") / f"case{7 * group:02d}"
        path = folder / "collection_result.json"
        e.check(path, closed["source_sha256"][str(path.resolve())])
        record = e.json(path)
        for p, signature in record["source_sha256"].items():
            if p.endswith((".png", ".obj", ".stl", ".xml", "/osc.py", "/control_utils.py", "/single_arm.py")):
                e.check(p, signature)
        xml = folder / "model.xml"
        e.check(xml, record["source_sha256"][str(xml.resolve())])
        m = mujoco.MjModel.from_xml_path(str(xml.resolve()))
        d = mujoco.MjData(m)
        ref = record["arrays_ref"]
        e.check(ref["path"], ref["sha256"])
        k = record["profile"]["anchor_cycle"]
        c = record["controller_contract"]
        with np.load(ref["path"], allow_pickle=False) as cache:
            state = cache["sim_states"][k].copy()
            pose = cache["controller_site_pose"][k].copy()
            twist = cache["source_twist"][k].copy()
        assert len(state) == 1 + m.nq + m.nv
        d.time, d.qpos[:], d.qvel[:] = state[0], state[1 : 1 + m.nq], state[1 + m.nq :]
        mujoco.mj_forward(m, d)
        sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, c["controller_site"])
        assert sid >= 0
        np.testing.assert_allclose(d.site_xpos[sid], pose[:3, 3], rtol=0, atol=1e-9)
        np.testing.assert_allclose(d.site_xmat[sid].reshape(3, 3), pose[:3, :3], rtol=0, atol=1e-9)
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        mujoco.mj_jacSite(m, d, jp, jr, sid)
        jfull = np.vstack([jp, jr])
        np.testing.assert_allclose(jfull @ d.qvel, twist, rtol=0, atol=1e-8)
        joints = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"robot0_joint{i}") for i in range(1, 8)]
        assert min(joints) >= 0
        dofs = m.jnt_dofadr[joints]
        mass = np.zeros((m.nv, m.nv))
        mujoco.mj_fullM(m, mass, d.qM)
        arm_mass, j = mass[np.ix_(dofs, dofs)], jfull[:, dofs]
        np.linalg.cholesky(arm_mass)
        inverse_inertia = j @ np.linalg.solve(arm_mass, j.T)
        lambdas = block_diag(np.linalg.pinv(inverse_inertia[:3, :3]), np.linalg.pinv(inverse_inertia[3:, 3:]))
        scale = (np.array(c["output_max"])[:3] - np.array(c["output_min"])[:3]) / (
            np.array(c["input_max"])[:3] - np.array(c["input_min"])[:3]
        )
        derivative = (j.T @ lambdas @ np.diag(c["kp"]))[:, :3] * scale[None, :]
        actuators = []
        for column, jid in enumerate(joints):
            ids = np.flatnonzero(m.actuator_trnid[:, 0] == jid)
            assert len(ids) == 1
            aid = int(ids[0])
            actuators.append(
                dict(
                    name=name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, aid),
                    ctrl_limited=bool(m.actuator_ctrllimited[aid]),
                    ctrl_range=m.actuator_ctrlrange[aid].tolist(),
                    gear=m.actuator_gear[aid].tolist(),
                    translation_torque_derivative=derivative[column].tolist(),
                )
            )
        contacts = []
        for contact in d.contact[: d.ncon]:
            geoms = [int(contact.geom1), int(contact.geom2)]
            bodies = [name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g])) for g in geoms]
            robot_contact = any(b.startswith(("robot0_", "gripper0_")) for b in bodies)
            contacts.append(
                dict(
                    geoms=[name(m, mujoco.mjtObj.mjOBJ_GEOM, g) for g in geoms],
                    bodies=bodies,
                    distance=float(contact.dist),
                    dimension=int(contact.dim),
                    efc_address=int(contact.efc_address),
                    robot_contact=robot_contact,
                )
            )
        groups.append(
            dict(
                anchor_group=group,
                anchor_cycle=k,
                contacts=contacts,
                robot_contact_count=sum(c["robot_contact"] for c in contacts),
                constraint_rows=int(d.nefc),
                arm_actuators=actuators,
            )
        )
    e.recheck()
    ROOT.mkdir(exist_ok=False)
    out = dict(
        complete=True,
        groups=groups,
        source_sha256=e.hashes,
        physics_steps=0,
        model_calls=0,
        new_fits=0,
        fresh_cases=0,
        actual_saturation_verified=False,
        contact_forces_measured=False,
        control_qualification=False,
        scope="offline restored geometry and assumed uncoupled OSC command derivative; no restored historical ctrl/caches",
    )
    with (ROOT / "result.json").open("x") as stream:
        json.dump(out, stream, indent=2, allow_nan=False)
        stream.write("\n")
    for g in groups:
        print(
            g["anchor_group"],
            "contacts",
            len(g["contacts"]),
            "robot",
            g["robot_contact_count"],
            "constraints",
            g["constraint_rows"],
            flush=True,
        )
        for c in g["contacts"]:
            if c["robot_contact"]:
                print(c["bodies"], c["distance"], flush=True)


if __name__ == "__main__":
    main()
