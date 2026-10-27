"""095 fixed-state constrained acceleration derivatives; not endpoint predictions."""

import json
from pathlib import Path

import mujoco
import numpy as np
from audit_wam_position import Evidence
from scipy.linalg import block_diag

ROOT = Path("outputs/contact_acceleration_sensitivity_20261025")


def main():
    e = Evidence()
    e.read(__file__)
    e.read("plans/v0.0.1/095_contact_acceleration_sensitivity_20261025.md")
    source_root = Path("outputs/paired_arm_response_20261025")
    closed = e.json(source_root / "closure_verified.json")
    trace_root = Path("outputs/native_controller_trace_20261025")
    trace_closed = e.json(trace_root / "closure_verified.json")
    assert closed["complete"] and closed["data_qualified"] and trace_closed["owner_normal_wait_verified"]
    groups = []
    for group in range(6):
        folder = source_root / f"case{7 * group:02d}"
        path = folder / "collection_result.json"
        e.check(path, closed["source_sha256"][str(path.resolve())])
        c = e.json(path)
        for p, signature in c["source_sha256"].items():
            if p.endswith((".xml", ".png", ".stl", ".obj")):
                e.check(p, signature)
        trace_verified = e.json(trace_root / f"group{group:02d}/independent_verified.json")
        assert trace_verified["complete"] and trace_verified["prefix_matches"] and trace_verified["actual_uncoupling"]
        for p, signature in trace_verified["source_sha256"].items():
            e.check(p, signature)
        t = e.json(trace_root / f"group{group:02d}/collection_result.json")
        ref = t["trace_ref"]
        e.check(ref["path"], ref["sha256"])
        first = e.json(ref["path"])["records"][0]
        ref = c["arrays_ref"]
        e.check(ref["path"], ref["sha256"])
        k = c["profile"]["anchor_cycle"]
        with np.load(ref["path"], allow_pickle=False) as cache:
            state = cache["sim_states"][k].copy()
            pose = cache["controller_site_pose"][k].copy()
            twist = cache["source_twist"][k].copy()
            offsets = cache["gripper_current_action_offsets"]
            motor = cache["gripper_current_action_flat"][offsets[k] : offsets[k + 1]].copy()
            command = cache["sent_commands"][k].copy()
        assert motor.shape == (2,)
        m = mujoco.MjModel.from_xml_path(str((folder / "model.xml").resolve()))
        d = mujoco.MjData(m)
        assert len(state) == 1 + m.nq + m.nv
        d.time, d.qpos[:], d.qvel[:] = state[0], state[1 : 1 + m.nq], state[1 + m.nq :]
        assert abs(d.time - first["time"]) < 1e-10
        jids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"robot0_joint{i}") for i in range(1, 8)]
        assert min(jids) >= 0
        dofs = m.jnt_dofadr[jids]
        for jid, ctrl in zip(jids, first["ctrl"], strict=True):
            ids = np.flatnonzero(m.actuator_trnid[:, 0] == jid)
            assert len(ids) == 1
            np.testing.assert_array_equal(m.actuator_gear[ids[0]], [1, 0, 0, 0, 0, 0])
            d.ctrl[ids[0]] = ctrl
        after = np.clip(motor + np.array([-1, 1]) * 0.01 * np.sign(command[-1]), -1, 1)
        for i in range(2):
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"gripper0_finger_joint{i + 1}")
            ids = np.flatnonzero(m.actuator_trnid[:, 0] == jid)
            assert len(ids) == 1
            d.ctrl[ids[0]] = np.array([0.02, -0.02])[i] + 0.02 * after[i]
        mujoco.mj_forward(m, d)
        sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, c["controller_contract"]["controller_site"])
        assert sid >= 0
        np.testing.assert_allclose(d.site_xpos[sid], pose[:3, 3], rtol=0, atol=1e-9)
        np.testing.assert_allclose(d.site_xmat[sid].reshape(3, 3), pose[:3, :3], rtol=0, atol=1e-9)
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        mujoco.mj_jacSite(m, d, jp, jr, sid)
        j = np.vstack([jp, jr])
        np.testing.assert_allclose(j @ d.qvel, twist, rtol=0, atol=1e-8)
        qpos, qvel, time = d.qpos.copy(), d.qvel.copy(), d.time
        mass = np.zeros((m.nv, m.nv))
        mujoco.mj_fullM(m, mass, d.qM)
        free = (j @ np.linalg.inv(mass))[:, dofs]

        def derivative(eps):
            columns = []
            for dof in dofs:
                samples = []
                for sign in (-1, 1):
                    d.qfrc_applied[:] = 0
                    d.qfrc_applied[dof] = sign * eps
                    d.qacc_warmstart[:] = 0
                    mujoco.mj_forward(m, d)
                    np.testing.assert_array_equal(d.qpos, qpos)
                    np.testing.assert_array_equal(d.qvel, qvel)
                    assert d.time == time
                    samples.append(j @ d.qacc.copy())
                columns.append((samples[1] - samples[0]) / (2 * eps))
            return np.column_stack(columns)

        constrained = derivative(1e-4)
        check = derivative(5e-5)
        numerical_difference = float(np.linalg.norm(check - constrained) / max(np.linalg.norm(constrained), 1e-12))
        arm_mass, jarm = mass[np.ix_(dofs, dofs)], j[:, dofs]
        a = jarm @ np.linalg.solve(arm_mass, jarm.T)
        torque_per_wrench = jarm.T @ block_diag(np.linalg.pinv(a[:3, :3]), np.linalg.pinv(a[3:, 3:]))
        groups.append(
            dict(
                anchor_group=group,
                constrained_acceleration_per_torque=constrained.tolist(),
                free_acceleration_per_torque=free.tolist(),
                numerical_relative_difference=numerical_difference,
                numerically_stable=numerical_difference < 1e-3,
                relative_constraint_effect=float(np.linalg.norm(constrained - free) / np.linalg.norm(free)),
                free_acceleration_per_desired_acceleration=(free @ torque_per_wrench).tolist(),
                constrained_acceleration_per_desired_acceleration=(constrained @ torque_per_wrench).tolist(),
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
        control_qualification=False,
        scope="restored-state forward-solve derivative with fixed ctrl and zero warmstart; offline truth; not historical force or endpoint response",
    )
    with (ROOT / "result.json").open("x") as f:
        json.dump(out, f, indent=2, allow_nan=False)
        f.write("\n")
    for row in groups:
        print(
            row["anchor_group"],
            "effect",
            row["relative_constraint_effect"],
            "numerical_delta",
            row["numerical_relative_difference"],
            "stable",
            row["numerically_stable"],
            flush=True,
        )


if __name__ == "__main__":
    main()
