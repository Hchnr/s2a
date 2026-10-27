"""091 no-integration frozen-inertia response diagnostic on all six old anchors."""

import json
from pathlib import Path

import mujoco
import numpy as np
from audit_wam_position import Evidence

from s2a.frozen_osc_response import goal_response

ROOT = Path("outputs/frozen_inertial_coupling_diagnosis_20261025")


def main():
    e = Evidence()
    for p in (
        __file__,
        "src/s2a/frozen_osc_response.py",
        "plans/v0.0.1/091_frozen_inertial_coupling_diagnosis_20261025.md",
    ):
        e.read(p)
    closed = e.json("outputs/paired_arm_response_20261025/closure_verified.json")
    assert closed["complete"] and closed["data_qualified"] and closed["owner_normal_wait_verified"]
    analysis_path = Path("outputs/paired_arm_response_analysis_20261025/complete_06.json")
    analysis = e.json(analysis_path)
    assert analysis["complete"] and analysis["verified_anchor_groups"] == 6
    rows = []
    for group in range(6):
        folder = Path("outputs/paired_arm_response_20261025") / f"case{group * 7:02d}"
        collection_path = folder / "collection_result.json"
        e.check(collection_path, closed["source_sha256"][str(collection_path.resolve())])
        c = e.json(collection_path)
        assert c["profile"]["branch"] == "baseline"
        xml = folder / "model.xml"
        e.check(xml, c["source_sha256"][str(xml.resolve())])
        # Every actual model asset was fingerprinted during collection.
        for p, signature in c["source_sha256"].items():
            if p.endswith((".png", ".stl", ".obj", ".xml", "/osc.py", "/control_utils.py", "/base_controller.py")):
                e.check(p, signature)
        model = mujoco.MjModel.from_xml_path(str(xml.resolve()))
        data = mujoco.MjData(model)
        ref = c["arrays_ref"]
        e.check(ref["path"], ref["sha256"])
        anchor = c["profile"]["anchor_cycle"]
        contract = c["controller_contract"]
        with np.load(ref["path"], allow_pickle=False) as cache:
            state = cache["sim_states"][anchor].copy()
            pose = cache["controller_site_pose"][anchor].copy()
            twist = cache["source_twist"][anchor].copy()
        assert len(state) == 1 + model.nq + model.nv
        data.time = state[0]
        data.qpos[:] = state[1 : 1 + model.nq]
        data.qvel[:] = state[1 + model.nq :]
        mujoco.mj_forward(model, data)
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, contract["controller_site"])
        assert sid >= 0
        np.testing.assert_allclose(data.site_xpos[sid], pose[:3, 3], rtol=0, atol=1e-9)
        np.testing.assert_allclose(data.site_xmat[sid].reshape(3, 3), pose[:3, :3], rtol=0, atol=1e-9)
        jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jp, jr, sid)
        full_j = np.vstack([jp, jr])
        np.testing.assert_allclose(full_j @ data.qvel, twist, rtol=0, atol=1e-8)
        dofs = []
        for i in range(1, 8):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"robot0_joint{i}")
            assert jid >= 0
            dofs.append(model.jnt_dofadr[jid])
        mass = np.zeros((model.nv, model.nv))
        mujoco.mj_fullM(model, mass, data.qM)
        j = full_j[:, dofs]
        assert np.linalg.matrix_rank(j) == 6
        response, a, b = goal_response(mass[np.ix_(dofs, dofs)], j, contract["kp"], contract["kd"], 0.05)
        scale = (np.array(contract["output_max"])[:3] - np.array(contract["output_min"])[:3]) / (
            np.array(contract["input_max"])[:3] - np.array(contract["input_min"])[:3]
        )
        original = analysis["groups"][group]
        assert original["anchor_group"] == group and original["anchor_cycle"] == anchor
        metrics = {}
        for key, sl in (
            ("position", slice(0, 3)),
            ("so3_increment", slice(3, 6)),
            ("linear_velocity", slice(6, 9)),
            ("angular_velocity", slice(9, 12)),
        ):
            measured = np.array(original["central_per_normalized_input"][key])
            predicted = response[sl, :3] * scale[None, :]
            if key == "position":
                nominal = np.array(original["nominal_position_per_normalized_input"])
            elif key == "linear_velocity":
                nominal = np.array(original["nominal_velocity_per_normalized_input"])
            else:
                nominal = np.zeros((3, 3))
            error, base_error = float(np.linalg.norm(predicted - measured)), float(np.linalg.norm(nominal - measured))
            metrics[key] = dict(
                predicted=predicted.tolist(),
                measured=measured.tolist(),
                frozen_error=error,
                nominal_error=base_error,
                improves=error <= base_error,
            )
        rows.append(
            dict(
                anchor_group=group,
                metrics=metrics,
                cartesian_inverse_inertia=a.tolist(),
                acceleration_coupling=b.tolist(),
            )
        )
    e.recheck()
    ROOT.mkdir(exist_ok=False)
    result = dict(
        complete=True,
        groups=rows,
        source_sha256=e.hashes,
        physics_steps=0,
        model_calls=0,
        fresh_cases=0,
        new_fits=0,
        scope="frozen unconstrained small-angle dynamics assuming uncoupled OSC; actual flag not directly measured",
        control_qualification=False,
    )
    with (ROOT / "result.json").open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")
    for row in rows:
        print(
            row["anchor_group"],
            {k: (m["frozen_error"], m["nominal_error"]) for k, m in row["metrics"].items()},
            flush=True,
        )


if __name__ == "__main__":
    main()
