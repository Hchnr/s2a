"""224 restore recorded states without integration and compare coupled prediction."""

import json

import mujoco
import numpy as np
from audit_wam_position import Evidence
from pose_velocity_arm_independent_replay import PoseVelocityArmReplay
from scipy.linalg import expm

from s2a.artifacts import REPO
from s2a.durable_stage_supervisor import publish
from s2a.frozen_osc_response import goal_response
from s2a.osc import OSCInterface


def evolution(coupling, kp, kd):
    system = np.zeros((18, 18))
    system[:6, 6:12] = np.eye(6)
    system[6:12, :6] = -coupling * np.asarray(kp)[None, :]
    system[6:12, 6:12] = -coupling * np.asarray(kd)[None, :]
    system[6:12, 12:] = coupling * np.asarray(kp)[None, :]
    return expm(system * 0.05)


def main():
    root = REPO / "outputs/preview_frozen_coupling_20261026"
    assert not root.exists()
    e = Evidence()
    for path in (
        __file__,
        "src/s2a/frozen_osc_response.py",
        "scripts/pose_velocity_arm_independent_replay.py",
        "plans/v0.0.1/224_preview_coupling_diagnosis_20261026.md",
    ):
        e.read(path)
    closure = e.json("outputs/preview_control_serialized_20261026/closure_verified.json")
    assert closure["complete"] and closure["owner_normal_wait_verified"]
    for path, signature in closure["source_sha256"].items():
        e.check(path, signature)
    results = []
    for index in (6, 9, 10, 12, 15):
        directory = REPO / f"outputs/preview_control_20261026/case{index:02d}"
        case = next((directory / "simulation").glob("*/*"))
        rows = [json.loads(line) for line in e.read(case / "steps.jsonl").splitlines()]
        contract = e.json(case / "controller.json")["contract"]
        manifest = e.json(case / "manifest.json")
        for path, signature in manifest["execution_sha256"].items():
            if path.endswith((".png", ".stl", ".obj", ".xml", "/osc.py", "/control_utils.py", "/base_controller.py")):
                e.check(path, signature)
        xml = case / "model.xml"
        e.check(xml, manifest["initial_model_xml_sha256"])
        model = mujoco.MjModel.from_xml_path(str(xml))
        data = mujoco.MjData(model)
        ref = e.json(directory / "collection_result.json")["arrays_ref"]
        e.check(ref["path"], ref["sha256"])
        with np.load(ref["path"], allow_pickle=False) as archive:
            states = archive["sim_states"].copy()
            poses = archive["controller_site_pose"].copy()
        assert states.shape == (len(rows) + 1, 1 + model.nq + model.nv)
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, contract["controller_site"])
        assert sid >= 0
        dofs = []
        for j in range(1, 8):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"robot0_joint{j}")
            assert jid >= 0
            dofs.append(model.jnt_dofadr[jid])
        interface = OSCInterface(*(contract[key] for key in ("input_min", "input_max", "output_min", "output_max")))
        scalar = PoseVelocityArmReplay(interface, contract["kp"], contract["kd"], 0.05)
        independent = evolution(np.eye(6), contract["kp"], contract["kd"])
        np.testing.assert_allclose(independent[:6, 6:12], np.diag(scalar.momentum), atol=1e-12, rtol=0)
        np.testing.assert_allclose(
            independent[6:12, 6:12], np.diag(1 - scalar.response - scalar.kd * scalar.momentum), atol=1e-12, rtol=0
        )
        selected = (
            range(max(0, len(rows) - 100), len(rows), 5)
            if index in (9, 10, 15)
            else range(190, 241, 5)
            if index == 6
            else range(70, 146, 5)
        )
        observations = []
        for step in selected:
            assert step < len(rows)
            state = states[step]
            data.time = state[0]
            data.qpos[:] = state[1 : 1 + model.nq]
            data.qvel[:] = state[1 + model.nq :]
            mujoco.mj_forward(model, data)
            np.testing.assert_allclose(data.site_xpos[sid], poses[step, :3, 3], atol=1e-9, rtol=0)
            np.testing.assert_allclose(data.site_xmat[sid].reshape(3, 3), poses[step, :3, :3], atol=1e-9, rtol=0)
            jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
            mujoco.mj_jacSite(model, data, jp, jr, sid)
            jacobian = np.vstack([jp, jr])
            row = rows[step]
            twist = np.asarray(row["qualified_forward_input"]["x_site_twist"])[0]
            np.testing.assert_allclose(jacobian @ data.qvel, twist, atol=1e-8, rtol=0)
            mass = np.zeros((model.nv, model.nv))
            mujoco.mj_fullM(model, mass, data.qM)
            arm_mass, arm_j = mass[np.ix_(dofs, dofs)], jacobian[:, dofs]
            assert np.linalg.matrix_rank(arm_j) == 6
            response, inverse_inertia, coupling = goal_response(arm_mass, arm_j, contract["kp"], contract["kd"], 0.05)
            transition = evolution(coupling, contract["kp"], contract["kd"])
            np.testing.assert_allclose(transition[:12, 12:], response, atol=1e-12, rtol=0)
            goal = np.asarray(row["qualified_forward_input"]["action_goal_delta"])[0]
            prediction = response @ goal + transition[:12, 6:12] @ twist
            actual = poses[step + 1, :3, 3] - poses[step, :3, 3]
            nominal = np.asarray(row["nominal_predicted_motion"])[:3]
            observations.append(
                dict(
                    step=step,
                    nominal_error_m=float(np.linalg.norm(actual - nominal)),
                    coupled_error_m=float(np.linalg.norm(actual - prediction[:3])),
                    actual_displacement=actual.tolist(),
                    nominal_prediction=nominal.tolist(),
                    coupled_prediction=prediction[:3].tolist(),
                    rotation_goal_contribution=(response[:3, 3:] @ goal[3:]).tolist(),
                    angular_velocity_contribution=(transition[:3, 9:12] @ twist[3:]).tolist(),
                    cartesian_inverse_inertia=inverse_inertia.tolist(),
                    acceleration_coupling=coupling.tolist(),
                    goal_delta=goal.tolist(),
                    current_twist=twist.tolist(),
                    reference_speed=float(np.linalg.norm(row["site_velocity_reference"]["velocity"])),
                )
            )
        metrics = dict(
            case=index,
            samples=len(observations),
            nominal_displacement_rmse_m=float(np.sqrt(np.mean([row["nominal_error_m"] ** 2 for row in observations]))),
            coupled_displacement_rmse_m=float(np.sqrt(np.mean([row["coupled_error_m"] ** 2 for row in observations]))),
            improved_samples=sum(row["coupled_error_m"] < row["nominal_error_m"] for row in observations),
        )
        results.append(dict(metrics=metrics, observations=observations))
    e.recheck()
    root.mkdir()
    publish(
        root / "verified.json",
        dict(
            complete=True,
            cases=results,
            source_sha256=e.hashes,
            mujoco_version=mujoco.__version__,
            physics_steps=0,
            new_fits=0,
            model_calls=0,
            seed=42,
            actual_uncoupling_field_observed=False,
            assumed_uncoupling=True,
            evidence_level="frozen current-state local linear mechanism; contacts/torque clipping/Jdot/finite rotations omitted",
            control_qualified=False,
        ),
    )
    print(json.dumps([case["metrics"] for case in results]))


if __name__ == "__main__":
    main()
