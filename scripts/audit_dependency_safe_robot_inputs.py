"""Bind logged planner inputs to saved native endpoints without stepping physics."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from s2a.artifacts import REPO
from s2a.clean_training_runtime import checked, publish, read, reference, require
from s2a.current_coupled_response import build_response, nominal_prediction
from s2a.geometry import transform
from s2a.osc import OSCInterface


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-index", type=int, required=True, choices=range(18))
    args = parser.parse_args()
    root = REPO / f"outputs/dependency_safe_robot_inputs_20261027/case{args.case_index:02d}"
    require(not root.exists(), "No overwrite")
    prepared = read(reference(REPO / "outputs/tracking_correct_reference_20261027/prepared.json"))
    entry = prepared["profiles"][args.case_index]
    run = REPO / f"outputs/dependency_safe_correct_reference_20261027/case{args.case_index:02d}_candidate"
    result_ref = reference(run / "result.json")
    result = read(result_ref)
    require(result["complete"] is result["passed"] is True, "Complete run")
    rows_ref = reference(run / "steps.json")
    rows = read(rows_ref)
    contract = read(entry["controller_ref"])["contract"]
    interface = OSCInterface(*(contract[k] for k in ("input_min", "input_max", "output_min", "output_max")))
    model = mujoco.MjModel.from_xml_path(str(checked(result["model_ref"])))
    data = mujoco.MjData(model)
    joints = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"robot0_joint{k}") for k in range(1, 8)]
    require(min(joints) >= 0, "Actual robot joints")
    qids, dofs = model.jnt_qposadr[joints], model.jnt_dofadr[joints]
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripper0_grip_site")
    require(sid >= 0, "Actual control site")

    def restore(state):
        require(len(state) == 1 + model.nq + model.nv + model.na, "Native flattened state layout")
        data.time = state[0]
        data.qpos[:] = state[1 : 1 + model.nq]
        data.qvel[:] = state[1 + model.nq : 1 + model.nq + model.nv]
        if model.na:
            data.act[:] = state[-model.na :]
        mujoco.mj_forward(model, data)
        jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jp, jr, sid)
        jac = np.vstack((jp, jr))
        pose = transform(data.site_xpos[sid].copy(), data.site_xmat[sid].reshape(3, 3).copy())
        return pose, jac @ data.qvel, jac[:, dofs]

    innovation = np.zeros(12)
    limited_innovation = np.zeros(12)
    previous = np.zeros(6)
    with np.load(checked(result["arrays_ref"]), allow_pickle=False) as arrays:
        require(len(rows) + 1 == len(arrays["sim_states"]), "Every endpoint")
        for i, row in enumerate(rows):
            site, twist, jac = restore(arrays["sim_states"][i])
            mass = np.zeros((model.nv, model.nv))
            mujoco.mj_fullM(model, mass, data.qM)
            d = row["dynamics"]
            for actual, logged in [
                (site, row["site_pose"]),
                (twist, row["site_twist"]),
                (jac, d["jacobian"]),
                (mass[np.ix_(dofs, dofs)], d["mass"]),
                (data.qpos[qids], d["robot_joint_position"]),
                (data.qvel[dofs], d["robot_joint_velocity"]),
                (innovation, row["coupled_innovation_before"]),
                (limited_innovation, row["refinement"]["history_before"]),
            ]:
                np.testing.assert_allclose(actual, logged, atol=1e-8, rtol=0)
            f = row["refinement"]
            plan = np.asarray(f["planned_commands"])
            require(plan.shape == (min(8, min(40, len(rows) - (i // 8) * 8) - i % 8), 6), "Complete available plan")
            require(
                np.isfinite(plan).all()
                and np.all(plan >= interface.input_min - 1e-12)
                and np.all(plan <= interface.input_max + 1e-12),
                "Plan bounds",
            )
            require(np.max(abs(np.diff(np.vstack((previous, plan)), axis=0))) <= 0.25 + 1e-12, "Plan rate")
            np.testing.assert_array_equal(plan[0], row["command"][:6])
            require(f["innovation_applied"] is False, "No undeclared limited-history correction")
            relative = np.asarray(row["controller_goal_ori"]) @ site[:3, :3].T
            response = build_response(
                np.asarray(d["mass"]), np.asarray(d["jacobian"]), np.asarray(d["kp"]), np.asarray(d["kd"]), relative
            )
            predicted = nominal_prediction(response, twist, interface.decode(plan[0])[:3])
            next_site, next_twist, _ = restore(arrays["sim_states"][i + 1])
            actual_rotation = Rotation.from_matrix(next_site[:3, :3] @ site[:3, :3].T)

            def residual(prediction):
                return np.r_[
                    next_site[:3, 3] - site[:3, 3] - prediction[:3],
                    (actual_rotation * Rotation.from_rotvec(prediction[3:6]).inv()).as_rotvec(),
                    0.05 * (next_twist - prediction[6:]),
                ]

            innovation = residual(predicted)
            limited_innovation = residual(np.asarray(f["nominal_response"]))
            np.testing.assert_allclose(innovation, row["coupled_innovation_after"], atol=1e-8, rtol=0)
            previous = plan[0]
    ref = publish(
        root / "verified.json",
        dict(
            complete=True,
            passed=True,
            result_ref=result_ref,
            rows_ref=rows_ref,
            source_ref=reference(Path(__file__)),
            rows=len(rows),
            physics_steps=0,
            world_model_calls=0,
            fresh_cases=0,
            scope="Saved XML/state binding of robot M/J/q/dq/site/twist; completed-period history recurrence and plan bounds. Not proof of optimizer optimality or full absence of undeclared inputs.",
        ),
    )
    print(json.dumps(ref), flush=True)


if __name__ == "__main__":
    main()
