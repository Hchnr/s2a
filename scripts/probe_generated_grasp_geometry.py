"""Static kinematic counterfactuals at generated closure poses, never a rollout."""

import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from s2a.artifacts import REPO
from s2a.clean_training_runtime import checked, publish, read, reference, require
from s2a.geometry import inverse, transform
from s2a.terminal_development_policy import case_order


def main():
    root = REPO / "outputs/generated_grasp_geometry_20261027"
    require(not root.exists(), "No overwrite or automatic retry")
    sources = [reference(Path(__file__))]
    reg = publish(
        root / "registration.json",
        dict(
            source_refs=sources,
            case_indices=[0, 1, 2],
            opening_threshold=0.35,
            static_opening_variants=["predicted", "zero"],
            physics_budget=0,
            world_model_calls=0,
            fresh_cases=0,
            scope="Relocate robot base to realize reference site pose, freeze object at query state; not joint feasibility, dynamics or actual grasp",
        ),
    )
    cases = []
    base = REPO / "outputs/tracking_preview_wam_development_v3_20261027/development/cases"
    for index in (0, 1, 2):
        worker_ref = reference(base / case_order()[index]["case_id"] / "worker_result.json")
        worker = read(worker_ref)
        require(worker["complete"] is worker["passed"] is True, "Closed600 producer")
        folder = Path(worker["physical_case_dir"])
        model_ref = reference(folder / "model.xml")
        model = mujoco.MjModel.from_xml_path(str(checked(model_ref)))
        data = mujoco.MjData(model)
        manifest = read(reference(folder / "manifest.json"))
        control = read(reference(folder / "controller.json"))
        tool = np.asarray(read(reference(Path(manifest["runtime_config"]["tool_contract"])))["source_from_tcp"])
        mapping = np.asarray(control["source_from_site"])
        chunks_ref = reference(folder / "chunks.json")
        chunks = read(chunks_ref)
        rid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot0_base")
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripper0_grip_site")
        require(
            rid >= 0 and sid >= 0 and model.body_parentid[rid] == 0 and model.body_jntnum[rid] == 0,
            "Fixed world-child robot base",
        )
        original_pos, original_quat = model.body_pos[rid].copy(), model.body_quat[rid].copy()
        names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or "" for i in range(model.ngeom)]
        object_geoms = {
            i
            for i, n in enumerate(names)
            if n.startswith("akita_black_bowl_1_") and (model.geom_contype[i] or model.geom_conaffinity[i])
        }
        pads = [names.index(f"gripper0_finger{k}_pad_collision") for k in (1, 2)]
        qids = [
            model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"gripper0_finger_joint{k}")]
            for k in (1, 2)
        ]
        body = int(model.geom_bodyid[min(object_geoms)])

        def contact():
            touching = [False, False]
            for c in data.contact[: data.ncon]:
                pair = {int(c.geom1), int(c.geom2)}
                if c.dist <= 0 and pair & object_geoms:
                    for k, pad in enumerate(pads):
                        touching[k] = touching[k] or pad in pair
            return touching

        def restore(state):
            model.body_pos[rid] = original_pos
            model.body_quat[rid] = original_quat
            data.time = state[0]
            data.qpos[:] = state[1 : 1 + model.nq]
            data.qvel[:] = state[1 + model.nq : 1 + model.nq + model.nv]
            if model.na:
                data.act[:] = state[-model.na :]
            mujoco.mj_forward(model, data)

        def place(desired):
            current = transform(data.site_xpos[sid].copy(), data.site_xmat[sid].reshape(3, 3).copy())
            oldbase = transform(data.xpos[rid].copy(), data.xmat[rid].reshape(3, 3).copy())
            newbase = desired @ inverse(current) @ oldbase
            model.body_pos[rid] = newbase[:3, 3]
            q = Rotation.from_matrix(newbase[:3, :3]).as_quat()
            model.body_quat[rid] = q[[3, 0, 1, 2]]
            object_before = data.xpos[body].copy()
            mujoco.mj_forward(model, data)
            np.testing.assert_allclose(data.site_xpos[sid], desired[:3, 3], atol=1e-9, rtol=0)
            np.testing.assert_allclose(data.site_xmat[sid].reshape(3, 3), desired[:3, :3], atol=1e-9, rtol=0)
            np.testing.assert_array_equal(data.xpos[body], object_before)

        records = []
        with np.load(checked(worker["arrays_ref"]), allow_pickle=False) as arrays:
            require(
                len(arrays["sim_states"]) == 601 and len(chunks) == 75, "Complete saved endpoints and generated blocks"
            )
            for query, chunk in enumerate(chunks):
                step = query * 8
                require(chunk["step"] == step, "Fixed query endpoint")
                state = arrays["sim_states"][step]
                restore(state)
                native_contact = contact()
                native_pose = transform(data.site_xpos[sid].copy(), data.site_xmat[sid].reshape(3, 3).copy())
                place(native_pose)
                require(contact() == native_contact, "Identity relocation preserves native contacts")
                close = np.flatnonzero(np.asarray(chunk["openings"]) < 0.35)
                if not len(close):
                    continue
                cursor = int(close[0])
                desired = np.asarray(chunk["effective_targets_tcp"])[cursor] @ inverse(tool) @ mapping
                variants = []
                for label, opening in [("predicted", float(chunk["openings"][cursor])), ("zero", 0.0)]:
                    restore(state)
                    gap = 0.001 + 0.079 * opening
                    data.qpos[qids] = [gap / 2, -gap / 2]
                    mujoco.mj_forward(model, data)
                    place(desired)
                    touching = contact()
                    variants.append(
                        dict(variant=label, opening=opening, pad_contacts=touching, both_pads=all(touching))
                    )
                records.append(dict(query=query, step=step, first_closure_index=cursor, variants=variants))
        cases.append(
            dict(
                case=case_order()[index],
                worker_ref=worker_ref,
                model_ref=model_ref,
                chunks_ref=chunks_ref,
                queries_with_closure=len(records),
                both_pads_predicted=sum(r["variants"][0]["both_pads"] for r in records),
                both_pads_zero=sum(r["variants"][1]["both_pads"] for r in records),
                queries=records,
            )
        )
    print(
        publish(
            root / "verified.json",
            dict(
                complete=True,
                passed=True,
                registration_ref=reg,
                cases=cases,
                physics_steps=0,
                world_model_calls=0,
                fresh_cases=0,
                scope="Static relocation and finger-opening counterfactuals against current frozen object; not causal rollout, feasible robot motion, force closure or task qualification",
            ),
        ),
        flush=True,
    )
    print(
        json.dumps(
            [{k: c[k] for k in ("queries_with_closure", "both_pads_predicted", "both_pads_zero")} for c in cases]
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
