"""Independent saved-state decomposition of the bowl-on-plate task predicate."""

import json
from pathlib import Path

import mujoco
import numpy as np

from s2a.artifacts import REPO
from s2a.clean_training_runtime import checked, read, reference, require
from s2a.geometry import inverse
from s2a.preview_result_publication import publish_result as publish


def main():
    root = REPO / "outputs/gripper_release_floor_placement_tracking_20261028"
    require(not root.exists(), "No overwrite")
    producer_ref = reference(REPO / "outputs/gripper_release_floor_pair_v2_20261028/result.json")
    producer = read(producer_ref)
    require(producer["complete"] is producer["passed"] is True, "Closed physical pair")
    reg = read(producer["registration_ref"])
    worker = read(reg["worker_ref"])
    folder = Path(worker["physical_case_dir"])
    model_ref = reference(folder / "model.xml")
    controller_ref = reference(folder / "controller.json")
    tool_ref = reference(Path(read(reg["manifest_ref"])["runtime_config"]["tool_contract"]))
    site_from_tcp = inverse(np.asarray(read(controller_ref)["source_from_site"])) @ np.asarray(
        read(tool_ref)["source_from_tcp"]
    )
    original_rows = [json.loads(x) for x in checked(reg["rows_ref"]).read_text().splitlines()]
    external = Path("/share/project/eai_pwm/datasets/cosmos3_action_tower_training/libero_sim/LIBERO/libero/libero")
    goal_refs = [
        reference(external / "bddl_files/libero_goal/put_the_bowl_on_the_plate.bddl"),
        reference(external / "envs/predicates/base_predicates.py"),
        reference(external / "envs/object_states/base_object_states.py"),
        reference(REPO / "outputs/libero_env311/lib/python3.11/site-packages/robosuite/utils/sim_utils.py"),
    ]
    registration = publish(
        root / "registration.json",
        dict(
            producer_ref=producer_ref,
            source_ref=reference(Path(__file__)),
            model_ref=model_ref,
            controller_ref=controller_ref,
            tool_ref=tool_ref,
            original_rows_ref=reg["rows_ref"],
            goal_refs=goal_refs,
            horizontal_threshold_m=0.03,
            physics_budget=0,
            world_model_calls=0,
            fresh_cases=0,
            scope="Offline saved-state task geometry only; object/goal states never supplied to online policy",
        ),
    )
    model = mujoco.MjModel.from_xml_path(str(checked(model_ref)))
    data = mujoco.MjData(model)
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripper0_grip_site")
    require(sid >= 0, "Actual control site")
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or "" for i in range(model.ngeom)]

    def geometries(prefix):
        return {
            i
            for i, n in enumerate(names)
            if n.startswith(prefix) and (model.geom_contype[i] or model.geom_conaffinity[i])
        }

    bowl = geometries("akita_black_bowl_1_")
    plate = geometries("plate_1_")
    require(bowl and plate, "Actual collision geometry groups")
    bid = int(model.geom_bodyid[min(bowl)])
    pid = int(model.geom_bodyid[min(plate)])
    cases = []
    for branch in producer["branches"]:
        rows = read(branch["steps_ref"])
        entries = []
        with np.load(checked(branch["states_ref"]), allow_pickle=False) as arrays:
            require(len(arrays["sim_states"]) == 601 and len(rows) == 600, "Complete601 states/600 predicates")
            for index, state in enumerate(arrays["sim_states"]):
                require(len(state) == 1 + model.nq + model.nv + model.na, "Flattened native layout")
                data.time = state[0]
                data.qpos[:] = state[1 : 1 + model.nq]
                data.qvel[:] = state[1 + model.nq : 1 + model.nq + model.nv]
                if model.na:
                    data.act[:] = state[-model.na :]
                mujoco.mj_forward(model, data)
                relative = data.xpos[bid] - data.xpos[pid]
                contact = any(
                    (int(c.geom1) in bowl and int(c.geom2) in plate) or (int(c.geom2) in bowl and int(c.geom1) in plate)
                    for c in data.contact[: data.ncon]
                )
                horizontal = float(np.linalg.norm(relative[:2]))
                above = bool(relative[2] >= 0)
                aligned = horizontal < 0.03
                success = contact and above and aligned
                if index:
                    require(success == rows[index - 1]["success"], "Independent original On predicate")
                ideal_horizontal = None
                if index and rows[index - 1]["grasp"]:
                    site = np.eye(4)
                    site[:3, :3] = data.site_xmat[sid].reshape(3, 3)
                    site[:3, 3] = data.site_xpos[sid]
                    tcp = site @ site_from_tcp
                    bowl_pose = np.eye(4)
                    bowl_pose[:3, :3] = data.xmat[bid].reshape(3, 3)
                    bowl_pose[:3, 3] = data.xpos[bid]
                    ideal = np.asarray(original_rows[index - 1]["desired_tcp"]) @ inverse(tcp) @ bowl_pose
                    ideal_horizontal = float(np.linalg.norm((ideal[:3, 3] - data.xpos[pid])[:2]))
                entries.append(
                    dict(
                        endpoint=index,
                        ideal_rigid_horizontal_distance_m=ideal_horizontal,
                        relative_bowl_to_plate_m=relative.copy(),
                        horizontal_distance_m=horizontal,
                        aligned=aligned,
                        above=above,
                        contact=contact,
                        success=success,
                        grasp=False if index == 0 else rows[index - 1]["grasp"],
                    )
                )
        grasped = [r for r in entries if r["grasp"]]
        cases.append(
            dict(
                branch=branch["branch"],
                minimum_rigid_projection_while_grasped_m=min(r["ideal_rigid_horizontal_distance_m"] for r in grasped),
                rigid_projection_aligned_while_grasped_count=sum(
                    r["ideal_rigid_horizontal_distance_m"] < 0.03 for r in grasped
                ),
                states_ref=branch["states_ref"],
                horizontal_aligned_endpoints=sum(r["aligned"] for r in entries),
                contact_endpoints=sum(r["contact"] for r in entries),
                contact_and_aligned_endpoints=sum(r["contact"] and r["aligned"] for r in entries),
                success_endpoints=sum(r["success"] for r in entries),
                closest_horizontal=min(entries, key=lambda r: r["horizontal_distance_m"]),
                closest_horizontal_while_grasped=None
                if not grasped
                else min(grasped, key=lambda r: r["horizontal_distance_m"]),
                endpoints_ref=publish(root / branch["branch"] / "endpoints.json", entries),
            )
        )
    print(
        publish(
            root / "verified.json",
            dict(
                complete=True,
                passed=True,
                registration_ref=registration,
                cases=cases,
                physics_steps=0,
                world_model_calls=0,
                fresh_cases=0,
                scope="Saved-state predicate plus ideal TCP rigid-grasp projection; no actual collision/force/dynamic feasibility or responsive WAM claim",
            ),
        ),
        flush=True,
    )
    print(
        json.dumps(
            [
                {
                    k: v
                    for k, v in c.items()
                    if k in ("branch", "horizontal_aligned_endpoints", "contact_endpoints", "success_endpoints")
                }
                for c in cases
            ]
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
