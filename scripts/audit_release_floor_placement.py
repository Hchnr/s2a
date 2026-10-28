"""Independent saved-state decomposition of the bowl-on-plate task predicate."""

import json
from pathlib import Path

import mujoco
import numpy as np

from s2a.artifacts import REPO
from s2a.clean_training_runtime import checked, read, reference, require
from s2a.preview_result_publication import publish_result as publish


def main():
    root = REPO / "outputs/gripper_release_floor_placement_20261028"
    require(not root.exists(), "No overwrite")
    producer_ref = reference(REPO / "outputs/gripper_release_floor_pair_v2_20261028/result.json")
    producer = read(producer_ref)
    require(producer["complete"] is producer["passed"] is True, "Closed physical pair")
    reg = read(producer["registration_ref"])
    worker = read(reg["worker_ref"])
    folder = Path(worker["physical_case_dir"])
    model_ref = reference(folder / "model.xml")
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
                entries.append(
                    dict(
                        endpoint=index,
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
                scope="Independent static endpoint predicate decomposition; no causal placement-policy qualification",
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
