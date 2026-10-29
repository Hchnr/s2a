"""Compare saved-state endpoint finger contacts and grasp geometry with a positive control."""

import json
from pathlib import Path

import mujoco
import numpy as np

from s2a.artifacts import REPO
from s2a.clean_training_runtime import checked, publish, read, reference, require
from s2a.terminal_development_policy import case_order


def describe(model_ref, arrays_ref):
    model = mujoco.MjModel.from_xml_path(str(checked(model_ref)))
    data = mujoco.MjData(model)
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or "" for i in range(model.ngeom)]
    object_geoms = {
        i
        for i, name in enumerate(names)
        if name.startswith("akita_black_bowl_1_") and (model.geom_contype[i] or model.geom_conaffinity[i])
    }
    fingers = [
        {i for i, name in enumerate(names) if name.startswith(f"gripper0_finger{k}_") and "collision" in name}
        for k in (1, 2)
    ]
    pads = [names.index(f"gripper0_finger{k}_pad_collision") for k in (1, 2)]
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "akita_black_bowl_1_main")
    if body < 0:
        body = int(model.geom_bodyid[min(object_geoms)])
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripper0_grip_site")
    require(object_geoms and min(pads) >= 0 and sid >= 0, "Actual collision geoms and control site")
    records = []
    with np.load(checked(arrays_ref), allow_pickle=False) as arrays:
        states = arrays["sim_states"]
        require(len(states) == 601, "All600 intervals and601 endpoints")
        for step, state in enumerate(states):
            require(len(state) == 1 + model.nq + model.nv + model.na, "Native state layout")
            data.time = state[0]
            data.qpos[:] = state[1 : 1 + model.nq]
            data.qvel[:] = state[1 + model.nq : 1 + model.nq + model.nv]
            if model.na:
                data.act[:] = state[-model.na :]
            mujoco.mj_forward(model, data)
            rotation = data.site_xmat[sid].reshape(3, 3)
            site = data.site_xpos[sid]
            center = data.xpos[body]
            midpoint = np.mean(data.geom_xpos[pads], axis=0)
            touching = [False, False]
            pad_touching = [False, False]
            for contact in data.contact[: data.ncon]:
                if contact.dist > 0:
                    continue
                pair = {int(contact.geom1), int(contact.geom2)}
                if pair & object_geoms:
                    for k in (0, 1):
                        touching[k] = touching[k] or bool(pair & fingers[k])
                        pad_touching[k] = pad_touching[k] or pads[k] in pair
            records.append(
                dict(
                    step=step,
                    opening=float(arrays["opening"][step]),
                    object_center_in_site_m=((center - site) @ rotation).tolist(),
                    object_center_from_pad_midpoint_in_site_axes_m=((center - midpoint) @ rotation).tolist(),
                    pad_center_distance_m=float(np.linalg.norm(data.geom_xpos[pads[0]] - data.geom_xpos[pads[1]])),
                    center_to_pad_midpoint_m=float(np.linalg.norm(center - midpoint)),
                    finger_target_contact=touching,
                    pad_target_contact=pad_touching,
                )
            )
    closest = min(records, key=lambda r: r["center_to_pad_midpoint_m"])
    closed = [r for r in records if r["opening"] < 0.35]
    return dict(
        model_ref=model_ref,
        arrays_ref=arrays_ref,
        endpoint_count=len(records),
        any_finger_target_contact_endpoints=sum(any(r["finger_target_contact"]) for r in records),
        both_fingers_target_contact_endpoints=sum(all(r["finger_target_contact"]) for r in records),
        both_pads_target_contact_endpoints=sum(all(r["pad_target_contact"]) for r in records),
        closed_opening_endpoints=len(closed),
        closest_object_center_to_pad_midpoint=closest,
        minimum_closed_center_to_pad_midpoint_m=None
        if not closed
        else min(r["center_to_pad_midpoint_m"] for r in closed),
        endpoints=records,
    )


def main():
    root = REPO / "outputs/tracking_grasp_geometry_20261027"
    require(not root.exists(), "No overwrite")
    entries = []
    base = REPO / "outputs/tracking_preview_wam_development_v3_20261027/development/cases"
    for i in (0, 1, 2):
        worker_ref = reference(base / case_order()[i]["case_id"] / "worker_result.json")
        worker = read(worker_ref)
        require(worker["complete"] is worker["passed"] is True, "Closed original case")
        entries.append(
            dict(
                label=case_order()[i]["case_id"],
                producer_ref=worker_ref,
                model_ref=reference(Path(worker["physical_case_dir"]) / "model.xml"),
                arrays_ref=worker["arrays_ref"],
            )
        )
    positive_ref = reference(REPO / "outputs/tracking_correct_reference_ordered_20261027/case15_candidate/result.json")
    positive = read(positive_ref)
    entries.append(
        dict(
            label="correct_reference_positive_case15",
            producer_ref=positive_ref,
            model_ref=positive["model_ref"],
            arrays_ref=positive["arrays_ref"],
        )
    )
    registration = publish(
        root / "registration.json",
        dict(
            source_ref=reference(Path(__file__)),
            entries=entries,
            physics_budget=0,
            world_model_calls=0,
            fresh_cases=0,
            scope="Saved endpoint geometry and fresh mj_forward contacts; no temporal rollout or contact forces",
        ),
    )
    results = []
    for entry in entries:
        detail = describe(entry["model_ref"], entry["arrays_ref"])
        ref = publish(root / entry["label"] / "geometry.json", detail)
        results.append(
            dict(
                label=entry["label"],
                geometry_ref=ref,
                **{k: v for k, v in detail.items() if k not in ("endpoints", "model_ref", "arrays_ref")},
            )
        )
    with (
        np.load(checked(entries[0]["arrays_ref"])) as negative,
        np.load(checked(entries[-1]["arrays_ref"])) as positive,
    ):
        np.testing.assert_array_equal(negative["sim_states"][0], positive["sim_states"][0])
    print(
        publish(
            root / "verified.json",
            dict(
                complete=True,
                passed=True,
                registration_ref=registration,
                cases=results,
                raw42_positive_initial_state_equal=True,
                physics_steps=0,
                world_model_calls=0,
                fresh_cases=0,
                scope="Independent saved-endpoint geometry descriptors; object center is not a grasp target, no causal isolation or new task qualification",
            ),
        ),
        flush=True,
    )
    print(
        json.dumps(
            [
                {k: v for k, v in r.items() if k not in ("geometry_ref", "closest_object_center_to_pad_midpoint")}
                for r in results
            ]
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
