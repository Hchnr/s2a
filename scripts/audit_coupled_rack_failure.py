"""Raw endpoint geometry for the first coupled-control task failure; no integration."""

import json
import os
from pathlib import Path

import numpy as np
from audit_wam_position import Evidence

from s2a.archived_forward_state import verify_forward_state
from s2a.artifacts import REPO
from s2a.durable_stage_supervisor import publish


def main():
    assert os.environ["CUDA_VISIBLE_DEVICES"] == os.environ["LD_LIBRARY_PATH"] == ""
    assert os.environ["MUJOCO_GL"] == "disable"
    import mujoco

    root = REPO / "outputs/current_coupled_rack_failure_20261026"
    assert not root.exists()
    e = Evidence()
    e.read(__file__)
    e.read("src/s2a/archived_forward_state.py")
    registration = e.json("configs/current_coupled_control_registered_20261026.json")
    assert registration["profiles"][11]["record"]["task"] == "put the wine bottle on the rack"
    cases = []
    for name in ("current_coupled_control_20261026", "preview_control_20261026"):
        folder = REPO / "outputs" / name / "case11"
        verified = e.json(folder / "independent_verified.json")
        assert verified["complete"] and verified["mechanical_labels_images_passed"]
        case = next(folder.glob("simulation/*/*"))
        rows = [json.loads(line) for line in e.read(case / "steps.jsonl").splitlines()]
        manifest = e.json(case / "manifest.json")
        library = Path(manifest["runtime_config"]["libero_root"]) / "libero/libero"
        for relative in (
            "envs/objects/site_object.py",
            "envs/object_states/base_object_states.py",
            "envs/predicates/base_predicates.py",
        ):
            source = library / relative
            e.check(source, manifest["simulator_source_sha256"][str(source)])
        bddl = library / "bddl_files/libero_goal/put_the_wine_bottle_on_the_rack.bddl"
        assert b"(And (On wine_bottle_1 wine_rack_1_top_region))" in e.read(bddl)
        arrays_ref = e.json(folder / "collection_result.json")["arrays_ref"]
        e.check(arrays_ref["path"], arrays_ref["sha256"])
        with np.load(arrays_ref["path"], allow_pickle=False) as archive:
            states = archive["sim_states"].copy()
        xml = case / "model.xml"
        e.check(xml, manifest["initial_model_xml_sha256"])
        model = mujoco.MjModel.from_xml_path(str(xml))
        data = mujoco.MjData(model)
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "wine_rack_1_top_region")
        bottle = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "wine_bottle_1_main")
        rack = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "wine_rack_1_main")
        assert min(sid, bottle, rack) >= 0
        segments = []
        for joint in range(model.njnt):
            kind = int(model.jnt_type[joint])
            if kind in (0, 1):
                segments.append(
                    dict(
                        joint=mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint),
                        qpos_start=int(model.jnt_qposadr[joint]) + (3 if kind == 0 else 0),
                    )
                )

        def descendants(body):
            result = {body}
            for candidate in range(body + 1, model.nbody):
                if int(model.body_parentid[candidate]) in result:
                    result.add(candidate)
            return result

        bottle_bodies, rack_bodies = descendants(bottle), descendants(rack)
        checks = []
        assert len(states) == len(rows) + 1 == registration["profiles"][11]["physics_steps"] + 1
        for step, row in enumerate(rows):
            state = states[step + 1]
            data.time = state[0]
            data.qpos[:], data.qvel[:] = state[1 : 1 + model.nq], state[1 + model.nq :]
            mujoco.mj_forward(model, data)
            verify_forward_state(
                state, np.r_[data.time, data.qpos, data.qvel], nq=model.nq, quaternion_segments=segments
            )
            position = data.xpos[bottle].copy()
            np.testing.assert_allclose(position, row["objects"]["wine_bottle_1"]["position"], atol=1e-9, rtol=0)
            site_position, site_rotation, size = (
                data.site_xpos[sid].copy(),
                data.site_xmat[sid].reshape(3, 3).copy(),
                model.site_size[sid].copy(),
            )
            # Match the byte-bound vendor SiteObject.under convention, including its matrix direction.
            delta = site_rotation @ (position - site_position)
            margins = np.r_[size[:2] - np.abs(delta[:2]), delta[2] - (size[2] - 0.005), size[2] + 0.10 - delta[2]]
            under = bool(np.all(margins > 0))
            contacts = []
            for contact in data.contact[: data.ncon]:
                bodies = [int(model.geom_bodyid[g]) for g in (contact.geom1, contact.geom2)]
                if (bodies[0] in bottle_bodies and bodies[1] in rack_bodies) or (
                    bodies[1] in bottle_bodies and bodies[0] in rack_bodies
                ):
                    contacts.append(
                        dict(
                            geoms=[
                                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(g))
                                for g in (contact.geom1, contact.geom2)
                            ],
                            distance=float(contact.dist),
                        )
                    )
            reconstructed = under and bool(contacts)
            checks.append(
                dict(
                    step=step,
                    logged_success=row["success"],
                    reconstructed_endpoint_success=reconstructed,
                    under=under,
                    rack_contact=bool(contacts),
                    region_margins_m=margins.tolist(),
                    bottle_position=position.tolist(),
                    bottle_rotation=data.xmat[bottle].reshape(3, 3).tolist(),
                    region_position=site_position.tolist(),
                    contacts=contacts,
                )
            )
        assert all(item["reconstructed_endpoint_success"] == item["logged_success"] for item in checks[-40:])
        cases.append(
            dict(
                name=name,
                outcome=verified["control"]["outcome"],
                tracking=verified["control"]["tracking"]["admitted"],
                all_endpoint_predicate_disagreements=[
                    item["step"] for item in checks if item["reconstructed_endpoint_success"] != item["logged_success"]
                ],
                terminal=checks[-1],
                endpoints=checks,
            )
        )
    e.recheck()
    root.mkdir()
    publish(
        root / "verified.json",
        dict(
            complete=True,
            case_index=11,
            cases=cases,
            source_sha256=e.hashes,
            physics_steps=0,
            new_fits=0,
            model_calls=0,
            seed=42,
            reconstructed_forces=False,
            evidence_level="saved endpoint FK and collision geometry;no integration, warmstart/force reconstruction or causal intervention",
            upstream_cause_proven=False,
        ),
    )
    print(
        json.dumps(
            [
                dict(
                    name=case["name"],
                    terminal=case["terminal"],
                    disagreements=case["all_endpoint_predicate_disagreements"],
                )
                for case in cases
            ]
        )
    )


if __name__ == "__main__":
    main()
