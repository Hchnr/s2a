"""Paired raw-state grasp chronology for case11; descriptive geometry, no intervention."""

import json
import os

import numpy as np
from audit_wam_position import Evidence
from scipy.spatial.transform import Rotation

from s2a.archived_forward_state import verify_forward_state
from s2a.artifacts import REPO
from s2a.durable_stage_supervisor import publish


def main():
    assert os.environ["CUDA_VISIBLE_DEVICES"] == os.environ["LD_LIBRARY_PATH"] == ""
    assert os.environ["MUJOCO_GL"] == "disable"
    import mujoco

    root = REPO / "outputs/current_coupled_grasp_departure_20261026"
    assert not root.exists()
    e = Evidence()
    e.read(__file__)
    e.read("src/s2a/archived_forward_state.py")
    predicate = e.json("outputs/current_coupled_rack_failure_20261026/verified.json")
    assert predicate["complete"] and not predicate["upstream_cause_proven"]
    histories, logs, contracts = [], [], []
    for name in ("current_coupled_control_20261026", "preview_control_20261026"):
        folder = REPO / "outputs" / name / "case11"
        verified = e.json(folder / "independent_verified.json")
        assert verified["complete"] and verified["mechanical_labels_images_passed"]
        case = next(folder.glob("simulation/*/*"))
        rows = [json.loads(line) for line in e.read(case / "steps.jsonl").splitlines()]
        logs.append(rows)
        contracts.append(e.json(case / "controller.json")["contract"])
        manifest = e.json(case / "manifest.json")
        ref = e.json(folder / "collection_result.json")["arrays_ref"]
        e.check(ref["path"], ref["sha256"])
        with np.load(ref["path"], allow_pickle=False) as archive:
            states = archive["sim_states"].copy()
            source = archive["source_pose"].copy()
            opening = archive["opening"].copy()
            offsets, flat = archive["gripper_current_action_offsets"], archive["gripper_current_action_flat"]
            motors = [flat[offsets[i] : offsets[i + 1]].copy() for i in range(len(states))]
        xml = case / "model.xml"
        e.check(xml, manifest["initial_model_xml_sha256"])
        model = mujoco.MjModel.from_xml_path(str(xml))
        data = mujoco.MjData(model)
        bottle = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "wine_bottle_1_main")
        assert bottle >= 0
        bottle_bodies = {bottle}
        for body in range(bottle + 1, model.nbody):
            if int(model.body_parentid[body]) in bottle_bodies:
                bottle_bodies.add(body)
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
        history = []
        for index, state in enumerate(states):
            data.time = state[0]
            data.qpos[:], data.qvel[:] = state[1 : 1 + model.nq], state[1 + model.nq :]
            mujoco.mj_forward(model, data)
            verify_forward_state(
                state, np.r_[data.time, data.qpos, data.qvel], nq=model.nq, quaternion_segments=segments
            )
            pose = np.eye(4)
            pose[:3, :3], pose[:3, 3] = data.xmat[bottle].reshape(3, 3), data.xpos[bottle]
            if index:
                np.testing.assert_allclose(
                    pose[:3, 3], rows[index - 1]["objects"]["wine_bottle_1"]["position"], atol=1e-9, rtol=0
                )
            contacts = []
            for contact in data.contact[: data.ncon]:
                bodies = [int(model.geom_bodyid[g]) for g in (contact.geom1, contact.geom2)]
                names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) or "" for body in bodies]
                if any(body in bottle_bodies for body in bodies) and any(
                    name.startswith(("robot0_", "gripper0_")) for name in names
                ):
                    contacts.append(
                        dict(
                            geoms=sorted(
                                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(g))
                                for g in (contact.geom1, contact.geom2)
                            ),
                            bodies=names,
                            distance=float(contact.dist),
                        )
                    )
            relative = np.linalg.inv(source[index]) @ pose
            history.append(
                dict(
                    endpoint=index,
                    time=float(data.time),
                    source_pose=source[index].tolist(),
                    bottle_pose=pose.tolist(),
                    source_from_bottle=relative.tolist(),
                    opening=float(opening[index]),
                    motor=motors[index].tolist(),
                    contacts=contacts,
                )
            )
        histories.append(history)
    assert contracts[0] == contracts[1] and len(logs[0]) == len(logs[1]) == 281
    comparisons = []
    for index, (left, right) in enumerate(zip(histories[0], histories[1], strict=True)):
        a, b = np.asarray(left["bottle_pose"]), np.asarray(right["bottle_pose"])
        sa, sb = np.asarray(left["source_pose"]), np.asarray(right["source_pose"])
        assert abs(left["time"] - right["time"]) < 1e-9
        if index < len(logs[0]):
            np.testing.assert_allclose(
                logs[0][index]["desired_source"], logs[1][index]["desired_source"], atol=1e-9, rtol=0
            )
        pa, pb = ({tuple(contact["geoms"]) for contact in row["contacts"]} for row in (left, right))
        comparisons.append(
            dict(
                endpoint=index,
                bottle_position_difference_m=float(np.linalg.norm(a[:3, 3] - b[:3, 3])),
                bottle_rotation_difference_rad=float(Rotation.from_matrix(a[:3, :3] @ b[:3, :3].T).magnitude()),
                source_position_difference_m=float(np.linalg.norm(sa[:3, 3] - sb[:3, 3])),
                source_rotation_difference_rad=float(Rotation.from_matrix(sa[:3, :3] @ sb[:3, :3].T).magnitude()),
                opening_difference=abs(left["opening"] - right["opening"]),
                motor_difference=float(np.max(np.abs(np.asarray(left["motor"]) - right["motor"]))),
                contact_pairs_differ=pa != pb,
            )
        )

    def first(predicate):
        return next((row["endpoint"] for row in comparisons if predicate(row)), None)

    command_differences = [
        i for i, (a, b) in enumerate(zip(logs[0], logs[1], strict=True)) if a["command"][-1] != b["command"][-1]
    ]
    summary = dict(
        gripper_command_difference_steps=command_differences,
        first_different_bottle_robot_contact_pairs_endpoint=first(lambda row: row["contact_pairs_differ"]),
        first_bottle_difference_above_1mm_endpoint=first(lambda row: row["bottle_position_difference_m"] > 0.001),
        first_bottle_difference_above_1cm_endpoint=first(lambda row: row["bottle_position_difference_m"] > 0.01),
        first_opening_difference_above_0p05_endpoint=first(lambda row: row["opening_difference"] > 0.05),
        first_motor_difference_endpoint=first(lambda row: row["motor_difference"] > 1e-12),
    )
    e.recheck()
    root.mkdir()
    publish(
        root / "verified.json",
        dict(
            complete=True,
            case_index=11,
            seed=42,
            endpoints=282,
            summary=summary,
            comparisons=comparisons,
            histories=histories,
            source_sha256=e.hashes,
            physics_steps=0,
            new_fits=0,
            model_calls=0,
            reconstructed_forces=False,
            descriptive_thresholds_only=True,
            task_gate_unchanged=True,
            causal_intervention_performed=False,
        ),
    )
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
