"""Reconstruct actual600-period causal gripper inputs and independently audit decisions."""

import json
from dataclasses import asdict
from pathlib import Path

import mujoco
import numpy as np
from loaded_velocity_release_floor_independent import LoadedVelocityReplay

from s2a.artifacts import REPO
from s2a.clean_training_runtime import checked, read, reference, require
from s2a.geometry import inverse
from s2a.loaded_velocity_release_floor import LoadedVelocityControl
from s2a.preview_result_publication import publish_result as publish
from s2a.preview_result_publication import result_json


def main():
    root = REPO / "outputs/gripper_release_floor_decisions_v2_20261028"
    require(not root.exists(), "No overwrite")
    pair_ref = reference(REPO / "outputs/gripper_release_floor_pair_v2_20261028/result.json")
    pair = read(pair_ref)
    require(pair["complete"] is pair["passed"] is True, "Closed paired experiment")
    reg = read(pair["registration_ref"])
    worker = read(reg["worker_ref"])
    folder = Path(worker["physical_case_dir"])
    config = read(worker["config_ref"])
    manifest = read(reg["manifest_ref"])
    contract_ref = reference(folder / "controller.json")
    contract = read(contract_ref)
    model_ref = reference(folder / "model.xml")
    tool_ref = reference(Path(manifest["runtime_config"]["tool_contract"]))
    offset = inverse(np.asarray(contract["source_from_site"])) @ np.asarray(read(tool_ref)["source_from_tcp"])
    candidate = pair["branches"][1]
    rows = read(candidate["steps_ref"])
    chunks = read(reg["chunks_ref"])
    parameters = read(config["response_parameters_ref"])
    require(len(rows) == 600 and len(chunks) == 75, "Original600/75 reference delivery")
    registration = publish(
        root / "registration.json",
        dict(
            pair_ref=pair_ref,
            source_refs=[
                reference(Path(__file__)),
                reference(REPO / "scripts/gripper_release_floor_independent.py"),
                reference(REPO / "scripts/loaded_velocity_release_floor_independent.py"),
            ],
            model_ref=model_ref,
            contract_ref=contract_ref,
            tool_ref=tool_ref,
            physics_budget=0,
            world_model_calls=0,
            fresh_cases=0,
            scope="Recorded current inputs and actual commands, independent scalar latch and complete MPC audit",
        ),
    )
    model = mujoco.MjModel.from_xml_path(str(checked(model_ref)))
    data = mujoco.MjData(model)
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripper0_grip_site")
    joints = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"gripper0_finger_joint{k}") for k in (1, 2)]
    qids = model.jnt_qposadr[joints]
    dofs = model.jnt_dofadr[joints]
    require(sid >= 0 and min(joints) >= 0, "Actual current gripper geometry")
    motor = np.asarray(contract["aligned_initial_motor_target"], dtype=float)
    previous = 0.0
    counts = {}
    with np.load(checked(candidate["states_ref"]), allow_pickle=False) as archive:
        require(len(archive["sim_states"]) == 601, "All physical endpoints")
        producer = None
        independent = None
        for step, row in enumerate(rows):
            state = archive["sim_states"][step]
            data.time = state[0]
            data.qpos[:] = state[1 : 1 + model.nq]
            data.qvel[:] = state[1 + model.nq : 1 + model.nq + model.nv]
            mujoco.mj_forward(model, data)
            opening = float(np.clip((data.qpos[qids[0]] - data.qpos[qids[1]] - 0.001) / 0.079, 0, 1))
            velocity = 0.05 * float(data.qvel[dofs[0]] - data.qvel[dofs[1]]) / 0.079
            site = np.eye(4)
            site[:3, :3] = data.site_xmat[sid].reshape(3, 3)
            site[:3, 3] = data.site_xpos[sid]
            measured = site @ offset
            if producer is None:
                producer = LoadedVelocityControl(
                    parameters, opening, velocity_envelope=config["gripper_velocity_envelope"]
                )
                independent = LoadedVelocityReplay(
                    parameters, opening, asdict(producer.latch.config), config["gripper_velocity_envelope"]
                )
            chunk = chunks[step // 8]
            require(chunk["step"] == 8 * (step // 8), "Actual generated reference availability")
            poses = np.asarray(chunk["effective_targets_tcp"])
            openings = np.asarray(chunk["openings"])
            computed = producer.update(
                opening, motor, velocity, poses, openings, step % 8, measured, previous_command=previous, dt=0.05
            )
            require(
                computed.command == row["command"][6] and computed.route == row["gripper_route"],
                "Recomputed actual recorded decision",
            )
            verified = independent.step(
                json.loads(result_json(asdict(computed))),
                opening,
                motor,
                velocity,
                poses,
                openings,
                step % 8,
                measured,
                previous,
                0.05,
            )
            counts[verified["route"]] = counts.get(verified["route"], 0) + 1
            previous = float(row["command"][6])
            motor = np.clip(motor + 0.25 * np.array([-1.0, 1.0]) * np.sign(previous), -1, 1)
    print(
        publish(
            root / "verified.json",
            dict(
                complete=True,
                passed=True,
                registration_ref=registration,
                periods=600,
                routes=counts,
                physics_steps=0,
                world_model_calls=0,
                fresh_cases=0,
                scope="Actual candidate commands bound to archived current inputs and independently audited latch/MPC; not new task success",
            ),
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
