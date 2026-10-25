"""Offline actuator loading reconstruction; no contact-force or task-success inference."""

import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path("outputs/velocity_model_control_20261025")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    case = ROOT / "case01"
    config_path = Path("configs/velocity_model_control_20261025.json")
    config = json.loads(config_path.read_text())
    source = Path(config["profiles"][1]["trajectory_ref"]["path"])
    log = next(case.glob("simulation/*/*/steps.jsonl"))
    xml = log.parent / "model.xml"
    actual = case / "actual_trajectory.npz"
    paths = [config_path, source, log, xml, actual, Path(__file__)]
    hashes = {str(p): digest(p) for p in paths}
    model = mujoco.MjModel.from_xml_path(str(xml.resolve()))
    parameters = []
    for index in (1, 2):
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"gripper0_gripper_finger_joint{index}")
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"gripper0_finger_joint{index}")
        assert model.actuator_trnid[aid, 0] == jid and model.actuator_trntype[aid] == mujoco.mjtTrn.mjTRN_JOINT
        assert (
            model.actuator_gaintype[aid] == mujoco.mjtGain.mjGAIN_FIXED
            and model.actuator_biastype[aid] == mujoco.mjtBias.mjBIAS_AFFINE
        )
        assert model.actuator_dyntype[aid] == mujoco.mjtDyn.mjDYN_NONE
        np.testing.assert_array_equal(model.actuator_gear[aid], [1, 0, 0, 0, 0, 0])
        assert model.qpos0[model.jnt_qposadr[jid]] == 0
        assert model.actuator_biasprm[aid, 2] == 0 and model.actuator_forcelimited[aid]
        parameters.append(
            dict(
                name=f"finger{index}",
                gain=float(model.actuator_gainprm[aid, 0]),
                bias=float(model.actuator_biasprm[aid, 0]),
                position_gain=float(model.actuator_biasprm[aid, 1]),
                force_range=model.actuator_forcerange[aid].tolist(),
                ctrl_range=model.actuator_ctrlrange[aid].tolist(),
            )
        )
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    n = len(rows)
    with np.load(actual, allow_pickle=False) as a, np.load(source, allow_pickle=False) as b:

        def loading(qpos, motor):
            ctrl = np.array([0.02, -0.02])[None, :] + 0.02 * motor
            force = np.column_stack(
                [
                    np.clip(p["gain"] * ctrl[:, i] + p["bias"] + p["position_gain"] * qpos[:, i], *p["force_range"])
                    for i, p in enumerate(parameters)
                ]
            )
            closing = np.maximum(force * np.array([-1.0, 1.0]), 0)
            return ctrl, force, closing

        native_motor = np.stack(
            [
                b["gripper_current_action_flat"][
                    b["gripper_current_action_offsets"][k] : b["gripper_current_action_offsets"][k + 1]
                ]
                for k in range(1, n + 1)
            ]
        )
        actual_motor = np.array([row["motor_target_after"] for row in rows])
        aq = a["gripper_qpos"][1:]
        bq = b["gripper_qpos"][1 : n + 1]
        ac, af, al = loading(aq, actual_motor)
        bc, bf, bl = loading(bq, native_motor)
        np.testing.assert_allclose(ac, [row["actual_gripper_actuator_control"] for row in rows], rtol=0, atol=1e-12)
        opening_error = a["opening"][1:] - np.array([row["opening_target"] for row in rows])
        interval = slice(90, 150)
        result = dict(
            case=1,
            interval_cycles=[90, 150],
            per_finger_actual_mean_closing_actuator_N=al[interval].mean(axis=0).tolist(),
            per_finger_native_mean_closing_actuator_N=bl[interval].mean(axis=0).tolist(),
            opening_rmse_interval=float(np.sqrt(np.mean(opening_error[interval] ** 2))),
            actual_motor_mean=actual_motor[interval].mean(axis=0).tolist(),
            native_motor_mean=native_motor[interval].mean(axis=0).tolist(),
            sample_cycle100=dict(
                actual_qpos=aq[100].tolist(),
                native_qpos=bq[100].tolist(),
                actual_actuator_force=af[100].tolist(),
                native_actuator_force=bf[100].tolist(),
            ),
        )
    assert all(digest(p) == sha for p, sha in hashes.items())
    output = dict(
        complete=True,
        evidence="fixed/affine actuator force reconstructed from compiled current model and endpoint ctrl/qpos; not object contact force",
        parameters=parameters,
        result=result,
        source_sha256=hashes,
        dynamics_model_compilations=1,
        physics_steps=0,
        model_calls=0,
        new_fits=0,
        fresh_cases=0,
    )
    payload = json.dumps(output, indent=2, allow_nan=False) + "\n"
    with (case / "actuator_loading_diagnosis.json").open("x") as stream:
        stream.write(payload)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
