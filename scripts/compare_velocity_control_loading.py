"""Read-only verified-prefix080/074 loading and aperture comparison; no policy selection."""

import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

CURRENT = Path("outputs/velocity_model_control_20261025")
PRIOR = Path("outputs/residual_motor_physical_20261025_v2")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    config_path = Path("configs/velocity_model_control_20261025.json")
    config = json.loads(config_path.read_text())
    prior_config = Path("configs/residual_motor_physical_20261025_v2.json")
    assert config["profiles"] == json.loads(prior_config.read_text())["profiles"]
    hashes = {str(p): digest(p) for p in [config_path, prior_config, Path(__file__)]}
    results = []
    for case in range(18):
        label = f"case{case:02d}"
        exit_path = CURRENT / "verification" / label / "exit.json"
        if not exit_path.is_file():
            continue
        if json.loads(exit_path.read_text())["returncode"] != 0:
            continue
        folder = CURRENT / label
        verified_path = folder / "independent_verified.json"
        prior_verified = PRIOR / label / "independent_verified.json"
        verified = json.loads(verified_path.read_text())
        old = json.loads(prior_verified.read_text())
        assert verified["mechanical_labels_images_passed"] and old["mechanical_labels_images_passed"]
        profile = config["profiles"][case]
        if profile["format"] != "native_endpoints":
            continue
        ref = Path(profile["trajectory_ref"]["path"])
        assert digest(ref) == profile["trajectory_ref"]["sha256"]
        xml = next(folder.glob("simulation/*/*/model.xml"))
        model = mujoco.MjModel.from_xml_path(str(xml.resolve()))
        coeff = []
        for i in (1, 2):
            aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"gripper0_gripper_finger_joint{i}")
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"gripper0_finger_joint{i}")
            assert (
                model.actuator_gaintype[aid] == mujoco.mjtGain.mjGAIN_FIXED
                and model.actuator_biastype[aid] == mujoco.mjtBias.mjBIAS_AFFINE
            )
            assert (
                model.actuator_dyntype[aid] == mujoco.mjtDyn.mjDYN_NONE
                and model.actuator_trntype[aid] == mujoco.mjtTrn.mjTRN_JOINT
            )
            assert (
                model.actuator_trnid[aid, 0] == jid
                and model.qpos0[model.jnt_qposadr[jid]] == 0
                and model.actuator_biasprm[aid, 2] == 0
                and model.actuator_forcelimited[aid]
            )
            np.testing.assert_array_equal(model.actuator_gear[aid], [1, 0, 0, 0, 0, 0])
            coeff.append(
                (
                    float(model.actuator_gainprm[aid, 0]),
                    float(model.actuator_biasprm[aid, 0]),
                    float(model.actuator_biasprm[aid, 1]),
                    model.actuator_forcerange[aid].copy(),
                )
            )
        actual_path = folder / "actual_trajectory.npz"
        old_path = PRIOR / label / "actual_trajectory.npz"
        for p in [exit_path, verified_path, prior_verified, xml, ref, actual_path, old_path]:
            hashes[str(p)] = digest(p)

        def read(path):
            with np.load(path, allow_pickle=False) as data:
                q = data["opening"][1:].copy()
                qpos = data["gripper_qpos"][1:].copy()
                off = data["gripper_current_action_offsets"]
                flat = data["gripper_current_action_flat"]
                m = np.stack([flat[off[k] : off[k + 1]] for k in range(1, len(q) + 1)])
            ctrl = np.array([0.02, -0.02])[None, :] + 0.02 * m
            force = np.column_stack(
                [
                    np.clip(g * ctrl[:, i] + bias + position * qpos[:, i], limits[0], limits[1])
                    for i, (g, bias, position, limits) in enumerate(coeff)
                ]
            )
            return q, m, np.maximum(force * np.array([-1.0, 1.0]), 0.0)

        nq, nm, nload = read(ref)
        aq, am, aload = read(actual_path)
        bq, bm, bload = read(old_path)
        assert len(nq) == len(aq) == len(bq) == profile["physics_steps"]
        loaded = nm[:, 0] < -0.99
        result = dict(
            case=case,
            reference=profile["trajectory_ref"],
            diagnostic_mask="post native motor first component<-.99;offline only",
            native_loaded_cycles=int(loaded.sum()),
            current080=dict(
                stable=verified["control"]["outcome"]["stable"],
                opening_rmse=float(np.sqrt(np.mean((aq - nq) ** 2))),
                mean_closing_actuator_N_on_native_loaded=aload[loaded].mean(axis=0).tolist() if loaded.any() else None,
            ),
            prior074=dict(
                stable=old["control"]["outcome"]["stable"],
                opening_rmse=float(np.sqrt(np.mean((bq - nq) ** 2))),
                mean_closing_actuator_N_on_native_loaded=bload[loaded].mean(axis=0).tolist() if loaded.any() else None,
            ),
            native_mean_closing_actuator_N=nload[loaded].mean(axis=0).tolist() if loaded.any() else None,
        )
        results.append(result)
        print(
            case,
            "stable080/074",
            result["current080"]["stable"],
            result["prior074"]["stable"],
            "qrmse080/074",
            round(result["current080"]["opening_rmse"], 5),
            round(result["prior074"]["opening_rmse"], 5),
            "closingload080",
            result["current080"]["mean_closing_actuator_N_on_native_loaded"],
        )
    assert all(digest(p) == sha for p, sha in hashes.items())
    out = Path("outputs/velocity_control_loading_pairs_20261025")
    out.mkdir(exist_ok=True)
    artifact = out / f"verified_prefix_{len(results):02d}.json"
    payload = (
        json.dumps(
            dict(
                complete_verified_snapshot=True,
                cases=results,
                source_sha256=hashes,
                evidence="compiled fixed-affine actuator force reconstruction;not contact force",
                dynamics_model_compilations=len(results),
                new_fits=0,
                model_calls=0,
                physics_steps=0,
                fresh_cases=0,
            ),
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    with artifact.open("x") as stream:
        stream.write(payload)


if __name__ == "__main__":
    main()
