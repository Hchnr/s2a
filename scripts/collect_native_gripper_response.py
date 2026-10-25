"""076 native prefix and fixed gripper interventions, current qvel included."""

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from run_matched_oracle_case import CONTROL, CONTROL_SHA

from s2a.artifacts import sha256
from s2a.native_environment import restore_cv2_environment, strict_native_environment

CONFIG = Path("configs/native_gripper_response_interventions_20261025.json")
ROOT = Path("outputs/native_gripper_response_interventions_20261025")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-index", type=int, choices=range(12), required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    strict_native_environment()
    assert sha256(CONTROL) == CONTROL_SHA
    guard = json.loads(CONTROL.read_text())["cv2_guard_registration"]
    restore_cv2_environment(guard["exact_probe_ref"], guard["runtime_source_refs"])
    from collect_wam_native_training import endpoint, save_arrays

    from s2a.libero_simulation import controller_contract, current_observation, make_environment
    from s2a.native_gripper_response import intervention_commands
    from s2a.native_training import pack_gripper_caches, synchronous_labels
    from s2a.settled_native import SettledGoalEnvironment
    from s2a.stability_recovery_simulation import simulator_inputs

    strict_native_environment()
    registration = json.loads(CONFIG.read_text())
    entry = registration["profiles"][args.case_index]
    refs = [CONFIG, CONTROL, Path(entry["trajectory_ref"]["path"]), Path(entry["case_ref"]["path"])]
    assert sha256(refs[2]) == entry["trajectory_ref"]["sha256"] and sha256(refs[3]) == entry["case_ref"]["sha256"]
    prior = json.loads(refs[3].read_text())
    prepared_ref = prior["prepared_ref"]
    prepared_path = Path(prepared_ref["path"])
    assert sha256(prepared_path) == prepared_ref["sha256"]
    refs.append(prepared_path)
    prepared = json.loads(prepared_path.read_text())["original_prepared"]
    policy = dict(prepared["policy_config"])
    record = entry["record"]
    assert record == prior["record"]
    policy.update(
        task_id=record["task_id"],
        task_text=record["task"],
        seed=42,
        control_fps=20,
        render_resolution=256,
        camera_observations=False,
        refresh_controller_after_restore=True,
    )
    with np.load(refs[2], allow_pickle=False) as native:
        commands = native["sent_commands"][: entry["physics_steps"]].copy()
    assert len(commands) == entry["physics_steps"] == entry["anchor_cycle"] + 24
    changed = intervention_commands(commands, entry["anchor_cycle"], entry["gripper_override"])
    np.testing.assert_array_equal(commands[:, :6], changed[:, :6])
    np.testing.assert_array_equal(commands[: entry["anchor_cycle"]], changed[: entry["anchor_cycle"]])
    hashes = {str(p.resolve()): sha256(p) for p in refs}
    if args.dry_run:
        print(
            json.dumps(
                dict(case=args.case_index, branch=entry["branch"], steps=len(changed), ready=True, physics_steps=0)
            )
        )
        return
    if entry["branch"] != "native_baseline":
        for index in (0, 4, 8):
            baseline = json.loads((ROOT / f"case{index:02d}" / "independent_verified.json").read_text())
            assert baseline["baseline_native_match"] and baseline["mechanical_verified"]
    folder = ROOT / f"case{args.case_index:02d}"
    folder.mkdir(parents=True, exist_ok=False)
    (folder / "execution_config.json").write_text(
        json.dumps(dict(profile=entry, policy=policy, source_sha256=hashes), indent=2) + "\n"
    )
    env = None
    samples = []
    receipts = []
    try:
        actual, _ = make_environment(policy, folder, record, trial=0)
        env = SettledGoalEnvironment(actual, record["native_steps"])
        hashes.update(simulator_inputs(env))
        actual_xml = env.sim.model.get_xml()
        (folder / "model.xml").write_text(actual_xml)
        hashes[str((folder / "model.xml").resolve())] = sha256(folder / "model.xml")
        for node in ET.fromstring(actual_xml).iter():
            asset = node.get("file")
            if asset:
                path = Path(asset)
                if not path.is_file():
                    raise FileNotFoundError(f"Actual model asset path unresolved: {asset}")
                hashes[str(path.resolve())] = sha256(path)
        _, contract = controller_contract(env)
        assert contract["control_freq"] == 20
        for k in range(len(changed) + 1):
            if k:
                action = changed[k - 1].copy()
                env.step(action)
                np.testing.assert_array_equal(action, changed[k - 1])
            sample, receipt = endpoint(env, k)
            before = env.sim.get_state().flatten().copy()
            obs = current_observation(env)
            direct = np.array([env.sim.data.qvel[i] for i in env.robots[0]._ref_gripper_joint_vel_indexes])
            np.testing.assert_allclose(obs["robot0_gripper_qvel"], direct, rtol=0, atol=1e-12)
            np.testing.assert_array_equal(before, env.sim.get_state().flatten())
            sample["gripper_qvel"] = direct.copy()
            samples.append(sample)
            receipts.append(receipt)
            with (folder / "endpoints.jsonl").open("a") as stream:
                stream.write(json.dumps(receipt) + "\n")
        arrays = {key: np.stack([s[key] for s in samples]) for key in samples[0] if key != "gripper_current_action"}
        arrays.update(pack_gripper_caches([s["gripper_current_action"] for s in samples]))
        arrays.update(
            synchronous_labels(
                arrays["source_pose"],
                arrays["opening"],
                np.asarray(prepared["tool_source_from_tcp"]),
                arrays["sim_time"],
            )
        )
        arrays["sent_commands"] = changed
        array_ref = save_arrays(folder / "actual_trajectory.npz", arrays)
        assert all(sha256(p) == s for p, s in hashes.items())
        output = dict(
            complete=True,
            case=args.case_index,
            profile=entry,
            physics_steps=len(changed),
            arrays_ref=array_ref,
            controller_contract=contract,
            source_sha256=hashes,
            gripper_velocity_sensor_matches_current_sim=True,
            model_calls=0,
            fits=0,
            fresh_cases=0,
            arm_goal_protocol="original SettledGoalEnvironment native-end measured-site latch",
        )
        with (folder / "collection_result.json").open("x") as stream:
            json.dump(output, stream, indent=2)
            stream.write("\n")
        print(json.dumps(dict(case=args.case_index, complete=True, physics_steps=len(changed))))
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
