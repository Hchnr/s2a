"""053 fixed six validation rollouts; physical body derived from034 collector.

The terminal control schedule is unchanged; only four native anchors are rendered.
No model, training fit, privileged online task input or future image is used.
"""

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np
from collect_wam_native_training import endpoint, render_anchor, save_arrays, write_once

from s2a.artifacts import sha256
from s2a.geometry import validate_transform
from s2a.native_environment import restore_cv2_environment, strict_native_environment
from s2a.native_training import pack_gripper_caches, synchronous_labels
from s2a.settled_native import HOLD_SEMANTICS, SettledGoalEnvironment, settled_command_schedule
from s2a.synchronous_validation import load_prepared, publish, reference


def require(condition, message):
    if not condition:
        raise ValueError(message)


def collect_case(prepared, prepared_ref, record, case_dir, progress):
    from s2a.libero_simulation import controller_contract, make_environment

    require(sha256(record["expert_path"]) == record["expert_sha256"], "selected native command hash changed")
    native = np.load(record["expert_path"], allow_pickle=False)
    commands = settled_command_schedule(native)
    require(len(native) == record["native_steps"], "native length differs from frozen roster")
    anchors = [j * (len(native) - 40) // 3 for j in range(4)]
    require(anchors == record["anchor_frames"], "anchor roster differs")
    h = validate_transform(prepared["tool_source_from_tcp"])
    policy = dict(prepared["policy_config"])
    policy.update(
        task_id=record["task_id"],
        task_text=record["task"],
        seed=42,
        control_fps=20,
        render_resolution=256,
        camera_observations=False,
        refresh_controller_after_restore=True,
    )
    case_dir.mkdir()
    env = None
    samples, receipts, rendered, sent = [], [], [], []
    try:
        progress["environment_initializations"] += 1
        actual, _ = make_environment(policy, case_dir, record, trial=0)
        env = SettledGoalEnvironment(actual, len(native))
        controller, contract = controller_contract(env)
        require(contract["control_freq"] == 20, "native control frequency changed")
        require(np.isclose(env.env.control_timestep, 0.05, atol=1e-12, rtol=0), "native control dt changed")
        require(env.env.action_dim == 7, "native action dimension changed")
        robot = env.robots[0]
        observation_contract = dict(
            source_position="robot0_eef_pos = actual grip-site position",
            source_rotation="robot0_eef_quat xyzw = actual robot EEF BODY orientation",
            actual_source_site_id=int(robot.eef_site_id),
            actual_source_site_name=env.sim.model.site_id2name(robot.eef_site_id),
            actual_eef_body_name=robot.robot_model.eef_name,
            actual_eef_body_id=int(env.sim.model.body_name2id(robot.robot_model.eef_name)),
            controller_site_name=controller.eef_name,
            controller_state="cached actual fields; no forced endpoint updates",
            source_from_tcp=h,
            tool_contract_ref=prepared["tool_contract_ref"],
            delta_convention="inverse(T_tcp[k]) @ T_tcp[k+1], local right multiply",
            units=dict(position="m", time="s", rotation="column rot6d"),
            labels_normalized=False,
            initial_seed=42,
            initial_trial=0,
            pre_rollout_steps=0,
            initial_controller_refreshed=True,
            camera_observables=False,
            terminal_hold_semantics=HOLD_SEMANTICS,
        )
        for index in range(len(commands) + 1):
            if index:
                actual = commands[index - 1].copy()
                progress["physics_steps"] += 1
                env.step(actual)
                # Record after the call too: the actual argument must not have
                # been modified by the environment's control implementation.
                require(np.array_equal(actual, commands[index - 1]), "environment mutated sent command")
                sent.append(actual)
            sample, receipt = endpoint(env, index)
            if index == 0:
                require(receipt["sim_state_sha256"] == record["initial_state_sha256"], "restored initial changed")
            samples.append(sample)
            receipts.append(receipt)
            with (case_dir / "endpoints.jsonl").open("a") as stream:
                stream.write(json.dumps(receipt, allow_nan=False) + "\n")
                stream.flush()
            if index in anchors:
                rendered.append(render_anchor(env, index, sample, receipt, case_dir))
                progress["current_canvas_images"] += 1
        arrays = {
            key: np.stack([sample[key] for sample in samples]) for key in samples[0] if key != "gripper_current_action"
        }
        arrays.update(pack_gripper_caches([sample["gripper_current_action"] for sample in samples]))
        arrays["native_commands"] = native.copy()
        arrays["sent_commands"] = np.stack(sent)
        arrays.update(synchronous_labels(arrays["source_pose"], arrays["opening"], h, arrays["sim_time"]))
        require(len(samples) == len(native) + 81, "endpoint budget mismatch")
        stable = bool(arrays["native_success"][-40:].all())
        arrays_ref = save_arrays(case_dir / "trajectory.npz", arrays)
        case_result = dict(
            schema_version="s2a.synchronous-validation-case.v1",
            complete=True,
            passed=True,
            native_quality_passed=stable,
            prepared_ref=prepared_ref,
            record=record,
            native_steps=len(native),
            physics_steps=len(sent),
            endpoint_count=len(samples),
            environment_initializations=1,
            current_canvas_images=len(rendered),
            future_rgb_frames=0,
            arrays=arrays_ref,
            array_index={key: dict(shape=list(value.shape), dtype=str(value.dtype)) for key, value in arrays.items()},
            original_dtypes={key: str(value.dtype) for key, value in arrays.items()},
            endpoint_receipts=reference(case_dir / "endpoints.jsonl"),
            controller_contract=contract,
            observation_contract=observation_contract,
            anchors=rendered,
            stable_success_last_steps=40,
            settling_steps=40,
            terminal_verification_steps=40,
            total_hold_steps=80,
            source_hdf5_reads=0,
            native_command_ref=dict(path=record["expert_path"], sha256=record["expert_sha256"]),
        )
        case_ref = write_once(case_dir / "result.json", case_result)
        goals_ref = write_once(case_dir / "controller_goals.json", env.goals)
        return dict(name=record["name"], case=case_ref, goals=goals_ref)
    finally:
        if env is not None:
            env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--dry", action="store_true")
    args = parser.parse_args()
    strict_native_environment()
    pref = reference(args.prepared)
    prepared = load_prepared(pref)
    admission = json.loads(args.admission.read_text())
    require(
        admission["prepared_ref"] == pref and admission["passed"] is True and admission["execution_authorized"] is True,
        "Independent053 metadata admission required",
    )
    require(
        admission["verifier_ref"] == reference(Path(__file__).with_name("verify_wam_synchronous_validation.py")),
        "Metadata verifier changed",
    )
    readiness = restore_cv2_environment(prepared["cv2_probe_ref"], prepared["cv2_runtime_source_refs"])
    root = Path(prepared["root"])
    if args.dry:
        publish(
            root / "actual_cli_dry.json",
            {
                "passed": True,
                "prepared_ref": pref,
                "admission_ref": reference(args.admission),
                "pid": os.getpid(),
                "readiness": readiness,
                "environments": 0,
                "physics_steps": 0,
            },
        )
        print("053 actual CLI dry passed", flush=True)
        return
    dry = json.loads((root / "actual_cli_dry.json").read_text())
    require(
        dry["passed"] is True and dry["prepared_ref"] == pref and dry["admission_ref"] == reference(args.admission),
        "Exact prior CLI dry required",
    )
    folder = root / "collection"
    folder.mkdir(exist_ok=False)
    publish(
        folder / "worker_started.json",
        {
            "pid": os.getpid(),
            "ppid": os.getppid(),
            "argv": sys.argv,
            "prepared_ref": pref,
            "admission_ref": reference(args.admission),
            "readiness": readiness,
        },
    )
    progress = {"environment_initializations": 0, "physics_steps": 0, "current_canvas_images": 0}
    result = {
        "schema_version": "s2a.synchronous-validation-collection.v1",
        "passed": False,
        "complete": False,
        "prepared_ref": pref,
        "cases": [],
        "model_calls": 0,
        "fresh_cases": 0,
    }
    try:
        for record in prepared["records"]:
            result["cases"].append(collect_case(prepared, pref, record, folder / record["name"], progress))
            strict_native_environment()
            load_prepared(pref)
            publish(folder / (record["name"] + "_complete.json"), dict(progress))
            print(json.dumps({"case": record["name"], "progress": progress}), flush=True)
        require(
            progress == {"environment_initializations": 6, "physics_steps": 1212, "current_canvas_images": 24},
            "Fixed053 budget differs",
        )
        result.update(passed=True, complete=True)
    except BaseException as error:
        result["error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        raise
    finally:
        result["progress"] = progress
        publish(folder / "result.json", result)


if __name__ == "__main__":
    main()
