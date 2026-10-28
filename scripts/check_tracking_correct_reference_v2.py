"""Independently replay logged actions through native physics and original gates.

This checks execution and performance, not optimizer optimality or a global
joint-feasibility certificate. Replay physics is counted separately.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from s2a.artifacts import REPO
from s2a.clean_training_runtime import checked, publish, read, reference, require
from s2a.geometry import pose_error
from s2a.native_environment import restore_cv2_environment, strict_native_environment
from s2a.terminal_development_execution import CONTROL_REF


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case-index", type=int, required=True, choices=range(18))
    args = p.parse_args()
    strict_native_environment()
    root = REPO / f"outputs/tracking_correct_reference_check_v2_20261027/case{args.case_index:02d}"
    root.mkdir(parents=True, exist_ok=False)
    prepared = read(reference(REPO / "outputs/tracking_correct_reference_20261027/prepared.json"))
    entry = prepared["profiles"][args.case_index]
    profile = entry["profile"]
    runroot = REPO / f"outputs/tracking_correct_reference_20261027/case{args.case_index:02d}_candidate"
    result_ref = reference(runroot / "result.json")
    result = read(result_ref)
    require(result["complete"] is result["passed"] is True, "Full execution")
    require(result["physics_steps"] == profile["physics_steps"], "Original budget")
    rows = read(reference(runroot / "steps.json"))
    chunks = read(reference(runroot / "chunks.json"))
    n = profile["physics_steps"]
    require(len(rows) == n and len(chunks) == (n + 7) // 8, "All cycles/chunks")
    manifest = read(entry["manifest_ref"])
    base = read(entry["execution_config_ref"])
    control = read(entry["controller_ref"])["contract"]
    tool = np.asarray(read(entry["tool_ref"])["source_from_tcp"])
    with np.load(checked(profile["trajectory_ref"]), allow_pickle=False) as source:
        poses = source["source_pose"][1:].copy() if profile["format"] == "native_endpoints" else source["poses"].copy()
        openings = (
            source["opening"][1:].copy() if profile["format"] == "native_endpoints" else source["openings"].copy()
        )
    guard = read(CONTROL_REF)["cv2_guard_registration"]
    restore_cv2_environment(guard["exact_probe_ref"], guard["runtime_source_refs"])
    from s2a.action_reference import decode_action_targets, encode_oracle_targets
    from s2a.gripper_motor_state import preserve_motor_target
    from s2a.libero_simulation import aperture, current_observation, make_environment, source_pose
    from s2a.loaded_velocity_control import LoadedVelocityControl
    from s2a.native_training import state_sha256
    from s2a.osc import OSCInterface
    from s2a.task_acceptance import task_outcome

    from robosuite.utils import transform_utils

    interface = OSCInterface(*(control[k] for k in ("input_min", "input_max", "output_min", "output_max")))
    env = None
    summary = dict(
        complete=False,
        passed=False,
        result_ref=result_ref,
        source_ref=reference(Path(__file__)),
        physics_steps=0,
        world_model_calls=0,
        fresh_cases=0,
    )
    try:
        env, _ = make_environment(manifest["runtime_config"], root / "native", manifest["record"], 0)
        robot = env.robots[0]
        controller = robot.controller
        robot.gripper.current_action = preserve_motor_target(robot.gripper.current_action)
        obs = current_observation(env)
        grip = LoadedVelocityControl(
            read(base["response_parameters_ref"]), aperture(obs), velocity_envelope=base["gripper_velocity_envelope"]
        )
        require(state_sha256(env.sim.get_state().flatten()) == entry["initial_state_sha256"], "Exact initial state")
        errors, successes = [], []
        previous = np.zeros(7)
        max_state_difference = 0.0
        with np.load(checked(result["arrays_ref"]), allow_pickle=False) as arrays:
            require(len(arrays["sim_states"]) == n + 1, "All endpoints")
            for i, row in enumerate(rows):
                np.testing.assert_allclose(env.sim.get_state().flatten(), arrays["sim_states"][i], atol=1e-8, rtol=0)
                if i % 8 == 0:
                    anchor = source_pose(obs) @ tool
                    raw = encode_oracle_targets(anchor, poses[i : i + 40] @ tool, openings[i : i + 40])
                    targets, desired_openings = decode_action_targets(anchor, raw, horizon=len(raw))
                    chunk = chunks[i // 8]
                    require(chunk["step"] == i, "Causal update schedule")
                    for key, value in [
                        ("anchor_tcp", anchor),
                        ("raw128", raw),
                        ("effective_targets_tcp", targets),
                        ("openings", desired_openings),
                    ]:
                        np.testing.assert_allclose(chunk[key], value, atol=1e-8, rtol=0)
                cursor = i % 8
                require(
                    row["step"] == i and row["reference_index"] == cursor and row["chunk_index"] == i // 8,
                    "Reference identity",
                )
                command = np.asarray(row["command"])
                require(command.shape == (7,) and np.isfinite(command).all(), "Finite action")
                require(
                    np.all(command[:6] >= interface.input_min - 1e-12)
                    and np.all(command[:6] <= interface.input_max + 1e-12),
                    "Action bounds",
                )
                require(
                    np.max(abs(command[:6] - previous[:6])) <= 0.25 + 1e-12 and abs(command[6]) <= 1,
                    "Rate and gripper bounds",
                )
                measured = source_pose(obs)
                np.testing.assert_allclose(measured, row["measured_source"], atol=1e-8, rtol=0)
                np.testing.assert_allclose(targets[cursor], row["desired_tcp"], atol=1e-8, rtol=0)
                finger_velocity = np.asarray(obs["robot0_gripper_qvel"])
                g = grip.update(
                    aperture(obs),
                    robot.gripper.current_action,
                    0.05 * (finger_velocity[0] - finger_velocity[1]) / 0.079,
                    targets,
                    desired_openings,
                    cursor,
                    measured @ tool,
                    previous_command=previous[6],
                    dt=0.05,
                )
                np.testing.assert_allclose(command[6], g.command, atol=1e-8, rtol=0)
                site_pos = env.sim.data.site_xpos[env.sim.model.site_name2id(controller.eef_name)].copy()
                site_rot = env.sim.data.site_xmat[env.sim.model.site_name2id(controller.eef_name)].reshape(3, 3).copy()
                delta = interface.decode(command[:6])
                expected_pos = site_pos + delta[:3]
                expected_rot = (
                    controller.goal_ori.copy()
                    if np.all(delta[3:] == 0)
                    else transform_utils.quat2mat(transform_utils.axisangle2quat(delta[3:])) @ site_rot
                )
                env.step(command)
                summary["physics_steps"] += 1
                obs = current_observation(env)
                state = env.sim.get_state().flatten()
                np.testing.assert_allclose(state, arrays["sim_states"][i + 1], atol=1e-8, rtol=0)
                require(state_sha256(arrays["sim_states"][i + 1]) == row["state_sha256"], "Logged endpoint hash")
                max_state_difference = max(
                    max_state_difference, float(np.max(abs(state - arrays["sim_states"][i + 1])))
                )
                for key, expected in [("pos", expected_pos), ("ori", expected_rot)]:
                    np.testing.assert_allclose(getattr(controller, f"goal_{key}"), expected, atol=1e-8, rtol=0)
                    np.testing.assert_allclose(row[f"controller_goal_{key}"], expected, atol=1e-8, rtol=0)
                error = pose_error(targets[cursor], source_pose(obs) @ tool)
                np.testing.assert_allclose(row["tcp_error"], error, atol=1e-8, rtol=0)
                success = bool(env.check_success())
                require(success == row["success"], "Native task predicate")
                errors.append(error)
                successes.append(success)
                previous = command.copy()
        errors = np.asarray(errors)
        pos = float(np.sqrt(np.mean(np.sum(errors[:, :3] ** 2, axis=1))))
        rot = float(np.sqrt(np.mean(np.sum(errors[:, 3:] ** 2, axis=1))))
        outcome = task_outcome(np.asarray(successes), expected_steps=n, stable_steps=40)
        qualified = bool(outcome["stable"] and pos <= 0.01 and rot <= 0.05)
        np.testing.assert_allclose([pos, rot], [result["position_rmse_m"], result["so3_rmse_rad"]], atol=1e-8, rtol=0)
        require(
            outcome == result["outcome"] and qualified == result["performance_qualified"], "Independent original gates"
        )
        summary.update(
            complete=True,
            passed=True,
            performance_qualified=qualified,
            position_rmse_m=pos,
            so3_rmse_rad=rot,
            outcome=outcome,
            maximum_state_difference=max_state_difference,
            scope="Independent native action replay and original performance gates; not optimizer optimality or whole18 qualification",
        )
    except BaseException as error:
        summary["error"] = repr(error)
        raise
    finally:
        if env is not None:
            env.close()
        publish(root / "verified.json", summary)
    print(json.dumps(dict(complete=True, performance_qualified=qualified)), flush=True)


if __name__ == "__main__":
    main()
