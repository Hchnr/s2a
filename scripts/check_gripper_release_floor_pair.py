"""Replay candidate commands independently through600 native periods and task predicates."""

import json
from pathlib import Path

import numpy as np

from s2a.artifacts import REPO
from s2a.clean_training_runtime import checked, read, reference, require
from s2a.native_environment import restore_cv2_environment, strict_native_environment
from s2a.preview_result_publication import publish_result as publish
from s2a.terminal_development_execution import CONTROL_REF


def main():
    strict_native_environment()
    root = REPO / "outputs/gripper_release_floor_pair_check_20261028"
    require(not root.exists(), "No overwrite or automatic retry")
    producer_ref = reference(REPO / "outputs/gripper_release_floor_pair_v2_20261028/result.json")
    producer = read(producer_ref)
    require(producer["complete"] is producer["passed"] is True, "Complete1200-period pair")
    reg = read(producer["registration_ref"])
    manifest = read(reg["manifest_ref"])
    candidate = producer["branches"][1]
    require(candidate["branch"] == "candidate" and candidate["physics_steps"] == 600, "Full candidate600")
    rows = read(candidate["steps_ref"])
    require(len(rows) == 600, "Full600 rows")
    original = [json.loads(x) for x in checked(reg["rows_ref"]).read_text().splitlines()]
    source = np.asarray([r["command"] for r in original])
    commands = np.asarray([r["command"] for r in rows])
    np.testing.assert_array_equal(source[:, :6], commands[:, :6])
    changed = np.flatnonzero(abs(source[:, 6] - commands[:, 6]) > 1e-12)
    require(len(changed) > 0, "Actual gripper intervention")
    registration = publish(
        root / "registration.json",
        dict(
            producer_ref=producer_ref,
            source_ref=reference(Path(__file__)),
            physics_budget=600,
            world_model_calls=0,
            fresh_cases=0,
            automatic_retry=False,
        ),
    )
    summary = dict(
        complete=False, passed=False, registration_ref=registration, physics_steps=0, world_model_calls=0, fresh_cases=0
    )
    guard = read(CONTROL_REF)["cv2_guard_registration"]
    restore_cv2_environment(guard["exact_probe_ref"], guard["runtime_source_refs"])
    from collect_wam_native_training import endpoint, render_anchor
    from s2a.cameras import render_current_cameras
    from s2a.checkpoint_simulation import object_progress
    from s2a.gripper_motor_state import preserve_motor_target
    from s2a.libero_simulation import make_environment, current_observation, aperture

    env = None
    try:
        env, _ = make_environment(manifest["runtime_config"], root / "native", manifest["record"], 0)
        robot = env.robots[0]
        current_observation(env)
        robot.gripper.current_action = preserve_motor_target(robot.gripper.current_action)
        initial_height = float(object_progress(env)["objects"]["akita_black_bowl_1"]["position"][2])
        successes = []
        contacts = []
        max_height = 0.0
        maximum_state_difference = 0.0
        with np.load(checked(candidate["states_ref"]), allow_pickle=False) as arrays:
            require(len(arrays["sim_states"]) == 601, "All601 saved endpoints")
            for i in range(601):
                sample, receipt = endpoint(env, i)
                state = env.sim.get_state().flatten()
                np.testing.assert_allclose(state, arrays["sim_states"][i], atol=1e-8, rtol=0)
                maximum_state_difference = max(
                    maximum_state_difference, float(np.max(abs(state - arrays["sim_states"][i])))
                )
                if i in (0, 16, 56):
                    render_anchor(env, i, sample, receipt, root)
                if i == 600:
                    break
                if i % 8 == 0:
                    render_current_cameras(env, manifest["runtime_config"]["render_resolution"])
                command = commands[i]
                require(np.isfinite(command).all() and abs(command[6]) <= 1, "Finite feasible actual command")
                env.step(command)
                summary["physics_steps"] += 1
                obs = current_observation(env)
                progress = object_progress(env)
                obj = progress["objects"]["akita_black_bowl_1"]
                require(progress["success"] == rows[i]["success"], "Independent actual task predicate")
                require(obj["grasp_contact"] == rows[i]["grasp"], "Independent actual grasp predicate")
                height = float(obj["position"][2] - initial_height)
                np.testing.assert_allclose(height, rows[i]["height_gain_m"], atol=1e-8, rtol=0)
                np.testing.assert_allclose(aperture(obs), rows[i]["opening"], atol=1e-8, rtol=0)
                successes.append(bool(progress["success"]))
                contacts.append(bool(obj["grasp_contact"]))
                max_height = max(max_height, height)
        flags = np.asarray(contacts)
        starts = np.flatnonzero(flags & ~np.r_[False, flags[:-1]])
        ends = np.flatnonzero(flags & ~np.r_[flags[1:], False])
        summary.update(
            complete=True,
            passed=True,
            maximum_state_difference=maximum_state_difference,
            success_steps=sum(successes),
            terminal=successes[-1],
            stable_last40=all(successes[-40:]),
            grasp_steps=sum(contacts),
            grasp_intervals=list(zip(starts.tolist(), ends.tolist(), strict=True)),
            max_target_height_gain_m=max_height,
            changed_gripper_commands=len(changed),
            first_changed_step=int(changed[0]),
            scope="Independent600 native command replay; fixed recorded-reference experiment, no responsive WAM/fresh qualification",
        )
    except BaseException as error:
        summary["error"] = repr(error)
        raise
    finally:
        if env is not None:
            env.close()
        publish(root / "verified.json", summary)
    print(json.dumps(dict(complete=True, passed=True)), flush=True)


if __name__ == "__main__":
    main()
