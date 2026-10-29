"""Independent native replay of saved commands, not of the reentry solver."""

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
    root = REPO / "outputs/tracking_v5_reentry_native_check_20261028"
    require(not root.exists(), "No overwrite/retry")
    producer_ref = reference(REPO / "outputs/tracking_v5_reentry_native_v2_20261028/result.json")
    producer = read(producer_ref)
    require(
        producer["complete"] is producer["passed"] is True and producer["physics_steps"] == 768,
        "Closed paired native768",
    )
    reg = read(producer["registration_ref"])
    capture = read(reg["capture_registration_ref"])
    manifest = read(capture["manifest_ref"])
    original = [json.loads(x) for x in checked(capture["rows_ref"]).read_text().splitlines()]
    registration = publish(
        root / "registration.json",
        dict(
            source_ref=reference(Path(__file__)),
            producer_ref=producer_ref,
            producer_registration_ref=producer["registration_ref"],
            physics_budget=768,
            world_model_calls=0,
            fresh_cases=0,
            automatic_retry=False,
            seed=42,
            scope="Replay actual saved commands from two fresh native environments; no candidate solver import",
        ),
    )
    guard = read(CONTROL_REF)["cv2_guard_registration"]
    restore_cv2_environment(guard["exact_probe_ref"], guard["runtime_source_refs"])
    from collect_wam_native_training import endpoint, render_anchor

    from s2a.cameras import render_current_cameras
    from s2a.gripper_motor_state import preserve_motor_target
    from s2a.libero_simulation import current_observation, make_environment

    env = None
    result = dict(complete=False, passed=False, registration_ref=registration, physics_steps=0, branches=[])
    try:
        for branch in producer["branches"]:
            with np.load(checked(branch["states_ref"]), allow_pickle=False) as archive:
                states, commands = archive["sim_states"], archive["commands"]
                require(len(states) == 385 and commands.shape == (384, 7), "Full384 commands/385 states")
                np.testing.assert_array_equal(commands[:383], np.asarray([r["command"] for r in original[:383]]))
                require(commands[383, 6] == original[383]["command"][6], "Gripper unchanged")
                if branch["branch"] == "baseline":
                    np.testing.assert_array_equal(commands[383], original[383]["command"])
                else:
                    np.testing.assert_array_equal(commands[383], reg["candidate_command"])
                env, _ = make_environment(
                    manifest["runtime_config"], root / branch["branch"] / "native", manifest["record"], 0
                )
                robot = env.robots[0]
                current_observation(env)
                robot.gripper.current_action = preserve_motor_target(robot.gripper.current_action)
                maximum = 0.0
                for i in range(385):
                    sample, receipt = endpoint(env, i)
                    actual = env.sim.get_state().flatten()
                    np.testing.assert_allclose(actual, states[i], atol=1e-8, rtol=0)
                    maximum = max(maximum, float(np.max(abs(actual - states[i]))))
                    if i in (0, 16, 56):
                        render_anchor(env, i, sample, receipt, root / branch["branch"])
                    if i == 384:
                        break
                    if i % 8 == 0:
                        render_current_cameras(env, manifest["runtime_config"]["render_resolution"])
                    env.step(commands[i])
                    result["physics_steps"] += 1
                    current_observation(env)
                q = np.asarray(env.sim.data.qpos)[robot._ref_joint_pos_indexes]
                bounds = np.asarray(
                    [env.sim.model.jnt_range[env.sim.model.joint_name2id(f"robot0_joint{k}")] for k in range(1, 8)]
                )
                margin = np.minimum(q - bounds[:, 0], bounds[:, 1] - q)
                np.testing.assert_allclose(margin, branch["final_joint_margin_rad"], atol=1e-12, rtol=0)
                require(bool(env.check_success()) == branch["native_success"], "Same native task predicate")
                result["branches"].append(
                    dict(
                        branch=branch["branch"],
                        maximum_state_difference=maximum,
                        minimum_final_margin_rad=float(margin.min()),
                        endpoints=385,
                        periods=384,
                    )
                )
                env.close()
                env = None
        checked(producer_ref)
        require(result["physics_steps"] == 768, "Exact independent budget")
        result.update(
            complete=True,
            passed=True,
            world_model_calls=0,
            fresh_cases=0,
            scope="Two full saved-command replays; endpoint recovery only, no inner2ms/multi-period/task qualification",
        )
    except BaseException as error:
        result["error"] = repr(error)
        raise
    finally:
        if env is not None:
            env.close()
        publish(root / "verified.json", result)
    print(json.dumps(dict(complete=True, passed=True)), flush=True)


if __name__ == "__main__":
    main()
