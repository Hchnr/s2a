"""Independent actual-length replay of causal-error control and explicit stops."""

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
    root = REPO / "outputs/tracking_v5_causal_inner_continuation_check_20261028"
    require(not root.exists(), "No overwrite or automatic retry")
    owner_ref = reference(REPO / "outputs/tracking_v5_causal_inner_continuation_execution_20261028/result.json")
    owner = read(owner_ref)
    require(owner["complete"] is owner["passed"] is True, "Closed actual native producer owner")
    command = owner["owned_command"]
    exited, released = read(command["exit"]), read(command["release"])
    require(
        exited["actual_wait"] is True
        and exited["returncode"] == 0
        and released["reaped"] is True
        and released["returncode"] == 0,
        "Actual wait0 and reap",
    )
    producer_ref = owner["result_ref"]
    producer = read(producer_ref)
    require(
        producer["complete"] is producer["passed"] is True and 737 <= producer["physics_steps"] <= 816,
        "Actual complete producer with bounded physical count",
    )
    require([b["branch"] for b in producer["branches"]] == ["original", "causal_error"], "Both paired branches")
    reg = read(producer["registration_ref"])
    worker = read(reg["worker_ref"])
    manifest = read(reg["manifest_ref"])
    chunks = read(reg["chunks_ref"])
    chunk = chunks[46]
    require(
        chunk["step"] == 368 and reg["prefix_steps"] == 368 and reg["maximum_continuation_steps"] == 40,
        "Same already-generated query46, remaining40 references",
    )
    originals = [json.loads(x) for x in checked(reg["rows_ref"]).read_text().splitlines()]
    tool_ref = reference(Path(manifest["runtime_config"]["tool_contract"]))
    tool = np.asarray(read(tool_ref)["source_from_tcp"])
    mapping = np.asarray(read(reference(Path(worker["physical_case_dir"]) / "controller.json"))["source_from_site"])
    registration = publish(
        root / "registration.json",
        dict(
            producer_owner_ref=owner_ref,
            producer_ref=producer_ref,
            producer_registration_ref=producer["registration_ref"],
            source_ref=reference(Path(__file__)),
            tool_ref=tool_ref,
            physics_budget=816,
            world_model_calls=0,
            fresh_cases=0,
            seed=42,
            automatic_retry=False,
            scope="Saved commands only; no candidate/production solver import; all actual endpoints/inner samples including stop endpoint",
        ),
    )
    guard = read(CONTROL_REF)["cv2_guard_registration"]
    restore_cv2_environment(guard["exact_probe_ref"], guard["runtime_source_refs"])
    from collect_wam_native_training import endpoint, render_anchor

    from s2a.cameras import render_current_cameras
    from s2a.checkpoint_simulation import object_progress
    from s2a.geometry import inverse, pose_error
    from s2a.gripper_motor_state import preserve_motor_target
    from s2a.libero_simulation import current_observation, make_environment, source_pose

    env = None
    result = dict(complete=False, passed=False, registration_ref=registration, physics_steps=0, branches=[])
    try:
        targets = np.asarray(chunk["effective_targets_tcp"])
        generated = targets @ inverse(tool) @ mapping
        require(len(targets) == 40, "All40 original generated targets")
        for branch in producer["branches"]:
            rows = read(branch["steps_ref"])
            n = branch["continuation_steps"]
            require(len(rows) == n and 0 < n <= 40, "Exact actual continuation length")
            require(branch["completed_all40"] == (n == 40), "Full-length flag exact")
            if branch["branch"] == "original":
                require(n == 40 and branch["stopped"] is None, "Full baseline")
            elif n < 40:
                stop = branch["stopped"]
                require(
                    stop["step"] == 368 + n and stop["action_dispatched"] is False,
                    "Stop at saved endpoint before dispatch",
                )
                require(stop["diagnostic"]["witness_found"] is False, "Explicit no-witness, not task completion")
                require(
                    stop["history_before"]["completed_periods"] == n,
                    "Stop history contains exactly executed candidate periods",
                )
            else:
                require(branch["stopped"] is None, "No stop on full40")
            require(
                branch["all40_success"] == (n == 40 and all(bool(r["success"]) for r in rows)),
                "Task success requires actual full40",
            )
            require(branch["terminal"] == bool(rows[-1]["success"]), "Actual last executed task predicate")
            require(branch["success_steps"] == sum(bool(r["success"]) for r in rows), "Actual task count")
            require(
                branch["grasp_steps"] == sum(bool(r["objects"]["akita_black_bowl_1"]["grasp_contact"]) for r in rows),
                "Actual grasp count",
            )
            total = 368 + n
            commands = np.asarray([r["command"] for r in originals[:368]] + [r["command"] for r in rows])
            require(
                commands.shape == (total, 7) and np.isfinite(commands).all() and np.max(abs(commands)) <= 1 + 1e-12,
                "Finite actual native commands",
            )
            require(
                np.max(abs(np.diff(np.vstack((np.zeros(7), commands)), axis=0)[:, :6])) <= 0.25 + 1e-12,
                "Original arm rate limit",
            )
            with np.load(checked(branch["states_ref"]), allow_pickle=False) as archive:
                states = archive["sim_states"].copy()
            require(len(states) == total + 1, "All actual saved endpoints including final stop")
            env, _ = make_environment(
                manifest["runtime_config"], root / branch["branch"] / "native", manifest["record"], 0
            )
            robot = env.robots[0]
            current_observation(env)
            robot.gripper.current_action = preserve_motor_target(robot.gripper.current_action)
            bounds = np.asarray(
                [env.sim.model.jnt_range[env.sim.model.joint_name2id(f"robot0_joint{k}")] for k in range(1, 8)]
            )
            inner = []
            native_step = env.sim.step

            def observed_step(*args, **kwargs):
                native_step(*args, **kwargs)
                inner.append(
                    (float(env.sim.data.time), np.asarray(env.sim.data.qpos)[robot._ref_joint_pos_indexes].copy())
                )

            maximum = inner_difference = 0.0
            all_inner, errors, successes = [], [], []
            for i in range(total + 1):
                sample, receipt = endpoint(env, i)
                actual = env.sim.get_state().flatten()
                np.testing.assert_allclose(actual, states[i], atol=1e-8, rtol=0)
                maximum = max(maximum, float(np.max(abs(actual - states[i]))))
                if i in (0, 16, 56):
                    render_anchor(env, i, sample, receipt, root / branch["branch"])
                if i == total:
                    break
                if i % 8 == 0:
                    render_current_cameras(env, manifest["runtime_config"]["render_resolution"])
                if i >= 368:
                    row = rows[i - 368]
                    require(row["step"] == i and row["cursor"] == i - 368, "Exact causal reference schedule")
                    expected_references = generated[i - 368 : min(i - 368 + 4, 40)]
                    np.testing.assert_allclose(row["seed"]["references"], expected_references, atol=1e-8, rtol=0)
                    inner.clear()
                    env.sim.step = observed_step
                env.step(commands[i])
                result["physics_steps"] += 1
                env.sim.step = native_step
                obs = current_observation(env)
                if i >= 368:
                    require(len(inner) == len(row["actual_inner"]) == 25, "Full25 inner endpoints")
                    saved_q = np.asarray([v["q"] for v in row["actual_inner"]])
                    actual_q = np.asarray([v[1] for v in inner])
                    np.testing.assert_allclose(actual_q, saved_q, atol=1e-8, rtol=0)
                    np.testing.assert_allclose(
                        [v[0] for v in inner], [v["time"] for v in row["actual_inner"]], atol=1e-12, rtol=0
                    )
                    inner_difference = max(inner_difference, float(np.max(abs(actual_q - saved_q))))
                    all_inner.extend(actual_q)
                    np.testing.assert_allclose(
                        env.sim.data.qpos[robot._ref_joint_pos_indexes], row["actual_joint_after"], atol=1e-12, rtol=0
                    )
                    error = pose_error(targets[i - 368], source_pose(obs) @ tool)
                    np.testing.assert_allclose(error, row["tcp_error"], atol=1e-8, rtol=0)
                    progress = object_progress(env)
                    require(progress["success"] == row["success"], "Same actual task predicate")
                    require(
                        progress["objects"]["akita_black_bowl_1"]["grasp_contact"]
                        == row["objects"]["akita_black_bowl_1"]["grasp_contact"],
                        "Same native grasp",
                    )
                    errors.append(error)
                    successes.append(bool(progress["success"]))
            q = np.asarray(all_inner)
            require(q.shape == (25 * n, 7), "All actually executed inner joint samples")
            margin = np.minimum(q - bounds[:, 0], bounds[:, 1] - q)
            saved = read(branch["inner_ref"])
            np.testing.assert_allclose(float(margin.min()), saved["minimum_margin_rad"], atol=1e-12, rtol=0)
            errors = np.asarray(errors)
            result["branches"].append(
                dict(
                    branch=branch["branch"],
                    maximum_state_difference=maximum,
                    maximum_inner_joint_difference=inner_difference,
                    minimum_inner_margin_rad=float(margin.min()),
                    inner_samples=25 * n,
                    continuation_steps=n,
                    completed_all40=n == 40,
                    stopped=branch["stopped"],
                    outside_xml_inner_samples=int(np.sum(np.any(margin < 0, axis=1))),
                    below_requested_buffer_inner_samples=int(np.sum(np.any(margin < 0.05, axis=1))),
                    position_rmse_m=float(np.sqrt(np.mean(np.sum(errors[:, :3] ** 2, axis=1)))),
                    so3_rmse_rad=float(np.sqrt(np.mean(np.sum(errors[:, 3:] ** 2, axis=1)))),
                    success_steps=sum(successes),
                )
            )
            env.close()
            env = None
        checked(producer_ref)
        require(result["physics_steps"] == producer["physics_steps"], "Exact independent physics budget")
        result.update(
            complete=True,
            passed=True,
            world_model_calls=0,
            fresh_cases=0,
            scope="Independent replay of actual saved commands including explicit no-dispatch stop; not causal-model or controller/task qualification",
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
