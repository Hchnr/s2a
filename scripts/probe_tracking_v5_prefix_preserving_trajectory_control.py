"""Repair all twelve predicted points while retaining each already qualified prefix."""

import json
import os
import time
from pathlib import Path

import numpy as np

from s2a.artifacts import REPO
from s2a.clean_training_runtime import checked, read, reference, require
from s2a.geometry import inverse
from s2a.mujoco_robot_dynamics import MujocoRobotDynamics
from s2a.native_environment import restore_cv2_environment, strict_native_environment
from s2a.native_goal_joint_response import native_goal_rotation
from s2a.native_goal_trajectory import rollout_native_goal_trajectory
from s2a.osc import OSCInterface
from s2a.prefix_preserving_trajectory_plan import solve_prefix_preserving_trajectory_response
from s2a.preview_result_publication import publish_result as publish
from s2a.terminal_development_execution import CONTROL_REF

ROOT = REPO / "outputs/tracking_v5_prefix_preserving_trajectory_control_20261028"


def main():
    strict_native_environment()
    require(not ROOT.exists(), "No overwrite or automatic retry")
    historical_fixture_migrations = {}
    historical_source_migrations_used = []

    def closed(ref):
        owner = read(ref)
        require(owner["complete"] is owner["passed"] is True, "Closed prerequisite owner")
        launch = read(owner["owned_command"]["launch"])
        exited, release = read(owner["owned_command"]["exit"]), read(owner["owned_command"]["release"])
        require(
            exited["actual_wait"] is True
            and exited["returncode"] == 0
            and release["reaped"] is True
            and release["returncode"] == 0
            and release["signal_sent"] is False
            and release["original_error"] is None
            and launch["child"]["pid"] == release["pid"]
            and launch["child"]["start_ticks"] == release["start_ticks"]
            and launch["selected_environment"]["CUDA_VISIBLE_DEVICES"] == ""
            and launch["selected_environment"]["LD_LIBRARY_PATH"] == ""
            and launch["automatic_retry"] is False
            and exited["automatic_retry"] is False,
            "Same CPU prerequisite actually waited/reaped0 without signal or automatic retry",
        )
        result = read(owner["result_ref"])
        require(result["complete"] is result["passed"] is True, "Closed prerequisite result")
        for path, digest in read(owner["registration_ref"])["source_sha256"].items():
            current = reference(path)
            if current["sha256"] != digest:
                migration = historical_fixture_migrations.get(path)
                require(
                    migration is not None
                    and migration["before_sha256"] == digest
                    and migration["after_ref"] == current,
                    "Historical source drift must match one exact previously audited fixture migration",
                )
                historical_source_migrations_used.append(dict(owner_ref=ref, **migration))
        return owner, result

    trajectory_check_owner_ref = reference(
        REPO / "outputs/tracking_v5_state_updated_trajectory_control_v2_check_execution_20261028/result.json"
    )
    trajectory_check_owner, trajectory_check = closed(trajectory_check_owner_ref)
    fixture_migrations = read(trajectory_check_owner["registration_ref"])["explicit_test_source_migrations"]
    expected_fixtures = {
        str(REPO / "tests" / name)
        for name in ("test_native_training_continuation_preparation.py", "test_native_training_runner.py")
    }
    require(
        len(fixture_migrations) == 2 and {item["path"] for item in fixture_migrations} == expected_fixtures,
        "Only the two exact previously audited fixture repairs are permitted",
    )
    for migration in fixture_migrations:
        require(
            migration["before_commit"] == "9db6efa"
            and migration["after_commit"] == "2f09739"
            and migration["path"] == migration["after_ref"]["path"],
            "Explicit preserved fixture migration identities",
        )
        checked(migration["after_ref"])
        historical_fixture_migrations[migration["path"]] = migration
    trajectory_check_ref = trajectory_check_owner["result_ref"]
    trajectory_check_reg = read(trajectory_check["registration_ref"])
    trajectory_ref = reference(REPO / "outputs/tracking_v5_state_updated_trajectory_control_20261028/verified.json")
    require(
        trajectory_check_reg["producer_ref"] == trajectory_ref
        and trajectory_check["trace_count"] == 4000
        and trajectory_check["numerical_inner_steps"] == 1200000
        and trajectory_check["mj_forward_queries"] == 1248001
        and trajectory_check["witness_found"] is False
        and trajectory_check["control_qualified"] is False,
        "Full independently closed original H12 failure before changing repair mechanism",
    )
    for ref in trajectory_check["source_refs_unchanged"]:
        checked(ref)
    trajectory = read(trajectory_ref)
    trajectory_reg = read(trajectory["registration_ref"])

    stop_owner_ref = reference(
        REPO / "outputs/tracking_v5_state_updated_stop_control_check_execution_20261028/result.json"
    )
    _, stop_check = closed(stop_owner_ref)
    stop_check_reg = read(stop_check["registration_ref"])
    stop_ref = stop_check_reg["producer_ref"]
    stop = read(stop_ref)
    require(stop["answer"]["witness_found"] is False, "Registered transition after finite stopped-state failure")
    stop_reg = read(stop["registration_ref"])
    early_ref = reference(REPO / "outputs/tracking_v5_fk_first_long_numeric_20261028/verified.json")
    early = read(early_ref)
    early_reg = read(early["registration_ref"])
    early_owner_ref = reference(REPO / "outputs/tracking_v5_fk_first_long_check_execution_20261028/result.json")
    _, early_check = closed(early_owner_ref)
    require(read(early_check["registration_ref"])["producer_ref"] == early_ref, "Exact audited early numerical source")
    original = read(early_reg["producer_ref"])
    original_reg = read(original["registration_ref"])
    physical_owner_ref = early_reg["independent_owner_ref"]
    _, physical_check = closed(physical_owner_ref)
    require(
        read(physical_check["registration_ref"])["producer_ref"] == early_reg["producer_ref"],
        "Exact independent original physical branch",
    )
    require(
        trajectory_reg["early_numeric_ref"] == early_ref
        and trajectory_reg["early_independent_owner_ref"] == early_owner_ref
        and trajectory_reg["original_producer_ref"] == early_reg["producer_ref"]
        and trajectory_reg["physical_independent_owner_ref"] == physical_owner_ref,
        "Same original warm and physical branch as the independently checked H12 attempt",
    )
    entry = next(e for e in early["entries"] if e["step"] == 368 and e["horizon"] == 12)
    warm = np.asarray(entry["answer"]["planned_commands"])
    normals = np.asarray(entry["answer"]["terminal_normals"])
    np.testing.assert_array_equal(normals, np.eye(7)[6:])
    require(warm.shape == (12, 6), "One registered twelve-command warm trajectory")
    profile = np.zeros((25, 7))
    np.testing.assert_array_equal(profile, entry["causal_history"])
    with np.load(checked(early_reg["states_ref"]), allow_pickle=False) as archive:
        state = archive["sim_states"][368].copy()
    # Only the completed prefix is consumed; no post368 row is a policy input.
    with checked(original_reg["rows_ref"]).open() as stream:
        for index, line in enumerate(stream):
            if index == 367:
                previous_row = json.loads(line)
                break
        else:
            raise ValueError("Missing completed actual367 controller history")
    previous = np.asarray(previous_row["command"])
    np.testing.assert_array_equal(previous, entry["previous_command"])
    latch = np.asarray(previous_row["controller_goal_after"])
    require(latch.shape == (3, 3), "Actual367 controller direction goal")
    contract_record = read(early_reg["control_ref"])
    control = contract_record["contract"]
    interface = OSCInterface(*(control[k] for k in ("input_min", "input_max", "output_min", "output_max")))
    previous_site = np.asarray(previous_row["qualified_forward_input"]["x_site_pose"])[0]
    previous_delta = interface.decode(previous[:6])
    require(not np.all(previous_delta[3:] == 0), "Completed367 nonzero reset identifies latch")
    np.testing.assert_allclose(
        native_goal_rotation(previous_delta[3:]) @ previous_site[:3, :3], latch, atol=1e-10, rtol=0
    )
    chunk = read(early_reg["chunks_ref"])[46]
    require(chunk["step"] == 368, "Previously generated chunk46 available at start")
    desired = np.asarray(chunk["effective_targets_tcp"])[0:12].copy()
    initial_ref = original_reg["braking_numeric_ref"]
    initial = np.asarray(read(initial_ref)["initial_joint"])
    site_from_tcp = inverse(np.asarray(contract_record["source_from_site"])) @ np.asarray(
        read(early_reg["tool_ref"])["source_from_tcp"]
    )
    source_refs = [
        reference(Path(__file__)),
        *[
            reference(REPO / p)
            for p in (
                "src/s2a/native_goal_trajectory.py",
                "src/s2a/prefix_preserving_trajectory_plan.py",
                "src/s2a/native_goal_joint_response.py",
                "src/s2a/nonlinear_joint_response.py",
                "src/s2a/mujoco_robot_dynamics.py",
                "src/s2a/joint_preview_constraint_audit.py",
                "plans/v0.0.1/prefix_preserving_trajectory_control_20261028.md",
            )
        ],
    ]
    registration = publish(
        ROOT / "registration.json",
        dict(
            prior_trajectory_check_owner_ref=trajectory_check_owner_ref,
            prior_trajectory_check_ref=trajectory_check_ref,
            prior_trajectory_ref=trajectory_ref,
            explicit_historical_source_migrations=historical_source_migrations_used,
            prior_stop_check_owner_ref=stop_owner_ref,
            prior_stop_ref=stop_ref,
            early_numeric_ref=early_ref,
            early_independent_owner_ref=early_owner_ref,
            physical_independent_owner_ref=physical_owner_ref,
            original_producer_ref=early_reg["producer_ref"],
            manifest_ref=original_reg["manifest_ref"],
            states_ref=early_reg["states_ref"],
            original_rows_ref=original_reg["rows_ref"],
            initial_ref=initial_ref,
            **{key: early_reg[key] for key in ("xml_ref", "control_ref", "tool_ref", "chunks_ref")},
            native_source_refs=stop_reg["native_source_refs"],
            native_transform_ref=stop_reg["native_transform_ref"],
            source_refs=source_refs,
            seed=42,
            endpoint=368,
            cursor=0,
            horizon=12,
            target_cursors=list(range(12)),
            initial_history_completed_periods=0,
            history_policy="Zero profile at368; no later actual state, command or residual may condition this search",
            latch_policy="Actual completed367 controller_goal_after, crosschecked with that period SITE and dispatched command",
            response_policy="Carry predicted q/v/native goals through12 periods, each25x2ms; updated M/J and frozen current fingers",
            warm_policy="Exact prior same-state H12 planned_commands; no alternative start",
            search=dict(
                total_iterations=60,
                mechanism="Preserve original pose caps of continuous qualified prefix; repair first failing cursor",
                expected_initial_prefix=6,
                objective="Current-cursor squared-cap epigraph plus1e-4 sum of all H12 normalized squared errors",
                prefix_policy="Original caps only, all72 commands remain free; no locked commands or old residual values",
                phase_policy="Advance at callback or normal optimizer-return boundary only after full audit and strictly longer continuous prefix",
                selection_policy="Longest continuous prefix, next failure max normalized square, full squared sum, first evaluation",
                witness_selection_policy="Global max normalized square, full squared sum, first evaluation",
                safety_policy="No restoration phase; repair requires full-safe warm with minimum nonlinear slack>=1e-8",
                no_progress_policy="Stop after a phase that cannot extend the prefix; no restart or alternative warm",
                response_budget=4000,
                difference_step=1e-4,
                difference_policy="Box-contained single-axis probes; rate enforced as explicit linear optimization constraint",
                optimization_safety_buffer=1e-8,
                ftol=1e-12,
                starts=1,
            ),
            constraints=dict(
                rate=0.25,
                joint_margin=0.05,
                braking_time=0.1,
                position_cap=0.01,
                rotation_cap=0.05,
                pose_caps_horizon=12,
                terminal_normals=normals,
            ),
            physics_steps=0,
            world_model_calls=0,
            fresh_cases=0,
            automatic_retry=False,
        ),
    )
    result = dict(
        complete=False,
        passed=False,
        registration_ref=registration,
        physics_steps=0,
        world_model_calls=0,
        fresh_cases=0,
        control_qualified=False,
    )
    try:
        guard = read(CONTROL_REF)["cv2_guard_registration"]
        result["environment_admission"] = restore_cv2_environment(
            guard["exact_probe_ref"], guard["runtime_source_refs"]
        )
        os.environ.pop("MUJOCO_GL", None)
        os.environ.pop("PYOPENGL_PLATFORM", None)
        import mujoco

        xml = checked(early_reg["xml_ref"])
        names = [f"robot0_joint{k}" for k in range(1, 8)]
        probe = MujocoRobotDynamics(xml, control["controller_site"], names, {})
        model, qids, dofs = probe.model, probe.qids, probe.dofs
        require(model.opt.timestep == 0.002, "Original2ms XML")
        fingers = {}
        for k in (1, 2):
            name = f"gripper0_finger_joint{k}"
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            require(jid >= 0, "Named current finger state")
            fingers[name] = [state[1 + int(model.jnt_qposadr[jid])], state[1 + model.nq + int(model.jnt_dofadr[jid])]]
        q, velocity = state[1 + qids].copy(), state[1 + model.nq + dofs].copy()
        np.testing.assert_array_equal(q, entry["current_q"])
        np.testing.assert_array_equal(velocity, entry["current_velocity"])
        query = MujocoRobotDynamics(xml, control["controller_site"], names, fingers)
        current = query(q, velocity)
        np.testing.assert_allclose(current["pose"], entry["current_site"], atol=1e-10, rtol=0)
        limits = model.jnt_range[probe.joints].copy()
        torque = np.asarray(stop["torque_limits"])
        kp, kd = np.asarray(control["kp"]), np.asarray(control["kd"])
        result.update(
            current_q=q,
            current_velocity=velocity,
            fixed_gripper_state=fingers,
            current_quantities=current,
            initial_joint=initial,
            kp=kp,
            kd=kd,
            torque_limits=torque,
            joint_limits=limits,
            site_from_tcp=site_from_tcp,
            desired_tcp=desired,
            previous_command=previous,
            previous_site=previous_site,
            previous_physical_delta=previous_delta,
            latched_rotation_goal=latch,
            warm=warm,
            past_inner_error_rad=profile,
            terminal_normals=normals,
            history_before_search=dict(
                completed_periods=0, last_completed_step=None, inner_position_error_rad=profile, future_guarantee=False
            ),
        )

        evaluations = 0

        def predict(commands):
            nonlocal evaluations
            response = rollout_native_goal_trajectory(
                q,
                velocity,
                commands,
                interface,
                initial,
                kp,
                kd,
                query,
                torque[:, 0],
                torque[:, 1],
                latched_rotation_goal=latch,
            )
            evaluations += 1
            if evaluations == 1 or evaluations % 100 == 0:
                print(
                    json.dumps(dict(responses_completed=evaluations, elapsed_s=time.monotonic() - started)), flush=True
                )
            return response

        started = time.monotonic()
        result["answer"] = solve_prefix_preserving_trajectory_response(
            predict,
            previous[:6],
            warm,
            interface,
            limits[:, 0],
            limits[:, 1],
            desired,
            site_from_tcp,
            past_inner_error_rad=profile,
            terminal_normals=normals,
            max_evaluations=4000,
            max_iterations=60,
            finite_difference_step=1e-4,
        )
        result["search_seconds"] = time.monotonic() - started
        answer = result["answer"]
        require(evaluations == answer["response_evaluations"], "Every prediction charged to the unique response budget")
        require(answer["initial_prefix_length"] == 6, "Original warm keeps exactly its first six consecutive points")
        np.testing.assert_array_equal(answer["initial"]["commands"], trajectory["answer"]["initial"]["commands"])
        np.testing.assert_allclose(
            answer["initial"]["tcp_errors"], trajectory["answer"]["initial"]["tcp_errors"], atol=1e-10, rtol=0
        )
        require(answer["selected_prefix_length"] >= 6, "Repair must retain every original qualified prefix point")
        result["numerical_inner_steps"] = answer["response_evaluations"] * 12 * 25
        require(answer["response_evaluations"] <= 4000 and answer["total_iterations"] <= 60, "Fixed shared budget")
        for ref in [*source_refs, trajectory_check_owner_ref, trajectory_check_ref, trajectory_ref]:
            checked(ref)
        for ref in trajectory_check["source_refs_unchanged"]:
            checked(ref)
        result.update(complete=True, passed=True)
    except BaseException as error:
        result["error"] = repr(error)
        raise
    finally:
        publish(ROOT / "verified.json", result)
    print(json.dumps(dict(complete=True, passed=True, witness_found=answer["witness_found"])), flush=True)


if __name__ == "__main__":
    main()
