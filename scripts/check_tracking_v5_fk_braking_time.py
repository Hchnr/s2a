"""Independently check same-state projected braking-time sensitivity; no search or physics."""

import json
import os
from pathlib import Path

import numpy as np

from s2a.artifacts import REPO
from s2a.clean_training_runtime import checked, read, reference, require
from s2a.geometry import inverse, pose_error, transform
from s2a.interior_joint_preview import build_joint_preview, rollout_joint_preview
from s2a.native_environment import restore_cv2_environment, strict_native_environment
from s2a.osc import OSCInterface
from s2a.preview_result_publication import publish_result as publish
from s2a.robot_site_kinematics import RobotSiteKinematics
from s2a.terminal_development_execution import CONTROL_REF


def closed_owner(ref):
    owner = read(ref)
    require(owner["complete"] is owner["passed"] is True, "Completed dependency owner")
    exit_record, release = read(owner["owned_command"]["exit"]), read(owner["owned_command"]["release"])
    require(
        exit_record["actual_wait"] is True
        and exit_record["returncode"] == 0
        and release["reaped"] is True
        and release["returncode"] == 0,
        "Actual dependency wait0/reap0",
    )
    result = read(owner["result_ref"])
    require(result["complete"] is result["passed"] is True, "Completed dependency result")
    return owner, result


def main():
    strict_native_environment()
    root = REPO / "outputs/tracking_v5_fk_braking_time_check_20261028"
    require(not root.exists(), "No overwrite or automatic retry")
    owner_ref = reference(REPO / "outputs/tracking_v5_fk_braking_time_numeric_execution_20261028/result.json")
    owner, producer = closed_owner(owner_ref)
    producer_ref = owner["result_ref"]
    reg = read(producer["registration_ref"])
    independent_owner, independent = closed_owner(reg["independent_owner_ref"])
    require(independent_owner["result_ref"] == reg["independent_audit_ref"], "Exact independent original audit")
    independent_reg = read(independent["registration_ref"])
    require(independent_reg["producer_ref"] == reg["original_audit_producer_ref"], "Exact certified original audit")
    original_audit = read(reg["original_audit_producer_ref"])
    require(
        read(original_audit["registration_ref"])["prior_native_ref"] == reg["prior_native_ref"], "Same native branch"
    )
    baseline = next(v for v in original_audit["entries"] if v["mode"] == "no_terminal")
    require(producer["baseline_no_terminal"] == baseline, "Unchanged complete .1s baseline entry")
    require(
        baseline["answer"]["braking_time_s"] == 0.1 and baseline["answer"]["witness_found"] is False,
        "Archived .1s no-terminal baseline remains uncertified",
    )
    require(
        reg["braking_times_s"] == [0.05, 0.025] and reg["baseline_braking_time_s"] == 0.1,
        "Exactly registered braking lookahead comparison",
    )
    native = read(reg["prior_native_ref"])
    require(native["complete"] is native["passed"] is True, "Original native producer complete")
    require(
        native["physics_steps"] == native["stopped"]["step"] == 377
        and native["last_consumed_cursor"] == 8
        and native["stopped"]["reason"] == "no_model_witness"
        and native["stopped"]["action_dispatched"] is False,
        "Original actual stop377 before cursor9 dispatch",
    )
    with np.load(checked(native["states_ref"]), allow_pickle=False) as archive:
        states = archive["sim_states"]
        require(len(states) == 378, "Every prefix endpoint including stopped state")
        state = states[-1].copy()
    rows = read(native["periods_ref"])
    previous = np.asarray(rows[-1]["command"])
    profile = np.asarray(rows[-1]["history_after"])
    np.testing.assert_array_equal(previous, producer["previous_command"])
    np.testing.assert_array_equal(previous, np.asarray(read(native["commands_ref"]))[-1])
    np.testing.assert_array_equal(profile, producer["history_before"])
    np.testing.assert_array_equal(profile, native["stopped"]["history_before"])
    original_answer = native["stopped"]["answer"]
    np.testing.assert_array_equal(profile, original_answer["past_inner_error_rad"])
    original_normals = np.asarray(original_answer["terminal_normals"]).reshape(-1, 7)
    np.testing.assert_array_equal(original_normals, np.eye(7)[[6]])
    require(original_answer["braking_time_s"] == 0.1, "Original H8 braking policy unchanged")
    np.testing.assert_array_equal(
        producer["supplied_warm"], np.asarray(original_answer["closest_linear_feasible_attempt"]["commands"])[:1]
    )
    for name in ("kp", "kd"):
        np.testing.assert_array_equal(producer[name], rows[-1]["current_dynamics"][name])
    numeric = read(reg["numeric_ref"])
    numeric_reg = read(numeric["registration_ref"])
    original = read(numeric_reg["producer_ref"])
    original_reg = read(original["registration_ref"])
    initial = np.asarray(read(original_reg["braking_numeric_ref"])["initial_joint"])
    np.testing.assert_array_equal(initial, producer["initial_joint"])
    for key in ("xml_ref", "control_ref", "tool_ref", "chunks_ref"):
        require(reg[key] == numeric_reg[key], "Exact original robot and generated reference sources")
    control = read(reg["control_ref"])
    contract = control["contract"]
    interface = OSCInterface(*(contract[k] for k in ("input_min", "input_max", "output_min", "output_max")))
    site_from_tcp = inverse(np.asarray(control["source_from_site"])) @ np.asarray(
        read(reg["tool_ref"])["source_from_tcp"]
    )
    chunk = read(reg["chunks_ref"])[46]
    require(chunk["step"] == 368, "Reference generated before current377 endpoint")
    targets = np.asarray(chunk["effective_targets_tcp"])
    robot_fk = RobotSiteKinematics(
        checked(reg["xml_ref"]).read_text(), contract["controller_site"], [f"robot0_joint{k}" for k in range(1, 8)]
    )
    registration = publish(
        root / "registration.json",
        dict(
            producer_owner_ref=owner_ref,
            producer_ref=producer_ref,
            prior_native_ref=reg["prior_native_ref"],
            independent_owner_ref=reg["independent_owner_ref"],
            xml_ref=reg["xml_ref"],
            source_ref=reference(Path(__file__)),
            original_audit_producer_ref=reg["original_audit_producer_ref"],
            independent_audit_ref=reg["independent_audit_ref"],
            correction="Only free/ball joint quaternion coordinates may normalize during mj_forward; all other qpos and all qvel remain bitwise exact",
            quaternion_normalization_tolerance=1e-10,
            virtual_query_policy="Reset every query from the same postinitial-forward qpos/qvel, replace only seven predicted robot joints, then independently verify quaternion normalization",
            endpoint=377,
            cursor=9,
            seed=42,
            original_horizon=8,
            diagnostic_horizon=1,
            original_braking_time_s=0.1,
            diagnostic_braking_times_s=[0.05, 0.025],
            modes=["braking_050", "braking_025"],
            box_rate_tolerance=1e-12,
            other_numeric_tolerance=1e-10,
            physics_steps=0,
            world_model_calls=0,
            fresh_cases=0,
            automatic_retry=False,
            scope="Independent saved candidate geometry and constraint audit; no optimizer or constraint-audit helper imported, no action or qualification",
        ),
    )
    guard = read(CONTROL_REF)["cv2_guard_registration"]
    restore_cv2_environment(guard["exact_probe_ref"], guard["runtime_source_refs"])
    os.environ.pop("MUJOCO_GL", None)
    os.environ.pop("PYOPENGL_PLATFORM", None)
    import mujoco

    result = dict(
        complete=False,
        passed=False,
        registration_ref=registration,
        physics_steps=0,
        world_model_calls=0,
        fresh_cases=0,
        entries=[],
    )
    try:
        model = mujoco.MjModel.from_xml_path(str(checked(reg["xml_ref"])))
        data = mujoco.MjData(model)
        joints = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"robot0_joint{k}") for k in range(1, 8)]
        require(all(v >= 0 for v in joints), "Native robot joints")
        qids, dofs = model.jnt_qposadr[joints], model.jnt_dofadr[joints]
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, contract["controller_site"])
        require(sid >= 0, "Native controller SITE")
        data.time = state[0]
        data.qpos[:] = state[1 : 1 + model.nq]
        data.qvel[:] = state[1 + model.nq :]
        initial_qpos, current_qvel = data.qpos.copy(), data.qvel.copy()
        quaternion_blocks = []
        for joint in range(model.njnt):
            kind, address = int(model.jnt_type[joint]), int(model.jnt_qposadr[joint])
            if kind == int(mujoco.mjtJoint.mjJNT_FREE):
                quaternion_blocks.append(np.arange(address + 3, address + 7))
            elif kind == int(mujoco.mjtJoint.mjJNT_BALL):
                quaternion_blocks.append(np.arange(address, address + 4))
        quaternion_qids = np.concatenate(quaternion_blocks) if quaternion_blocks else np.empty(0, dtype=int)
        nonquaternion_qids = np.setdiff1d(np.arange(model.nq), quaternion_qids)
        require(not np.intersect1d(qids, quaternion_qids).size, "All seven controlled robot joints are scalar")
        other_scalar_qids = np.setdiff1d(nonquaternion_qids, qids)
        other_dofs = np.setdiff1d(np.arange(model.nv), dofs)

        def check_quaternion_normalization(before, after):
            expected = before.copy()
            for indices in quaternion_blocks:
                norm = np.linalg.norm(before[indices])
                require(np.isfinite(norm) and norm > 0, "Finite nonzero free/ball joint quaternion")
                expected[indices] /= norm
            np.testing.assert_array_equal(after[nonquaternion_qids], before[nonquaternion_qids])
            np.testing.assert_allclose(after[quaternion_qids], expected[quaternion_qids], atol=1e-10, rtol=0)
            for indices in quaternion_blocks:
                np.testing.assert_allclose(np.linalg.norm(after[indices]), 1.0, atol=1e-10, rtol=0)
            return dict(
                before_qpos=before.copy(),
                expected_normalized_qpos=expected,
                after_qpos=after.copy(),
                difference=after - before,
                maximum_normalization_residual=float(np.max(abs(after - expected))),
                changed_coordinate_indices=np.flatnonzero(after != before),
                quaternion_coordinate_indices=quaternion_qids,
                quaternion_blocks=quaternion_blocks,
            )

        mujoco.mj_forward(model, data)
        result["initial_quaternion_normalization"] = check_quaternion_normalization(initial_qpos, data.qpos)
        current_qpos = data.qpos.copy()
        np.testing.assert_array_equal(data.qvel, current_qvel)
        np.testing.assert_array_equal(data.qpos[qids], producer["current_q"])
        np.testing.assert_array_equal(data.qvel[dofs], producer["current_velocity"])
        np.testing.assert_array_equal(data.qpos[qids], rows[-1]["actual_inner"][-1]["q"])
        np.testing.assert_array_equal(data.qvel[dofs], rows[-1]["actual_inner"][-1]["qvel"])
        mass = np.zeros((model.nv, model.nv))
        mujoco.mj_fullM(model, mass, data.qM)
        mass = mass[np.ix_(dofs, dofs)]
        jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jp, jr, sid)
        jacobian = np.vstack((jp, jr))[:, dofs]
        site = transform(data.site_xpos[sid].copy(), data.site_xmat[sid].reshape(3, 3).copy())
        np.testing.assert_allclose(mass, producer["current_mass"], atol=1e-10, rtol=0)
        np.testing.assert_allclose(jacobian, producer["current_jacobian"], atol=1e-10, rtol=0)
        np.testing.assert_allclose(site, producer["current_site"], atol=1e-10, rtol=0)
        lower, upper = model.jnt_range[joints].T
        np.testing.assert_array_equal(lower, producer["lower"])
        np.testing.assert_array_equal(upper, producer["upper"])
        local = build_joint_preview(
            mass, jacobian, producer["kp"], producer["kd"], producer["current_q"], initial, null_kp=10.0, dt=0.002
        )

        def check_candidate(candidate, answer, normals, horizon):
            commands = np.asarray(candidate["commands"])
            require(commands.shape == (horizon, 6) and np.isfinite(commands).all(), "Exact finite command horizon")
            selected = answer if answer["witness_found"] else answer["closest_linear_feasible_attempt"]
            require(selected is not None, "Selected saved witness or closest linear-feasible attempt exists")
            np.testing.assert_array_equal(
                commands, selected["planned_commands"] if answer["witness_found"] else selected["commands"]
            )
            raw = rollout_joint_preview(
                local,
                producer["current_velocity"],
                np.asarray([interface.decode(v) for v in commands]),
                site,
                lower,
                upper,
                enforce_limits=False,
            )
            for name in ("joint_position", "joint_velocity", "inner_joint_position"):
                np.testing.assert_allclose(raw[name], candidate["rollout"][name], atol=1e-10, rtol=0)
                if answer["witness_found"]:
                    np.testing.assert_allclose(raw[name], answer["rollout"][name], atol=1e-10, rtol=0)
            checks = []
            errors = []
            for period in range(horizon):
                q, velocity = raw["joint_position"][period], raw["joint_velocity"][period]
                data.qpos[:] = current_qpos
                data.qvel[:] = current_qvel
                data.qpos[qids] = q
                data.qvel[dofs] = velocity
                virtual_qpos, virtual_qvel = data.qpos.copy(), data.qvel.copy()
                mujoco.mj_forward(model, data)
                normalization = check_quaternion_normalization(virtual_qpos, data.qpos)
                np.testing.assert_array_equal(data.qvel, virtual_qvel)
                native_pose = transform(data.site_xpos[sid].copy(), data.site_xmat[sid].reshape(3, 3).copy())
                jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
                mujoco.mj_jacSite(model, data, jp, jr, sid)
                native_jac = np.vstack((jp, jr))[:, dofs]
                xml_pose, xml_jac = robot_fk(q)
                np.testing.assert_allclose(native_pose, xml_pose, atol=1e-10, rtol=0)
                np.testing.assert_allclose(native_jac, xml_jac, atol=1e-10, rtol=0)
                np.testing.assert_allclose(native_pose, candidate["rollout"]["pose"][period], atol=1e-10, rtol=0)
                np.testing.assert_allclose(
                    native_jac @ velocity, candidate["rollout"]["twist"][period], atol=1e-10, rtol=0
                )
                error = pose_error(targets[9 + period], native_pose @ site_from_tcp)
                errors.append(error)
                np.testing.assert_allclose(error, candidate["all_tcp_errors"][period], atol=1e-10, rtol=0)
                np.testing.assert_allclose(error, selected["all_tcp_errors"][period], atol=1e-10, rtol=0)
                checks.append(
                    dict(
                        period=period,
                        target_cursor=9 + period,
                        native_site=native_pose,
                        native_jacobian=native_jac,
                        native_tcp_error=error,
                        pose_difference=float(np.max(abs(native_pose - xml_pose))),
                        jacobian_difference=float(np.max(abs(native_jac - xml_jac))),
                        quaternion_normalization=normalization,
                    )
                )
                np.testing.assert_array_equal(data.qpos[other_scalar_qids], current_qpos[other_scalar_qids])
                np.testing.assert_array_equal(data.qvel[other_dofs], current_qvel[other_dofs])
                require(float(data.time) == state[0], "No physical time advance")
            errors = np.asarray(errors)
            first_caps = bool(
                np.linalg.norm(errors[0, :3]) <= 0.01 + 1e-10 and np.linalg.norm(errors[0, 3:]) <= 0.05 + 1e-10
            )
            require(not answer["witness_found"] or first_caps, "Witness retains original first pose caps")
            np.testing.assert_allclose(errors[0], selected["first_tcp_error"], atol=1e-10, rtol=0)
            margin = np.tile(0.05 + profile.max(axis=0), (25 * horizon, 1))
            margin[:25] = 0.05 + profile
            np.testing.assert_allclose(margin, answer["requested_position_margin_rad"], atol=1e-10, rtol=0)
            delta = np.diff(np.vstack((previous[:6], commands)), axis=0)
            inner = raw["inner_joint_position"]
            braking = raw["joint_position"] + answer["braking_time_s"] * raw["joint_velocity"]
            groups = dict(
                box_lower=commands - interface.input_min,
                box_upper=interface.input_max - commands,
                rate_lower=0.25 + delta,
                rate_upper=0.25 - delta,
                inner_lower=inner - lower - margin,
                inner_upper=upper - margin - inner,
                braking_lower=braking - lower - 0.05,
                braking_upper=upper - 0.05 - braking,
                terminal=-normals @ raw["joint_velocity"][-1],
            )
            audit = candidate["audit"]
            require(audit["horizon"] == horizon and audit["joint_count"] == 7, "Exact constraint audit dimensions")
            require(audit["activity_tolerance"] == 1e-8, "Predeclared activity threshold")
            require(set(audit["groups"]) == set(groups), "All nine original constraint groups")
            np.testing.assert_allclose(margin, audit["position_margin_rad"], atol=1e-10, rtol=0)
            first_minima, future_minima, checked_groups = {}, {}, {}
            for name, slack in groups.items():
                tolerance = 1e-12 if name.startswith(("box_", "rate_")) else 1e-10
                require(np.isfinite(slack).all() and np.all(slack >= -tolerance), f"Reconstructed {name} bounds")
                saved = audit["groups"][name]
                require(saved["tolerance"] == tolerance, "Exact group feasibility tolerance")
                require(np.asarray(saved["slack"]).shape == slack.shape, "Exact slack array shape")
                np.testing.assert_allclose(saved["slack"], slack, atol=tolerance, rtol=0)
                minimum = float(slack.min()) if slack.size else None
                if minimum is None:
                    require(saved["minimum"] is None, "Empty terminal minimum is absent")
                else:
                    np.testing.assert_allclose(saved["minimum"], minimum, atol=tolerance, rtol=0)
                active, violated = np.argwhere(slack <= 1e-8), np.argwhere(slack < -tolerance)
                np.testing.assert_array_equal(np.asarray(saved["active_indices"]).reshape(-1, slack.ndim), active)
                np.testing.assert_array_equal(np.asarray(saved["violated_indices"]).reshape(-1, slack.ndim), violated)
                checked_groups[name] = dict(
                    slack=slack,
                    minimum=minimum,
                    active_indices=active,
                    violated_indices=violated,
                    tolerance=tolerance,
                )
                if name != "terminal":
                    split = 25 if name.startswith("inner_") else 1
                    first_minima[name] = float(slack[:split].min())
                    future_minima[name] = float(slack[split:].min()) if slack[split:].size else None
                    np.testing.assert_allclose(
                        audit["first_period_minima"][name], first_minima[name], atol=tolerance, rtol=0
                    )
                    if future_minima[name] is None:
                        require(audit["future_period_minima"][name] is None, "No future minimum at H1")
                    else:
                        np.testing.assert_allclose(
                            audit["future_period_minima"][name], future_minima[name], atol=tolerance, rtol=0
                        )
            require(audit["feasible"] is True, "Saved feasibility agrees with reconstructed bounds")
            require(
                set(audit["first_period_minima"]) == set(audit["future_period_minima"]) == set(groups) - {"terminal"},
                "Complete first and future minima exclude horizon terminal condition",
            )
            return dict(
                horizon=horizon,
                model_witness=answer["witness_found"],
                native_first_caps_satisfied=first_caps,
                rows=checks,
                groups=checked_groups,
                first_period_minima=first_minima,
                future_period_minima=future_minima,
                maximum_pose_difference=max(v["pose_difference"] for v in checks),
                maximum_jacobian_difference=max(v["jacobian_difference"] for v in checks),
            )

        result["original_candidate"] = check_candidate(
            producer["original_candidate"], original_answer, original_normals, 8
        )
        require(
            [v["mode"] for v in producer["entries"]] == ["braking_050", "braking_025"],
            "Exactly two registered diagnostic modes",
        )
        for entry in producer["entries"]:
            mode, answer, candidate = entry["mode"], entry["answer"], entry["candidate"]
            require(
                np.isfinite(entry["compute_seconds"]) and entry["compute_seconds"] >= 0, "Actual finite search duration"
            )
            require(
                answer["pose_caps_horizon"] == 1 and answer["all_pose_caps_enforced"] is True,
                "H1 first pose is the full supplied horizon",
            )
            require(
                answer["response_budget"] == 160 and answer["iteration_budget"] == 60, "Original finite search budgets"
            )
            require(
                answer["response_evaluations"] <= 160 and answer.get("iterations", 0) <= 60,
                "Actual finite search counts",
            )
            np.testing.assert_array_equal(answer["seed_commands"], previous[None, :6])
            np.testing.assert_array_equal(answer["past_inner_error_rad"], profile)
            normals = np.asarray(answer["terminal_normals"]).reshape(-1, 7)
            expected_normals = np.empty((0, 7))
            expected_braking = 0.05 if mode == "braking_050" else 0.025
            require(
                entry["braking_time_s"] == answer["braking_time_s"] == expected_braking,
                "Each candidate checked under its own declared braking projection",
            )
            np.testing.assert_array_equal(normals, expected_normals)
            if candidate is None:
                require(
                    answer["witness_found"] is False and answer.get("closest_linear_feasible_attempt") is None,
                    "Missing candidate only when no saved feasible attempt exists",
                )
                result["entries"].append(dict(mode=mode, model_witness=False, candidate_available=False))
            else:
                result["entries"].append(
                    dict(mode=mode, candidate_available=True, **check_candidate(candidate, answer, normals, 1))
                )
        result.update(
            complete=True,
            passed=True,
            scope="Independent same-state geometry and bounds at each declared braking time; no action, full future safety, physical qualification or global infeasibility claim",
        )
    except BaseException as error:
        result["error"] = repr(error)
        raise
    finally:
        publish(root / "verified.json", result)
    print(json.dumps(dict(complete=True, passed=True)), flush=True)


if __name__ == "__main__":
    main()
