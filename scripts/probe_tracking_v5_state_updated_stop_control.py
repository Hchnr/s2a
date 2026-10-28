"""Rebuild causal native-goal response history and search one stopped-state action."""

import os
import time
from pathlib import Path

import numpy as np

from s2a.artifacts import REPO
from s2a.causal_inner_error import CausalInnerError
from s2a.clean_training_runtime import checked, read, reference, require
from s2a.mujoco_robot_dynamics import MujocoRobotDynamics
from s2a.native_environment import restore_cv2_environment, strict_native_environment
from s2a.native_goal_joint_response import rollout_native_goal_joint_response
from s2a.nonlinear_single_period_plan import solve_single_period_response
from s2a.osc import OSCInterface
from s2a.preview_result_publication import publish_result as publish
from s2a.terminal_development_execution import CONTROL_REF

ROOT = REPO / "outputs/tracking_v5_state_updated_stop_control_20261028"


def main():
    strict_native_environment()
    require(not ROOT.exists(), "No overwrite or automatic retry")
    owner_ref = reference(REPO / "outputs/tracking_v5_state_updated_response_check_execution_20261028/result.json")
    owner = read(owner_ref)
    require(owner["complete"] is owner["passed"] is True, "Previous response audit closed")
    exited, released = read(owner["owned_command"]["exit"]), read(owner["owned_command"]["release"])
    require(
        exited["actual_wait"] is True
        and exited["returncode"] == 0
        and released["reaped"] is True
        and released["returncode"] == 0,
        "Actual wait0/reap0",
    )
    previous_audit = read(owner["result_ref"])
    require(previous_audit["complete"] is previous_audit["passed"] is True, "Independent response result passed")
    prior_audit_registration = read(previous_audit["registration_ref"])
    previous_ref = prior_audit_registration["producer_ref"]
    previous_result = read(previous_ref)
    preg = read(previous_result["registration_ref"])
    native = read(preg["native_ref"])
    nreg = read(native["registration_ref"])
    require(
        native["stopped"]["step"] == 377 and native["stopped"]["action_dispatched"] is False,
        "Exact actual377 endpoint, no original cursor9 dispatch",
    )
    original_stop_ref = reference(REPO / "outputs/tracking_v5_fk_stop_constraint_audit_20261028/verified.json")
    original_stop = read(original_stop_ref)
    require(original_stop["complete"] is original_stop["passed"] is True, "Original stopped-state audit")
    rows = read(preg["periods_ref"])
    require([r["step"] for r in rows] == list(range(368, 377)), "Exactly nine completed past periods")
    with np.load(checked(preg["states_ref"]), allow_pickle=False) as archive:
        states = archive["sim_states"].copy()
    require(len(states) == 378, "All actual endpoints through377")
    control = read(preg["control_ref"])["contract"]
    interface = OSCInterface(*(control[k] for k in ("input_min", "input_max", "output_min", "output_max")))
    chunk = read(nreg["chunks_ref"])[46]
    require(chunk["step"] == 368, "Already generated reference chunk46")
    desired = np.asarray(chunk["effective_targets_tcp"])[9]
    warm = np.asarray(native["stopped"]["answer"]["closest_linear_feasible_attempt"]["commands"])[0]
    previous_command = np.asarray(rows[-1]["command"])
    np.testing.assert_array_equal(previous_command, original_stop["previous_command"])
    np.testing.assert_array_equal(warm, np.asarray(original_stop["original_candidate"]["commands"])[0])
    source_refs = [
        reference(Path(__file__)),
        *[
            reference(REPO / p)
            for p in (
                "src/s2a/native_goal_joint_response.py",
                "src/s2a/nonlinear_single_period_plan.py",
                "src/s2a/nonlinear_joint_response.py",
                "src/s2a/mujoco_robot_dynamics.py",
                "src/s2a/causal_inner_error.py",
                "src/s2a/joint_preview_constraint_audit.py",
                "plans/v0.0.1/state_updated_stop_control_20261028.md",
            )
        ],
    ]
    registration = publish(
        ROOT / "registration.json",
        dict(
            independent_response_owner_ref=owner_ref,
            prior_response_ref=previous_ref,
            prior_response_audit_ref=owner["result_ref"],
            native_ref=preg["native_ref"],
            states_ref=preg["states_ref"],
            periods_ref=preg["periods_ref"],
            xml_ref=preg["xml_ref"],
            control_ref=preg["control_ref"],
            tool_ref=preg["tool_ref"],
            initial_ref=preg["initial_ref"],
            chunks_ref=nreg["chunks_ref"],
            original_stop_ref=original_stop_ref,
            native_source_refs=preg["native_source_refs"],
            native_transform_ref=prior_audit_registration["native_transform_source_ref"],
            source_refs=source_refs,
            seed=42,
            history_steps=list(range(368, 377)),
            endpoint=377,
            cursor=9,
            goal_policy="New native-JIT float32 quaternion decoding; keep old double-Exp evidence unchanged",
            response_policy="State-updated M/J, same robot-only free response, implicit damping Euler25x2ms; current finger q/v frozen",
            history_policy="New native-goal predictor, per-period current state initialization and own previously dispatched commands; predict before reading actual labels; monotone25-sample profile",
            search=dict(
                algorithm="SLSQP epigraph; at most20 safety-restoration iterations then remaining pose iterations",
                total_iterations=60,
                response_budget=600,
                difference_step=1e-4,
                difference_policy="Absolute normalized-command one-sided differences directed into box/rate bounds; all probes count",
                cache="Exact command bytes shared across objective, constraints and Jacobians",
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
                terminal="No terminal normal in H1; same declared no-terminal scope",
            ),
            scope="Numerical command search only, no action dispatch or physical/task qualification",
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
        history=[],
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

        xml = checked(preg["xml_ref"])
        names = [f"robot0_joint{k}" for k in range(1, 8)]
        probe = MujocoRobotDynamics(xml, control["controller_site"], names, {})
        model, qids, dofs = probe.model, probe.qids, probe.dofs
        initial = np.asarray(previous_result["initial_joint"])
        kp, kd = np.asarray(control["kp"]), np.asarray(control["kd"])
        site_from_tcp = np.asarray(previous_result["site_from_tcp"])
        torque = np.asarray(previous_result["torque_limits"])
        limits = model.jnt_range[probe.joints].copy()
        np.testing.assert_array_equal(limits, previous_result["joint_limits"])
        require(model.opt.timestep == 0.002, "Original2ms robot XML")
        fingers = {}
        for k in (1, 2):
            name = f"gripper0_finger_joint{k}"
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            require(jid >= 0, "Two named current finger coordinates")
            fingers[name] = (int(model.jnt_qposadr[jid]), int(model.jnt_dofadr[jid]))

        def condition(step):
            state = states[step]
            fixed = {name: [state[1 + qi], state[1 + model.nq + vi]] for name, (qi, vi) in fingers.items()}
            q, v = state[1 + qids].copy(), state[1 + model.nq + dofs].copy()
            query = MujocoRobotDynamics(xml, control["controller_site"], names, fixed)
            return q, v, fixed, query

        def predict(query, q, velocity, command, latched_rotation_goal=None):
            return rollout_native_goal_joint_response(
                q,
                velocity,
                interface.decode(command[:6]),
                initial,
                kp,
                kd,
                query,
                torque[:, 0],
                torque[:, 1],
                dt=0.002,
                substeps=25,
                update_matrices=True,
                latched_rotation_goal=latched_rotation_goal,
            )

        history = CausalInnerError(7)
        for row in rows:
            step = row["step"]
            q, velocity, fixed, query = condition(step)
            current = query(q, velocity)
            for key in ("mass", "jacobian"):
                np.testing.assert_allclose(current[key], row["current_dynamics"][key], atol=1e-10, rtol=0)
            before = history.snapshot()
            response = predict(query, q, velocity, np.asarray(row["command"]))
            history.stage(step, response["joint_position"])
            actual = np.asarray([v["q"] for v in row["actual_inner"]])
            history.complete(step, actual)
            result["history"].append(
                dict(
                    step=step,
                    q0=q,
                    v0=velocity,
                    fixed_gripper_state=fixed,
                    command=row["command"],
                    response=response,
                    actual_inner_q=actual,
                    history_before=before,
                    history_after=history.snapshot(),
                )
            )
        q, velocity, fixed, query = condition(377)
        current = query(q, velocity)
        for value, key in (
            (q, "current_q"),
            (velocity, "current_velocity"),
            (current["mass"], "current_mass"),
            (current["jacobian"], "current_jacobian"),
            (current["pose"], "current_site"),
        ):
            np.testing.assert_allclose(value, original_stop[key], atol=1e-10, rtol=0)
        profile = history.snapshot()["inner_position_error_rad"]
        require(
            history.snapshot()["completed_periods"] == 9 and history.snapshot()["last_completed_step"] == 376,
            "Only completed own past history",
        )
        latched_rotation_goal = result["history"][-1]["response"]["goal"][:3, :3].copy()
        result.update(
            current_q=q,
            current_velocity=velocity,
            fixed_gripper_state=fixed,
            current_quantities=current,
            initial_joint=initial,
            kp=kp,
            kd=kd,
            torque_limits=torque,
            joint_limits=limits,
            site_from_tcp=site_from_tcp,
            desired_tcp=desired,
            previous_command=previous_command,
            warm=warm,
            history_before_search=history.snapshot(),
            past_inner_error_rad=profile,
            latched_rotation_goal=latched_rotation_goal,
        )
        started = time.monotonic()
        result["answer"] = solve_single_period_response(
            lambda command: predict(query, q, velocity, command, latched_rotation_goal),
            previous_command[:6],
            warm,
            interface,
            limits[:, 0],
            limits[:, 1],
            desired,
            site_from_tcp,
            past_inner_error_rad=profile,
            max_evaluations=600,
            max_iterations=60,
            finite_difference_step=1e-4,
        )
        result["search_seconds"] = time.monotonic() - started
        require(
            result["answer"]["response_evaluations"] <= 600 and result["answer"]["total_iterations"] <= 60,
            "Shared fixed search budgets respected",
        )
        for ref in source_refs:
            checked(ref)
        result.update(complete=True, passed=True)
    except Exception as error:
        result["error"] = repr(error)
        raise
    finally:
        publish(ROOT / "verified.json", result)


if __name__ == "__main__":
    main()
