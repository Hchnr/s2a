"""093 exact old baseline replay with observational inner-controller traces."""

import argparse
from pathlib import Path

import numpy as np
from audit_wam_position import Evidence
from run_matched_oracle_case import CONTROL, CONTROL_SHA

from s2a.native_environment import restore_cv2_environment, strict_native_environment

ROOT = Path("outputs/native_controller_trace_20261025")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", type=int, choices=range(6), required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    strict_native_environment()
    e = Evidence()
    e.read(__file__)
    e.read("src/s2a/controller_trace.py")
    e.check(CONTROL, CONTROL_SHA)
    guard = e.json(CONTROL)["cv2_guard_registration"]
    restore_cv2_environment(guard["exact_probe_ref"], guard["runtime_source_refs"])
    from collect_wam_native_training import endpoint, save_arrays, write_once

    from s2a.controller_trace import ControllerTorqueTrace
    from s2a.libero_simulation import make_environment, measured_twist
    from s2a.settled_native import SettledGoalEnvironment
    from s2a.stability_recovery_simulation import simulator_inputs

    strict_native_environment()
    closed = e.json("outputs/paired_arm_response_20261025/closure_verified.json")
    assert closed["complete"] and closed["data_qualified"] and closed["owner_normal_wait_verified"]
    old_folder = Path("outputs/paired_arm_response_20261025") / f"case{args.group * 7:02d}"
    path = old_folder / "collection_result.json"
    e.check(path, closed["source_sha256"][str(path.resolve())])
    old = e.json(path)
    profile = old["profile"]
    assert profile["branch"] == "baseline" and profile["anchor_group"] == args.group
    prior_ref = profile["case_ref"]
    e.check(prior_ref["path"], prior_ref["sha256"])
    pref = e.json(prior_ref["path"])["prepared_ref"]
    e.check(pref["path"], pref["sha256"])
    prepared = e.json(pref["path"])["original_prepared"]
    record = profile["record"]
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
    ref = old["arrays_ref"]
    e.check(ref["path"], ref["sha256"])
    with np.load(ref["path"], allow_pickle=False) as cache:
        expected = {k: cache[k].copy() for k in ("sim_states", "controller_site_pose", "source_twist", "sent_commands")}
    commands = expected["sent_commands"]
    assert len(commands) == profile["physics_steps"] == profile["anchor_cycle"] + 1
    if args.dry_run:
        e.recheck()
        print(dict(group=args.group, ready=True, physics_steps=0, registered_control_periods=len(commands)))
        return
    folder = ROOT / f"group{args.group:02d}"
    folder.mkdir(parents=True, exist_ok=False)
    env = trace = None
    try:
        actual, _ = make_environment(policy, folder, record, trial=0)
        env = SettledGoalEnvironment(actual, record["native_steps"])
        runtime = simulator_inputs(env)
        for p, signature in runtime.items():
            e.check(p, signature)
        robot, controller = env.robots[0], env.robots[0].controller

        def observe():
            sim = env.sim
            ctrl = np.asarray(sim.data.ctrl)[robot._ref_joint_actuator_indexes].copy()
            np.testing.assert_array_equal(ctrl, robot.torques)
            contacts = []
            for contact in sim.data.contact[: sim.data.ncon]:
                geoms = [int(contact.geom1), int(contact.geom2)]
                bodies = [sim.model.body_id2name(int(sim.model.geom_bodyid[g])) for g in geoms]
                contacts.append(dict(bodies=bodies, distance=float(contact.dist)))
            return dict(
                time=float(sim.data.time),
                ctrl=ctrl,
                uncoupling=bool(controller.uncoupling),
                initial_joint=np.array(controller.initial_joint, copy=True),
                joint_pos=np.array(controller.joint_pos, copy=True),
                joint_vel=np.array(controller.joint_vel, copy=True),
                goal_pos=np.array(controller.goal_pos, copy=True),
                goal_ori=np.array(controller.goal_ori, copy=True),
                contacts=contacts,
            )

        trace = ControllerTorqueTrace(robot, observe)
        states, poses, twists = [], [], []
        max_error = 0.0
        for k in range(len(commands) + 1):
            if k:
                trace.active = k == len(commands)
                sent = commands[k - 1].copy()
                env.step(sent)
                np.testing.assert_array_equal(sent, commands[k - 1])
            sample, _ = endpoint(env, k)
            twist = measured_twist(env).copy()
            for value, key in (
                (sample["sim_states"], "sim_states"),
                (sample["controller_site_pose"], "controller_site_pose"),
                (twist, "source_twist"),
            ):
                max_error = max(max_error, float(np.max(np.abs(value - expected[key][k]))))
                np.testing.assert_allclose(value, expected[key][k], rtol=0, atol=1e-9, err_msg=f"prefix{k}:{key}")
            states.append(sample["sim_states"])
            poses.append(sample["controller_site_pose"])
            twists.append(twist)
        assert len(trace.records) == 25
        arrays_ref = save_arrays(
            folder / "endpoints.npz",
            dict(
                sim_states=np.stack(states),
                controller_site_pose=np.stack(poses),
                source_twist=np.stack(twists),
                sent_commands=commands,
            ),
        )
        trace_ref = write_once(
            folder / "inner_trace.json",
            dict(
                records=trace.records, phase="after robot.control before integration; contacts from controller forward"
            ),
        )
        e.recheck()
        write_once(
            folder / "collection_result.json",
            dict(
                complete=True,
                group=args.group,
                physics_steps=len(commands),
                explicit_integration_substeps=25 * len(commands),
                environment_initializations=1,
                initialization_internal_steps_not_counted=True,
                inner_records=25,
                maximum_prefix_error=max_error,
                arrays_ref=arrays_ref,
                trace_ref=trace_ref,
                baseline_ref=ref,
                source_sha256=e.hashes,
                model_calls=0,
                new_fits=0,
                fresh_cases=0,
                control_qualification=False,
            ),
        )
        print(dict(group=args.group, complete=True, physics_steps=len(commands), inner_records=25), flush=True)
    finally:
        if trace is not None:
            trace.close()
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
