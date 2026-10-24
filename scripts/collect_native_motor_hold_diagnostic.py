"""Retain native motor targets by command zero; record real 500Hz controls."""

import argparse
import os
import sys
import traceback
from pathlib import Path
from unittest.mock import patch

import collect_wam_native_training as native_collector
import numpy as np
from collect_native_pose_hold_diagnostic import GoalRecordingEnvironment
from run_native_motor_hold_diagnostic import read_prepared, require

from s2a.native_environment import restore_cv2_environment, strict_native_environment
from s2a.native_training import native_command_schedule


def motor_hold_schedule(native):
    schedule = native_command_schedule(native)
    schedule[len(native) :, -1] = 0
    return schedule


class MotorRecordingEnvironment(GoalRecordingEnvironment):
    def __init__(self, env, native_steps, method):
        super().__init__(env, native_steps, "latched_pose" if method == "pose_motor_hold" else "baseline")
        self.robot = env.robots[0]
        self.original_grip_action = self.robot.grip_action
        self.robot.grip_action = self.grip_action
        self.motor_calls, self.motor_targets = [], []
        self.substeps = 0
        self.actuator_names = list(self.robot.gripper.actuators)
        self.actuator_ids = [env.sim.model.actuator_name2id(name) for name in self.actuator_names]
        require(len(self.actuator_ids) == 2, "two actual finger actuators required")

    def motor_values(self):
        return dict(
            actuator_ctrl=np.asarray(self.actual.sim.data.ctrl[self.actuator_ids]).copy(),
            actuator_ctrlrange=np.asarray(self.actual.sim.model.actuator_ctrlrange[self.actuator_ids]).copy(),
            actuator_names=self.actuator_names,
        )

    def grip_action(self, gripper, gripper_action):
        require(gripper is self.robot.gripper, "unexpected native gripper")
        before = np.asarray(gripper.current_action).copy()
        command = np.asarray(gripper_action).copy()
        timestamp = float(self.actual.sim.data.time)
        value = self.original_grip_action(gripper=gripper, gripper_action=gripper_action)
        self.motor_calls.append(
            dict(
                step_index=self.steps,
                substep_index=self.substeps,
                sim_time=timestamp,
                input_action=command,
                cache_before=before,
                cache_after=np.asarray(gripper.current_action).copy(),
                **self.motor_values(),
            )
        )
        self.substeps += 1
        return value

    def step(self, action):
        self.substeps = 0
        value = super().step(action)
        require(self.substeps == 25, "exactly25 actual grip updates required")
        self.motor_targets.append(
            dict(
                step_index=self.steps - 1,
                sim_time=float(self.actual.sim.data.time),
                current_action=np.asarray(self.robot.gripper.current_action).copy(),
                **self.motor_values(),
            )
        )
        return value

    def close(self):
        self.robot.grip_action = self.original_grip_action
        super().close()


def run(args):
    ref = dict(path=str(Path(args.prepared).resolve()), sha256=args.prepared_sha256)
    prepared = read_prepared(ref)
    folder = Path(prepared["root"]) / "collection"
    folder.mkdir(exist_ok=False)
    prior = prepared["pose_prepared"]["static_prepared"]
    result = dict(passed=False, complete=False, cases=[], prepared_ref=ref, training_authorized=False)
    progress = dict(environment_initializations=0, physics_steps=0, current_canvas_images=0)
    try:
        readiness = restore_cv2_environment(prepared["cv2_probe_ref"], prepared["cv2_runtime_source_refs"])
        native_collector.write_once(folder / "environment_readiness.json", readiness)
        native_collector.write_once(
            folder / "worker_started.json",
            dict(
                pid=os.getpid(),
                ppid=os.getppid(),
                argv=sys.argv,
                python_version=sys.version,
                cuda=os.environ.get("CUDA_VISIBLE_DEVICES"),
                ld=os.environ.get("LD_LIBRARY_PATH"),
            ),
        )
        from s2a import libero_simulation

        original_make = libero_simulation.make_environment
        for method in prepared["config"]["methods"]:
            method_root = folder / method
            method_root.mkdir()
            for record in prepared["records"]:
                wrappers = []

                def make(policy, output, source, trial):
                    env, obs = original_make(policy, output, source, trial)
                    wrapped = MotorRecordingEnvironment(env, source["native_steps"], method)
                    wrappers.append(wrapped)
                    return wrapped, obs

                with (
                    patch.object(libero_simulation, "make_environment", make),
                    patch.object(native_collector, "native_command_schedule", motor_hold_schedule),
                ):
                    case_ref = native_collector.collect_case(
                        prior["native_prepared"], ref, record, method_root / record["name"], progress
                    )
                require(len(wrappers) == 1, "one environment required")
                wrapper = wrappers[0]
                destination = method_root / record["name"]
                result["cases"].append(
                    dict(
                        method=method,
                        name=record["name"],
                        case=case_ref,
                        goals=native_collector.write_once(destination / "controller_goals.json", wrapper.goals),
                        motor_targets=native_collector.write_once(
                            destination / "motor_targets.json", wrapper.motor_targets
                        ),
                        motor_calls=native_collector.write_once(destination / "motor_calls.json", wrapper.motor_calls),
                    )
                )
                strict_native_environment()
                read_prepared(ref)
        require(progress == dict(environment_initializations=8, physics_steps=1648, current_canvas_images=40), "budget")
        result.update(passed=True, complete=True)
    except BaseException as error:
        result["error"] = dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
        raise
    finally:
        result["progress"] = progress
        native_collector.write_once(folder / "result.json", result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--prepared-sha256", required=True)
    parser.add_argument("--seed", type=int, choices=(42,), required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
