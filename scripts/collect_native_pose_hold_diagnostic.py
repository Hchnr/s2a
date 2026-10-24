"""Reproduce four native prefixes and intervene only on terminal OSC goals."""

import argparse
import os
import sys
import traceback
from pathlib import Path
from unittest.mock import patch

import numpy as np
from collect_wam_native_training import collect_case, write_once
from run_native_pose_hold_diagnostic import read_prepared, require

from s2a.native_environment import restore_cv2_environment, strict_native_environment


class GoalRecordingEnvironment:
    def __init__(self, env, native_steps, method):
        self.actual = env
        self.native_steps = native_steps
        self.method = method
        self.steps = 0
        self.latch = None
        self.goals = []
        self.controller = env.robots[0].controller
        self.original_set_goal = self.controller.set_goal
        self.controller.set_goal = self.set_goal

    def __getattr__(self, name):
        return getattr(self.actual, name)

    def set_goal(self, action, set_pos=None, set_ori=None):
        require(set_pos is None and set_ori is None, "unexpected existing absolute goal")
        active = self.method == "latched_pose" and self.steps >= self.native_steps
        if active:
            require(self.latch is not None, "terminal measurement not captured")
            set_pos, set_ori = self.latch
        value = self.original_set_goal(action, set_pos=set_pos, set_ori=set_ori)
        self.goals.append(
            dict(
                step_index=self.steps,
                action=np.asarray(action).copy(),
                overridden=active,
                set_pos=None if set_pos is None else set_pos.copy(),
                set_ori=None if set_ori is None else set_ori.copy(),
                actual_goal_pos=np.asarray(self.controller.goal_pos).copy(),
                actual_goal_ori=np.asarray(self.controller.goal_ori).copy(),
            )
        )
        return value

    def step(self, action):
        require(self.steps < self.native_steps + 40, "step budget exceeded")
        if self.steps == self.native_steps:
            name = self.controller.eef_name
            self.latch = (
                np.asarray(self.actual.sim.data.get_site_xpos(name)).copy(),
                np.asarray(self.actual.sim.data.get_site_xmat(name)).reshape(3, 3).copy(),
            )
        before = len(self.goals)
        result = self.actual.step(action)
        require(len(self.goals) == before + 1, "expected exactly one policy goal per physical control step")
        self.steps += 1
        return result

    def close(self):
        self.controller.set_goal = self.original_set_goal
        self.actual.close()


def run(args):
    ref = dict(path=str(Path(args.prepared).resolve()), sha256=args.prepared_sha256)
    prepared = read_prepared(ref)
    root = Path(prepared["root"])
    folder = root / "collection"
    folder.mkdir(exist_ok=False)
    prior = prepared["static_prepared"]
    result = dict(passed=False, complete=False, cases=[], prepared_ref=ref, training_authorized=False)
    progress = dict(environment_initializations=0, physics_steps=0, current_canvas_images=0)
    try:
        readiness = restore_cv2_environment(prior["cv2_probe_ref"], prior["cv2_runtime_source_refs"])
        write_once(folder / "environment_readiness.json", readiness)
        write_once(folder / "worker_started.json", dict(pid=os.getpid(), ppid=os.getppid(), argv=sys.argv))
        from s2a import libero_simulation

        actual_make = libero_simulation.make_environment
        for method in prepared["config"]["methods"]:
            method_root = folder / method
            method_root.mkdir()
            for record in prepared["records"]:
                wrappers = []

                def make(policy, output, source, trial):
                    env, obs = actual_make(policy, output, source, trial)
                    wrapped = GoalRecordingEnvironment(env, source["native_steps"], method)
                    wrappers.append(wrapped)
                    return wrapped, obs

                with patch.object(libero_simulation, "make_environment", make):
                    case_ref = collect_case(
                        prior["native_prepared"], ref, record, method_root / record["name"], progress
                    )
                require(len(wrappers) == 1, "exactly one environment required")
                wrapper = wrappers[0]
                goals = write_once(method_root / record["name"] / "controller_goals.json", wrapper.goals)
                result["cases"].append(dict(method=method, name=record["name"], case=case_ref, goals=goals))
                strict_native_environment()
                read_prepared(ref)
        require(progress == dict(environment_initializations=8, physics_steps=1648, current_canvas_images=40), "budget")
        result.update(passed=True, complete=True)
    except BaseException as error:
        result["error"] = dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
        raise
    finally:
        result["progress"] = progress
        write_once(folder / "result.json", result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--prepared-sha256", required=True)
    parser.add_argument("--seed", type=int, choices=(42,), required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
