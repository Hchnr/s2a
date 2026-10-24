"""A fixed native prefix followed by measured-pose settling and verification."""

import numpy as np

from s2a.native_training import native_anchor_indices, native_command_schedule

HOLD_SEMANTICS = (
    "hold native-end measured controller SITE pose; continue last native gripper command; "
    "first 40 settling, last 40 verification"
)


def settled_anchor_indices(native_steps):
    return native_anchor_indices(native_steps)[:4] + [int(native_steps) + 40]


def settled_command_schedule(native):
    schedule = native_command_schedule(native)
    return np.concatenate((schedule, schedule[-40:].copy()))


class SettledGoalEnvironment:
    """Override only the 80 terminal controller goals with one current measurement.

    The last native gripper command continues through the normal controller.
    No task predicate, future state or measured opening enters a command.
    """

    def __init__(self, env, native_steps):
        native_anchor_indices(native_steps)
        self.actual = env
        self.native_steps = int(native_steps)
        self.steps = 0
        self.latch = None
        self.goals = []
        self.controller = env.robots[0].controller
        self.original_set_goal = self.controller.set_goal
        self.controller.set_goal = self.set_goal

    def __getattr__(self, name):
        return getattr(self.actual, name)

    def set_goal(self, action, set_pos=None, set_ori=None):
        if set_pos is not None or set_ori is not None:
            raise ValueError("unexpected existing absolute goal")
        active = self.steps >= self.native_steps
        if active:
            if self.latch is None:
                raise ValueError("native-end measured controller SITE pose missing")
            set_pos, set_ori = (value.copy() for value in self.latch)
        result = self.original_set_goal(action, set_pos=set_pos, set_ori=set_ori)
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
        return result

    def step(self, action):
        if self.steps >= self.native_steps + 80:
            raise ValueError("settled native step budget exceeded")
        if self.steps == self.native_steps:
            data = self.actual.sim.data
            name = self.controller.eef_name
            self.latch = (
                np.asarray(data.get_site_xpos(name)).copy(),
                np.asarray(data.get_site_xmat(name)).reshape(3, 3).copy(),
            )
        before = len(self.goals)
        result = self.actual.step(action)
        if len(self.goals) != before + 1:
            raise ValueError("expected one controller goal per physical control step")
        self.steps += 1
        return result

    def close(self):
        self.controller.set_goal = self.original_set_goal
        self.actual.close()
