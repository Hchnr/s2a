"""Exercise dispatch boundaries and replay without a planner or GPU."""

from types import SimpleNamespace

import numpy as np
import pytest

from s2a.research.native_runtime.development_continuation import run_continuation


def fake_runtime(*, outside=False):
    class Sim:
        def __init__(self):
            self.data = SimpleNamespace(time=0.0, qpos=np.zeros(7), qvel=np.zeros(7))
            self.inner = 0

        def step(self):
            self.inner += 1
            self.data.time += 0.002
            if outside and self.inner >= 3:
                self.data.qpos[0] = 1.01

    sim = Sim()
    robot = SimpleNamespace(_ref_joint_pos_indexes=np.arange(7), _ref_joint_vel_indexes=np.arange(7))

    class Env:
        robots = [robot]

        def __init__(self):
            self.sim = sim

        def step(self, _command):
            for _ in range(25):
                self.sim.step()

    env = Env()

    def endpoint(_env, index):
        state = np.zeros(79)
        state[0] = sim.data.time
        state[1:8] = sim.data.qpos
        return {"sim_states": state, "source_pose": np.eye(4)}, {"endpoint_step": index}

    observers = SimpleNamespace(endpoint=endpoint, current=lambda _env: None, render_cameras=lambda *_args: None)

    def state_reader(_env):
        return sim.data.qpos.copy(), np.tile([-1.0, 1.0], (7, 1)), np.zeros((6, 7))

    def task_observer(_env):
        return {"success": False, "objects": {"target": {"grasp_contact": False}}}

    terminal = {"endpoint_receipt": {"endpoint_step": 1}}
    return env, observers, state_reader, task_observer, terminal


def execute(*, outside=False, expected=None):
    env, observers, state_reader, task_observer, terminal = fake_runtime(outside=outside)
    calls = []

    def action(q, _limits, _jacobian, _previous, offset):
        calls.append((offset, q.copy()))
        return {"command": np.zeros(7)}

    receipt = {"control_steps_attempted": 0, "control_steps_completed": 0, "newly_selected_commands": 0}
    result = run_continuation(
        env,
        terminal,
        observers,
        receipt,
        start_step=1,
        previous_command=np.zeros(7),
        targets=np.tile(np.eye(4), (8, 1, 1)),
        tool=np.eye(4),
        target_object="target",
        choose_action=action,
        stop_on_outside=True,
        expected_periods=expected,
        state_reader=state_reader,
        task_observer=task_observer,
    )
    return result, receipt, calls


def test_candidate_stops_before_next_dispatch_after_actual_inner_violation():
    result, receipt, calls = execute(outside=True)
    assert len(calls) == result["steps_completed"] == 1
    assert receipt["control_steps_attempted"] == receipt["control_steps_completed"] == 1
    assert result["periods"][0]["inner_outside_samples"] == 23
    assert result["stop"]["next_action_dispatched"] is False


def test_recorded_actions_replay_and_detect_changed_inner_state():
    producer, _, _ = execute()
    checked, _, _ = execute(expected=producer["periods"])
    assert checked["steps_completed"] == 8
    producer["periods"][2]["actual_inner"][3]["q"][0] = 0.1
    with pytest.raises(AssertionError):
        execute(expected=producer["periods"])
