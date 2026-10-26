from types import SimpleNamespace

import numpy as np

from s2a.controller_trace import ControllerTorqueTrace


def test_trace_preserves_command_return_and_clipped_torque_and_restores():
    raw = np.array([2.0, -3.0])
    controller = SimpleNamespace(run_controller=lambda: raw)
    robot = SimpleNamespace(controller=controller, torque_limits=(np.array([-1.0, -2.0]), np.ones(2)))
    commands = []

    def control(command):
        commands.append(command.copy())
        output = controller.run_controller()
        assert output is raw
        robot.torques = np.clip(output, *robot.torque_limits)
        return "unchanged"

    robot.control = control
    original_run = controller.run_controller
    trace = ControllerTorqueTrace(robot, lambda: dict(time=0.0))
    command = np.array([0.2, 0.7])
    assert robot.control(command) == "unchanged"
    assert trace.records == []
    trace.active = True
    assert robot.control(command) == "unchanged"
    np.testing.assert_array_equal(commands, [command, command])
    np.testing.assert_array_equal(trace.records[0]["raw_torque"], [2, -3])
    np.testing.assert_array_equal(trace.records[0]["clipped_torque"], [1, -2])
    np.testing.assert_array_equal(raw, [2, -3])
    trace.close()
    assert robot.control is control and controller.run_controller is original_run
