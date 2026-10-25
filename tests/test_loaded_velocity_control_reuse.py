"""Complete080 control loop remains unchanged outside081 grip routing/history."""

import ast
from pathlib import Path


def test_only_route_initialization_causal_command_history_and_receipt_change():
    old = Path("src/s2a/velocity_model_simulation.py").read_text()
    new = Path("src/s2a/loaded_velocity_simulation.py").read_text()
    new = new.replace("def run_loaded_velocity_case(", "def run_velocity_model_case(")
    new = new.replace(
        '    VelocityModelPlanner(response_parameters, velocity_envelope=config["gripper_velocity_envelope"])',
        '    motor_planner = VelocityModelPlanner(response_parameters, velocity_envelope=config["gripper_velocity_envelope"])',
    )
    init = """        loaded_controller = LoadedVelocityControl(
            response_parameters, initial_opening, velocity_envelope=config["gripper_velocity_envelope"]
        )
"""
    assert new.count(init) == 1
    new = new.replace(init, "")
    new = new.replace("081 native-close loading and frozen079 tracking", "080 frozen079 finite-command planner")
    new = new.replace("        previous_gripper_command = 0.0\n", "")
    new = new.replace("                previous_gripper_command = float(grip_command)\n", "")
    call = """                grip_result = loaded_controller.update(
                    opening,
                    motor_before,
                    period_velocity,
                    full_effective_targets,
                    full_openings,
                    index - 1,
                    measured_tcp,
                    previous_command=previous_gripper_command,
                    dt=dt,
                )"""
    assert new.count(call) == 1
    new = new.replace(
        call,
        "                grip_result = motor_planner.plan(opening, motor_before, period_velocity, full_openings[index - 1 :])",
    )

    def functions(text):
        return {
            n.name: ast.dump(n, include_attributes=False)
            for n in ast.parse(text).body
            if isinstance(n, ast.FunctionDef)
        }

    assert functions(new) == functions(old)
