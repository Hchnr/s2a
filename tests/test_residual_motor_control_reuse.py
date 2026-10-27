"""Prove the entire physical loop is unchanged outside the registered intervention."""

import ast
from pathlib import Path


def test_only_residual_gripper_and_causal_history_change():
    old = Path("src/s2a/motor_state_simulation.py").read_text()
    new = Path("src/s2a/residual_motor_simulation.py").read_text()
    new = new.replace("from s2a.residual_motor_inverse import ResidualMotorInverse\n", "")
    new = new.replace("def run_residual_motor_case(", "def run_motor_state_case(")
    new = new.replace("    inverse_parameters,\n", "")
    new = new.replace(
        '    if variant["governor"]:\n        raise ValueError("Residual inverse requires ungoverned endpoint reference")\n    motor_inverse = ResidualMotorInverse(inverse_parameters)\n',
        "",
    )
    new = new.replace(
        '    if c["control_fps"] != 20:\n        raise ValueError("Frozen residual inverse requires20Hz native cycles")\n',
        "",
    )
    new = new.replace("        previous_opening = None", "        previous_command = np.zeros(7)")
    new = new.replace(
        """                grip_result = motor_inverse.update(
                    opening,
                    motor_before,
                    full_openings[index - 1 :],
                    previous_opening=previous_opening,
                )
                previous_opening = opening""",
        """                grip_result = gripper.update(
                    desired_opening,
                    opening,
                    motor_opening=motor_opening,
                    previous_command=previous_command[-1],
                    dt=dt,
                )""",
    )
    new = new.replace(
        "                endpoint_observer(env, step + 1)\n",
        "                endpoint_observer(env, step + 1)\n                previous_command = command.copy()\n",
    )
    assert ast.dump(ast.parse(new), include_attributes=False) == ast.dump(ast.parse(old), include_attributes=False)
