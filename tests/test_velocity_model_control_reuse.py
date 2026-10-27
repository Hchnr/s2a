"""Preserve the complete qualified074 physical loop outside the080 gripper intervention."""

import ast
from pathlib import Path


def test_only_planner_current_velocity_and_receipts_change():
    old = Path("src/s2a/residual_motor_simulation.py").read_text()
    new = Path("src/s2a/velocity_model_simulation.py").read_text()
    new = new.replace("def run_velocity_model_case(", "def run_residual_motor_case(").replace(
        "    response_parameters,", "    inverse_parameters,"
    )
    new = new.replace(
        '    motor_planner = VelocityModelPlanner(response_parameters, velocity_envelope=config["gripper_velocity_envelope"])',
        "    motor_inverse = ResidualMotorInverse(inverse_parameters)",
    )
    new = new.replace(
        '                gripper_algorithm="080 frozen079 finite-command planner",\n                gripper_velocity_envelope=config["gripper_velocity_envelope"],\n',
        "",
    )
    new = new.replace(
        "        endpoint_observer(env, 0)\n", "        endpoint_observer(env, 0)\n        previous_opening = None\n"
    )
    block = """                gripper_velocity = np.asarray(obs["robot0_gripper_qvel"], dtype=float).copy()
                direct_velocity = np.array([env.sim.data.qvel[i] for i in env.robots[0]._ref_gripper_joint_vel_indexes])
                np.testing.assert_allclose(gripper_velocity, direct_velocity, rtol=0, atol=1e-12)
                period_velocity = dt * (gripper_velocity[0] - gripper_velocity[1]) / 0.079
                grip_result = motor_planner.plan(opening, motor_before, period_velocity, full_openings[index - 1 :])"""
    assert new.count(block) == 1
    new = new.replace(
        block,
        """                grip_result = motor_inverse.update(
                    opening,
                    motor_before,
                    full_openings[index - 1 :],
                    previous_opening=previous_opening,
                )
                previous_opening = opening""",
    )
    new = new.replace(
        "                    measured_gripper_qvel=gripper_velocity,\n                    measured_period_velocity=period_velocity,\n",
        "",
    )

    def functions(text):
        return {
            n.name: ast.dump(n, include_attributes=False)
            for n in ast.parse(text).body
            if isinstance(n, ast.FunctionDef)
        }

    assert functions(new) == functions(old)
