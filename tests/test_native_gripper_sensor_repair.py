"""Exact collector repair scope: no command/control or tolerance change."""

import ast
from pathlib import Path


def test_only_namespace_and_redundant_forward_sensor_read_change():
    old = Path("scripts/collect_native_gripper_response.py").read_text()
    new = Path("scripts/collect_native_gripper_response_v2.py").read_text()
    new = new.replace(
        "native_gripper_response_interventions_20261025_v2", "native_gripper_response_interventions_20261025"
    )
    new = new.replace(
        "controller_contract, make_environment", "controller_contract, current_observation, make_environment"
    )
    assert new.count("env.env._get_observations(force_update=True)") == 1
    new = new.replace("env.env._get_observations(force_update=True)", "current_observation(env)")
    assert ast.dump(ast.parse(new), include_attributes=False) == ast.dump(ast.parse(old), include_attributes=False)
