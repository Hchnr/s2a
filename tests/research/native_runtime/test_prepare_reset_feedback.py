"""Feedback preserves the original gripper and reference contract."""

import copy

import pytest

from scripts.simulation.runtime.prepare_reset_feedback import validate_gripper_contract


def test_original_gripper_parameters_config_and_clock_are_required():
    ref = {"path": "/repo/parameters.json", "sha256": "abc"}
    manifest = {
        "variant": {"governor": False},
        "experiment": {
            "response_parameters_ref": ref,
            "control_fps": 20,
            "gripper_velocity_envelope": 0.3,
            "gripper_position_hold": {"close_threshold": 0.6},
        },
    }
    controller = {
        "gripper_velocity_envelope": 0.3,
        "gripper_config": {"close_threshold": 0.6},
        "gripper_algorithm": "081 native-close loading and frozen079 tracking",
    }
    assert validate_gripper_contract(manifest, controller, {ref["path"]: "abc"}) == ref
    altered = copy.deepcopy(controller)
    altered["gripper_config"]["close_threshold"] = 0.5
    with pytest.raises(ValueError, match="gripper parameters"):
        validate_gripper_contract(manifest, altered, {ref["path"]: "abc"})
    with pytest.raises(ValueError, match="gripper parameters"):
        validate_gripper_contract(manifest, controller, {ref["path"]: "wrong"})
