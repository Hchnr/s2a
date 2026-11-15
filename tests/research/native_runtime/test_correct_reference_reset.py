"""Oracle reset admission must preserve source and measured-state distinctions."""

import copy

import numpy as np
import pytest

from s2a.research.native_runtime.correct_reference_reset import verify_correct_reset
from s2a.research.wam.training.native.native_training import gripper_cache_receipt
from tests.research.native_runtime.test_reset_endpoint import sample


def test_oracle_reset_verifies_state_without_a_model_request():
    expected, receipt, chunk, controller = sample()
    result = verify_correct_reset(
        expected, receipt, copy.deepcopy(expected), receipt, controller, np.eye(4), chunk["anchor_tcp"], 0.8
    )
    assert result["state_bytes_equal"] and result["max_field_difference"] == 0
    assert result["reference_kind"] == "recorded_correct_oracle" and result["world_model_request"] is False
    assert "request_sha256" not in result and "image_sha256" not in result


@pytest.mark.parametrize("failure", ["anchor", "opening", "motor", "receipt", "controller"])
def test_oracle_reset_rejects_misaligned_observation_or_contract(failure):
    expected, receipt, chunk, controller = sample()
    actual, actual_receipt = copy.deepcopy(expected), dict(receipt)
    anchor, opening = chunk["anchor_tcp"].copy(), 0.8
    if failure == "anchor":
        anchor[0, 3] += 0.01
    elif failure == "opening":
        opening = 0.7
    elif failure == "motor":
        actual["gripper_current_action"][0] = 0.1
        actual_receipt.update(gripper_cache_receipt(actual["gripper_current_action"]))
    elif failure == "receipt":
        actual_receipt["sim_state_sha256"] = "incorrect"
    elif failure == "controller":
        controller["actual_initial_physics_sha256"] = "incorrect"
    with pytest.raises(ValueError):
        verify_correct_reset(expected, receipt, actual, actual_receipt, controller, np.eye(4), anchor, opening)
