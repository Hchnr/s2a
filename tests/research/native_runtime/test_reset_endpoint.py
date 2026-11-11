"""Reset admission must bind the observed state, motor target and first query."""

import copy

import numpy as np
import pytest

from s2a.research.native_runtime.development_prefix_replay import SAMPLE_FIELDS
from s2a.research.native_runtime.reset_endpoint import saved_reset_endpoint, verify_reset_endpoint
from s2a.research.wam.training.native.native_training import gripper_cache_receipt, state_sha256


def sample():
    state = np.array([0.25, 0.1, 0.2], dtype=np.float64)
    values = {name: np.zeros(2) for name in SAMPLE_FIELDS}
    values.update(
        sim_states=state,
        sim_time=np.array(0.25),
        source_pose=np.eye(4),
        opening=np.array(0.8),
        gripper_current_action=np.zeros(2),
    )
    receipt = {
        "endpoint_step": 0,
        "sim_state_sha256": state_sha256(state),
        "sim_state_dtype": "float64",
        "sim_state_shape": [3],
        "sim_time": 0.25,
        **gripper_cache_receipt(values["gripper_current_action"]),
    }
    chunk = {
        "step": 0,
        "physics_before_sha256": receipt["sim_state_sha256"],
        "physics_after_sha256": receipt["sim_state_sha256"],
        "physics_unchanged": True,
        "physics_before_time_s": 0.25,
        "physics_after_time_s": 0.25,
        "anchor_opening": 0.8,
        "anchor_tcp": np.eye(4),
        "effective_targets_tcp": np.tile(np.eye(4), (40, 1, 1)),
        "openings": np.ones(40),
        "image_sha256": "image",
        "request_sha256": "request",
        "response_sha256": "response",
    }
    controller = {
        "actual_initial_physics_sha256": receipt["sim_state_sha256"],
        "aligned_initial_motor_target": [0.0, 0.0],
    }
    return values, receipt, chunk, controller


def test_reset_archive_exposes_only_initial_state_and_motor():
    values, _receipt, _chunk, _controller = sample()
    archive = {name: np.stack((values[name], values[name] + 1)) for name in SAMPLE_FIELDS}
    archive.update(
        endpoint_step=np.array([0, 1]),
        gripper_current_action_offsets=np.array([0, 2, 4]),
        gripper_current_action_flat=np.array([0.0, 0.0, 1.0, 1.0]),
        native_success=np.array([False, True]),
    )
    first = saved_reset_endpoint(archive)
    assert "native_success" not in first and len(first) == len(SAMPLE_FIELDS) + 1
    np.testing.assert_array_equal(first["sim_states"], values["sim_states"])
    np.testing.assert_array_equal(first["gripper_current_action"], [0.0, 0.0])
    archive["gripper_current_action_offsets"][1] = 3
    with pytest.raises(ValueError, match="motor cache"):
        saved_reset_endpoint(archive)


def test_reset_checks_state_identity_reference_and_motor_independently():
    expected, receipt, chunk, controller = sample()
    actual = copy.deepcopy(expected)
    result = verify_reset_endpoint(expected, receipt, actual, receipt, chunk, controller, np.eye(4))
    assert result["state_bytes_equal"] and result["max_field_difference"] == 0
    actual["gripper_current_action"][0] = 0.01
    actual_receipt = dict(receipt, **gripper_cache_receipt(actual["gripper_current_action"]))
    with pytest.raises(ValueError, match="gripper_current_action"):
        verify_reset_endpoint(expected, receipt, actual, actual_receipt, chunk, controller, np.eye(4))
    broken = dict(chunk, physics_before_sha256="wrong")
    with pytest.raises(ValueError, match="reference"):
        verify_reset_endpoint(expected, receipt, expected, receipt, broken, controller, np.eye(4))
    broken_receipt = dict(receipt, sim_state_sha256="wrong")
    with pytest.raises(ValueError, match="receipt"):
        verify_reset_endpoint(expected, receipt, expected, broken_receipt, chunk, controller, np.eye(4))
