"""Measured capture must preserve causality and partial physical prefixes."""

import copy

import numpy as np
import pytest

from s2a.research.native_runtime.training_recovery_capture import TrainingRecoveryCapture
from s2a.research.wam.training.native.native_training import gripper_cache_receipt, state_sha256


def sample(index):
    state = np.array([index * 0.05, index], dtype=np.float64)
    pose = np.eye(4)
    pose[0, 3] = index * 0.001
    value = dict(
        endpoint_step=np.int64(index),
        sim_time=np.float64(index * 0.05),
        native_success=np.bool_(False),
        sim_states=state,
        source_pose=pose,
        opening=np.float64(0.8),
        gripper_current_action=np.zeros(2),
    )
    receipt = dict(
        endpoint_step=index,
        sim_time=index * 0.05,
        native_success=False,
        sim_state_sha256=state_sha256(state),
        **gripper_cache_receipt(value["gripper_current_action"]),
    )
    return value, receipt


def image(_env, index, _sample, receipt, _directory):
    return dict(frame_index=index, current_endpoint=copy.deepcopy(receipt))


def test_actual_capture_and_anchors_keep_partial_future_unavailable(tmp_path):
    capture = TrainingRecoveryCapture(np.eye(4), tmp_path, image)
    initial, receipt = sample(0)
    capture.append(None, initial, receipt)
    reader = capture.wrap_endpoint(lambda _env, step: sample(step))
    for step in range(1, 58):
        reader(None, step)
    initial["source_pose"][0, 3] = 99
    arrays = capture.arrays()
    assert len(arrays["source_pose"]) == 58 and arrays["source_pose"][0, 0, 3] == 0
    assert arrays["action_valid"][:-1].all() and not arrays["action_valid"][-1]
    assert [item["frame_index"] for item in capture.images] == [0, 16, 56]
    assert capture.anchor_windows() == [
        dict(step=0, image_reached=True, full_h40=True),
        dict(step=16, image_reached=True, full_h40=True),
        dict(step=56, image_reached=True, full_h40=False),
    ]
    with pytest.raises(ValueError, match="Consecutive actual endpoint"):
        reader(None, 57)


def test_reset_only_produces_no_action_label_and_rejects_tampered_receipts(tmp_path):
    capture = TrainingRecoveryCapture(np.eye(4), tmp_path, image)
    initial, receipt = sample(0)
    bad = dict(receipt, sim_state_sha256="changed")
    with pytest.raises(ValueError, match="Consecutive actual endpoint"):
        capture.append(None, initial, bad)
    capture.append(None, initial, receipt)
    assert "action128" not in capture.arrays()
    assert not any(item["full_h40"] for item in capture.anchor_windows())
