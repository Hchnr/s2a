"""Recovery labels must remain measured, grouped and aligned with the training domain."""

import copy
import json

import numpy as np
import pytest

from s2a.research.wam.training.clean.clean_training_data import DOMAIN
from s2a.research.wam.training.native.native_training import synchronous_labels
from s2a.research.wam.training.recovery_h40_contract import audit_window, source_identity, window_indices
from scripts.verification.wam.training import audit_recovery_h40_contract as run


def window_data():
    pose = np.repeat(np.eye(4)[None], 41, axis=0)
    pose[:, 0, 3] = np.arange(41) * 0.001
    opening = np.linspace(0.2, 0.8, 41)
    times = np.arange(41) * 0.05
    data = dict(
        source_pose=pose,
        actual_source_site_pose=pose.copy(),
        actual_eef_body_pose=pose.copy(),
        opening=opening,
        sim_time=times,
        sim_states=np.zeros((41, 2)),
        endpoint_step=np.arange(41, dtype=np.int64),
        sent_commands=np.zeros((40, 7)),
    )
    data["sent_commands"][:, 6] = -1
    data.update(synchronous_labels(pose, opening, np.eye(4), times))
    transform = dict(
        normalization_mask=[True] * 10 + [False] * 118,
        scale=[1.0] * 9 + [4.0] + [1.0] * 118,
        bias=[0.0] * 128,
        clip_min=-1.0,
        clip_max=1.0,
    )
    normal = dict(domain=DOMAIN, normalization_id=31, projection_id=0, transform=transform)
    return data, normal


def test_h40_retains_measured_next_opening_separately_from_sent_command():
    data, normal = window_data()
    result = audit_window(data, 0, np.eye(4), normal)
    np.testing.assert_array_equal(result["observed_openings"], data["opening"][1:])
    np.testing.assert_array_equal(result["actual_gripper_commands"], np.full(40, -1.0))
    assert result["normalized_shape"] == [40, 10] and result["normalized_dtype"] == "float32"
    assert result["clipped_per_active_slot"][9] == int(np.sum(4 * data["opening"][1:] > 1))
    assert result["clipped_count"] == sum(result["clipped_per_active_slot"])
    assert len(result["normalized_sha256"]) == 64


def test_h40_rejects_command_as_opening_label_and_incomplete_mask():
    data, normal = window_data()
    changed = copy.deepcopy(data)
    changed["action128"][:-1, 9] = changed["sent_commands"][:, 6]
    with pytest.raises(ValueError, match="measured future"):
        audit_window(changed, 0, np.eye(4), normal)
    changed = copy.deepcopy(data)
    changed["action_mask"][20, 9] = False
    with pytest.raises(ValueError, match="active10"):
        audit_window(changed, 0, np.eye(4), normal)
    changed = copy.deepcopy(data)
    changed["actual_source_site_pose"][10, 0, 3] += 0.01
    with pytest.raises(ValueError, match="grip-site"):
        audit_window(changed, 0, np.eye(4), normal)


def test_window_overlap_does_not_create_independent_transitions():
    result = window_indices([0, 16, 56], 40, 96)
    assert result["total"] == 120 and result["unique"] == 96
    with pytest.raises(ValueError, match="complete"):
        window_indices([0, 16, 56], 40, 95)
    with pytest.raises(ValueError, match="Distinct"):
        window_indices([0, 16, 16], 40, 96)


def test_source_group_rejects_forbidden_or_changed_initial_identity():
    sha = "a" * 64
    identity = dict(task_index=0, demo="demo_8", initial_state_sha256=sha)
    case = dict(task_index=0, source_name="task0_native_demo_8", initial_sha256=sha)
    prepared = dict(
        training_anchors=[dict(source_identity=identity)],
        exclusion=dict(forbidden_demo_keys=[], forbidden_initial_state_sha256=[]),
    )
    assert source_identity(case, prepared, (0, "demo_8")) == identity
    bad = copy.deepcopy(prepared)
    bad["exclusion"]["forbidden_demo_keys"].append([0, "demo_8"])
    with pytest.raises(ValueError, match="non-forbidden"):
        source_identity(case, bad, (0, "demo_8"))
    bad = copy.deepcopy(prepared)
    bad["exclusion"]["forbidden_initial_state_sha256"].append(sha)
    with pytest.raises(ValueError, match="non-forbidden"):
        source_identity(case, bad, (0, "demo_8"))
    with pytest.raises(ValueError, match="full task/demo/initial-state"):
        source_identity(dict(case, initial_sha256="b" * 64), prepared, (0, "demo_8"))


def test_audit_configuration_keeps_all_windows_and_forbids_training():
    config = json.loads(run.CONFIG.read_text())
    run.validate_config(config)
    for change in (dict(anchor_steps=[0, 16]), dict(training_authorized=True), dict(fresh_cases=1)):
        with pytest.raises(ValueError, match="exact audit"):
            run.validate_config(dict(config, **change))
    changed = copy.deepcopy(config)
    changed["normalization"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="input identity"):
        run.validate_config(changed)
