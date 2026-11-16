"""Independently reconstruct FP32 labels and split metadata."""

import copy

import numpy as np
import pytest

from s2a.research.wam.training.broader_recovery_h40 import clipping_flags
from s2a.research.wam.training.clean.clean_training_data import normalize_active10
from scripts.verification.wam.training import verify_broader_recovery_h40 as run
from tests.research.wam.training.test_recovery_event_coverage import metadata
from tests.research.wam.training.test_recovery_h40_contract import window_data


def test_scalar_fp32_normalization_matches_frozen_window_byte_for_byte():
    data, normalization = window_data()
    actions = data["action128"][:-1].copy()
    actions[:3, 9] = [
        0.25,
        np.nextafter(np.float32(0.25), np.float32(0)),
        np.nextafter(np.float32(0.25), np.float32(1)),
    ]
    expected = normalize_active10(actions, **normalization)
    actual, flags = run.scalar_normalization(actions, normalization)
    assert actual.tobytes() == expected["normalized_active10"].tobytes()
    np.testing.assert_array_equal(flags, clipping_flags(actions, normalization))
    np.testing.assert_array_equal(flags.sum(axis=0), expected["clipped_per_active_slot"])
    assert flags.sum() == expected["clipped_count"]


def test_independent_identity_sets_detect_overlap_with_validation():
    groups, training, preflight, validation, verification = metadata()
    groups += [dict(task_index=i, demo=f"demo_{i + 300}", initial_state_sha256=f"{i + 500:064x}") for i in range(3)]
    training["training_anchors"] = [dict(source_identity=copy.deepcopy(row), anchor_frame=0) for row in groups]
    training["exclusion"].update(forbidden_demo_keys=[], forbidden_initial_state_sha256=[])
    run.check_source_separation(groups, training, preflight, validation, verification)
    for field in ("demo", "initial_state_sha256"):
        changed = copy.deepcopy(groups)
        changed[0][field] = preflight["records"][0][field]
        altered_training = copy.deepcopy(training)
        altered_training["training_anchors"][0]["source_identity"][field] = changed[0][field]
        with pytest.raises(ValueError, match="identities must be disjoint"):
            run.check_source_separation(changed, altered_training, preflight, validation, verification)


def test_command_counts_treat_signed_zero_as_the_same_command():
    assert run.count_commands([-1.0, -0.0, 0.0, 1.0]) == {"-1.0": 1, "0.0": 2, "1.0": 1}
