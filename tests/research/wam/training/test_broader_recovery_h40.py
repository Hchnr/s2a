"""Preserve FP32 clipping semantics and source-disjoint validation metadata."""

import copy

import numpy as np
import pytest

from s2a.research.wam.training.broader_recovery_h40 import clipping_flags, validate_six_sources
from s2a.research.wam.training.recovery_h40_contract import audit_window
from tests.research.wam.training.test_recovery_event_coverage import metadata
from tests.research.wam.training.test_recovery_h40_contract import window_data


def test_row_level_clip_flags_match_original_frozen_fp32_window():
    data, normalization = window_data()
    flags = clipping_flags(data["action128"][:-1], normalization)
    result = audit_window(data, 0, np.eye(4), normalization)
    assert flags.shape == (40, 10) and flags.dtype == np.bool_
    np.testing.assert_array_equal(flags.sum(axis=0), result["clipped_per_active_slot"])
    assert flags.sum() == result["clipped_count"]
    assert clipping_flags(data["action128"][:7], normalization).shape == (7, 10)


def test_clipping_rejects_changed_domain_masks_or_nonfinite_labels():
    data, normalization = window_data()
    bad = copy.deepcopy(normalization)
    bad["domain"] = "different"
    with pytest.raises(ValueError, match="normalization contract"):
        clipping_flags(data["action128"][:-1], bad)
    bad = copy.deepcopy(normalization)
    bad["transform"]["normalization_mask"][17] = True
    with pytest.raises(ValueError, match="normalization contract"):
        clipping_flags(data["action128"][:-1], bad)
    labels = data["action128"][:-1].copy()
    labels[0, 0] = np.nan
    with pytest.raises(ValueError, match="normalization contract"):
        clipping_flags(labels, normalization)


def test_six_sources_remain_disjoint_from_validation_and_fresh():
    groups, training, preflight, validation, verification = metadata()
    groups += [dict(task_index=i, demo=f"demo_{i + 300}", initial_state_sha256=f"{i + 500:064x}") for i in range(3)]
    result = validate_six_sources(groups, training, preflight, validation, verification)
    assert result["training_groups"] == 6 and result["identity_sets_disjoint"]
    assert not result["validation_future_labels_read"] and not result["fresh_future_labels_read"]
    altered = copy.deepcopy(groups)
    altered[5]["initial_state_sha256"] = altered[0]["initial_state_sha256"]
    with pytest.raises(ValueError, match="Six unique"):
        validate_six_sources(altered, training, preflight, validation, verification)
    altered = copy.deepcopy(groups)
    altered[5]["initial_state_sha256"] = preflight["records"][0]["initial_state_sha256"]
    with pytest.raises(ValueError, match="identities differ"):
        validate_six_sources(altered, training, preflight, validation, verification)
