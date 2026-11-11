"""The physical branch must use the aligned raw chunk already generated."""

import numpy as np
import pytest

from scripts.simulation.runtime.prepare_cached_calibration_gripper import validate_chunk


def aligned():
    poses = np.repeat(np.eye(4)[None], 40, axis=0)
    anchor = np.eye(4)
    case = {"spec": {"prefix_steps": 32}, "targets": poses.copy(), "anchor": anchor.copy()}
    chunk = {
        "step": 32,
        "model_rotation_alpha": 1,
        "retiming": False,
        "effective_targets_tcp": poses.copy(),
        "raw_targets_tcp": poses.copy(),
        "anchor_tcp": anchor.copy(),
        "openings": [0.8] * 40,
    }
    return case, chunk


def test_matched_chunk_and_query_anchor_are_required():
    case, chunk = aligned()
    assert validate_chunk(case, "task2", chunk) is case
    chunk["step"] += 8
    with pytest.raises(ValueError, match="already|generated|raw chunk"):
        validate_chunk(case, "task2", chunk)
    case, chunk = aligned()
    chunk["anchor_tcp"][0, 3] = 0.01
    with pytest.raises(AssertionError):
        validate_chunk(case, "task2", chunk)


def test_changed_raw_reference_and_retiming_are_rejected():
    case, chunk = aligned()
    chunk["effective_targets_tcp"][0, 0, 3] = 0.01
    with pytest.raises(ValueError, match="raw chunk"):
        validate_chunk(case, "task2", chunk)
    case, chunk = aligned()
    chunk["retiming"] = {"different": True}
    with pytest.raises(ValueError, match="raw chunk"):
        validate_chunk(case, "task2", chunk)
