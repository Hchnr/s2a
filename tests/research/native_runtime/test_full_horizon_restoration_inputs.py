"""Restoration uses the previously analyzed warning state and reference."""

import numpy as np
import pytest

from scripts.tracking.trajectory_search import audit_full_horizon_restoration as audit


def test_restoration_reuses_exact_earlier_warning_and_generated_chunk():
    inputs = audit.load_inputs()
    assert [case["spec"]["prefix_steps"] for case in inputs["cases"]] == [39, 47]
    assert [case["intervention_spec"]["chunk_index"] for case in inputs["cases"]] == [4, 5]


def test_changed_warning_result_is_rejected_before_model_loading(monkeypatch):
    monkeypatch.setattr(audit, "WARNING_RESULT", dict(audit.WARNING_RESULT, sha256="0" * 64))
    with pytest.raises(ValueError, match="Changed input"):
        audit.load_inputs()


def test_joint_only_plan_exposes_all_tcp_errors_without_claiming_quality():
    poses = np.tile(np.eye(4), (8, 1, 1))
    targets = poses.copy()
    targets[-1, 0, 3] = 0.1
    result = audit.with_all_tcp_errors({"witness_found": True, "rollout": {"pose": poses}}, targets, np.eye(4))
    assert result["tcp_errors"].shape == (8, 6)
    assert result["tcp_errors"][-1, 0] == pytest.approx(0.1)
    assert "quality_witness_verified" not in result
