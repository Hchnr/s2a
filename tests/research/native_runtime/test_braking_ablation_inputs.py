"""Ablation must bind the same state and a verified strict-braking warm plan."""

import numpy as np
import pytest

from scripts.tracking.trajectory_search import audit_braking_constraint_ablation as audit


def test_ablation_inputs_use_verified_strict_warm_plans():
    inputs = audit.load_inputs()
    assert [case["spec"]["prefix_steps"] for case in inputs["cases"]] == [39, 47]
    for case in inputs["cases"]:
        plan = case["strict_warm"]
        previous = case["commands"][-1][:6]
        assert plan.shape == (8, 6) and np.isfinite(plan).all()
        assert np.max(abs(plan)) <= 1 + 1e-12
        assert np.max(abs(np.diff(np.vstack((previous, plan)), axis=0))) <= 0.25 + 1e-12
        assert case["strict_normals"].ndim == 2 and case["strict_normals"].shape[1] == 7


def test_changed_strict_result_is_rejected_before_model_construction(monkeypatch):
    monkeypatch.setattr(audit, "STRICT_RESULT", dict(audit.STRICT_RESULT, sha256="0" * 64))
    with pytest.raises(ValueError, match="Changed input"):
        audit.load_inputs()
