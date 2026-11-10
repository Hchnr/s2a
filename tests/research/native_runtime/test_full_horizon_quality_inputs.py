"""Multi-period inputs must come from the same original warning endpoint."""

import numpy as np
import pytest

from scripts.tracking.trajectory_search import audit_full_horizon_quality as audit


def test_registered_starts_use_verified_first_point_warm_plans():
    inputs = audit.load_inputs()
    cases = inputs["cases"]
    assert [case["spec"]["prefix_steps"] for case in cases] == [43, 52]
    for case in cases:
        warm = case["first_point_warm"]
        assert warm.shape == (8, 6) and np.isfinite(warm).all()
        previous = np.asarray(case["commands"][-1])[:6]
        assert np.max(abs(warm)) <= 1 + 1e-12
        assert np.max(abs(np.diff(np.vstack((previous, warm)), axis=0))) <= 0.25 + 1e-12


def test_changed_model_diagnostic_is_rejected_before_model_loading(monkeypatch):
    monkeypatch.setattr(audit, "FIRST_POINT", dict(audit.FIRST_POINT, sha256="0" * 64))
    with pytest.raises(ValueError, match="Changed input"):
        audit.load_inputs()
