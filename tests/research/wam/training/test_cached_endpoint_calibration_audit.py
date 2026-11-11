"""Independent audit rejects a changed pose solution or invalid LP certificate."""

import numpy as np
import pytest

from s2a.research.wam.training.cached_endpoint_calibration import fit_opening, fit_pose
from scripts.verification.wam.training.verify_cached_endpoint_calibration import check_fit


def test_independent_fit_certificate_accepts_valid_and_rejects_tampered_parameters():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(80, 7))
    y = rng.normal(size=(80, 6))
    g = np.column_stack((np.ones(80), rng.random((80, 6))))
    opening = rng.random(80)
    truth = np.clip(opening + 0.1, 0, 1)
    config = {"pose_ridge": 0.1, "opening_l1": 0.01}
    coefficients, diagnostics = fit_opening(g, opening, truth, 0.01, 10)
    candidate = dict(
        pose_coefficients=fit_pose(x, y, 0.1).tolist(),
        opening_coefficients=coefficients.tolist(),
        opening_lp=diagnostics,
    )
    data = x, y, g, opening, truth
    assert check_fit(candidate, data, config)["opening_primal_dual_gap"] < 1e-7
    candidate["pose_coefficients"][0][0] += 0.01
    with pytest.raises(ValueError, match="least squares"):
        check_fit(candidate, data, config)
    candidate["pose_coefficients"][0][0] -= 0.01
    candidate["opening_lp"]["inequality_dual"][0] = 1.0
    with pytest.raises(ValueError, match="optimality certificate"):
        check_fit(candidate, data, config)
