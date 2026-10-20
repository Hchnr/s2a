import importlib
from pathlib import Path

import numpy as np


def test_pooled_opening_mae_uses_absolute_errors_and_matching_prefix(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    module = importlib.import_module("diagnose_reference_gains")
    cases = [
        {
            "methods": {
                "raw": {
                    "position_errors_m": [0.03, 0.04],
                    "rotation_errors_rad": [0.1, 0.3],
                    "opening_errors": [-0.2, 0.2],
                }
            }
        },
        {"methods": {"raw": {"position_errors_m": [0], "rotation_errors_rad": [0.2], "opening_errors": [-0.5]}}},
    ]
    scored = module.pooled(cases, [1, 2])
    assert scored["1"]["raw"]["endpoints"] == 2
    np.testing.assert_allclose(scored["1"]["raw"]["opening_mae"], 0.35)
    np.testing.assert_allclose(scored["2"]["raw"]["opening_mae"], 0.3)
    np.testing.assert_allclose(scored["2"]["raw"]["position_rmse_m"], np.sqrt((0.03**2 + 0.04**2) / 3))
    np.testing.assert_allclose(scored["2"]["raw"]["rotation_rmse_rad"], np.sqrt((0.1**2 + 0.3**2 + 0.2**2) / 3))
