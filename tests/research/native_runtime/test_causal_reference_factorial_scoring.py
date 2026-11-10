"""Selected and original reference errors must remain separately labeled."""

import copy

import numpy as np
import pytest

from s2a.geometry import pose_error
from scripts.simulation.runtime import evaluate_causal_reference_factorial as scoring
from tests.research.native_runtime.test_causal_event_bridge_scoring import fixture


def test_scores_all_four_references_without_relabeling_selected_as_bridge(monkeypatch):
    case, bridge = fixture(monkeypatch)
    selected = copy.deepcopy(bridge)
    selected["poses"][8, 0, 3] = 0.12
    selected["openings"][8] = 0.2
    error = pose_error(selected["poses"][8], np.eye(4))
    result = scoring.score_completed(None, case, selected, bridge, 8, error)
    assert set(result["reference_errors"]) == {"latest", "remembered", "bridge", "selected"}
    np.testing.assert_allclose(result["reference_errors"]["bridge"][:3], [0, 0, 0.08])
    np.testing.assert_allclose(result["reference_errors"]["selected"][:3], [0.12, 0, 0.08])
    assert result["opening_errors"]["bridge"] == pytest.approx(-0.1)
    assert result["opening_errors"]["selected"] == pytest.approx(0.2)
    scoring.score_completed(None, case, selected, bridge, 8, error, expected=copy.deepcopy(result))
    result["opening_targets"]["bridge"] += 0.1
    with pytest.raises(AssertionError):
        scoring.score_completed(None, case, selected, bridge, 8, error, expected=result)
