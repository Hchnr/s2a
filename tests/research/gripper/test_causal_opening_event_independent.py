"""Independent event math rejects changed features, coefficients and decisions."""

import copy

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from s2a.geometry import transform
from s2a.research.gripper.causal_opening_event import event_features, fit_event_model
from s2a.research.gripper.causal_opening_event_metrics import select_threshold
from scripts.verification.wam.training.verify_causal_opening_event import (
    independent_features,
    verify_fit,
    verify_selection,
)
from tests.research.gripper.test_causal_opening_event_metrics import config, data


def test_independent_features_match_nonidentity_base_frame():
    anchor = transform(np.array([0.2, -0.3, 0.4]), Rotation.from_euler("xyz", [0.2, -0.4, 0.5]).as_matrix())
    poses = np.repeat(anchor[None], 40, axis=0)
    poses[:, :3, 3] += np.arange(1, 41)[:, None] * [0.001, -0.002, 0.003]
    opening = np.linspace(1, 0, 40)
    np.testing.assert_allclose(
        event_features(anchor, 0.8, poses, opening),
        independent_features(anchor, 0.8, poses, opening),
        atol=1e-12,
        rtol=0,
    )


def test_independent_gradient_rejects_tampered_coefficients():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(200, 44))
    x[:, 0] = 1
    y = (x[:, 1] > 0).astype(int)
    cfg = {"positive_weight": 2.0, "ridge": 0.003, "gradient_tolerance": 1e-6}
    weights, reported = fit_event_model(x, y, **cfg, max_iterations=400)
    assert verify_fit(x, y, weights, cfg, reported)["max_abs_gradient"] <= 1e-6
    weights[1] += 0.1
    with pytest.raises(ValueError, match="logistic optimum"):
        verify_fit(x, y, weights, cfg, reported)


def test_independent_gate_rejects_tampered_selection():
    raw, truth, probability, tasks = data()
    selection = select_threshold(raw, truth, probability, tasks, config())
    assert verify_selection(raw, truth, probability, tasks, config(), selection)["selected_threshold"] == 0.9
    modified = copy.deepcopy(selection)
    modified["selected_threshold"] = 0.5
    with pytest.raises(ValueError, match="selection differs"):
        verify_selection(raw, truth, probability, tasks, config(), modified)
