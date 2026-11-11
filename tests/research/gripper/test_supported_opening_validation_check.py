"""Independent checks for reference integration, support and grouped scores."""

import copy

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from s2a.geometry import matrix_to_rot6d, pose9_to_transform
from s2a.research.gripper.causal_opening_event_metrics import summarize
from scripts.verification.wam.training import verify_supported_opening_validation as audit


def test_independent_reference_integration_obeys_local_se3_and_column_rot6d():
    anchor_rotation = Rotation.from_euler("z", 0.4).as_matrix()
    anchor = np.r_[[0.3, -0.1, 0.5], matrix_to_rot6d(anchor_rotation)]
    delta_rotation = Rotation.from_euler("y", -0.2).as_matrix()
    action = np.zeros((40, 128))
    action[:, :9] = np.r_[[0.001, 0.002, -0.001], matrix_to_rot6d(delta_rotation)]
    action[:, 9] = 0.7
    measured, poses, opening = audit.independent_targets(anchor, action)
    expected = pose9_to_transform(anchor)
    for step in range(40):
        expected = expected @ pose9_to_transform(action[step, :9])
        np.testing.assert_allclose(poses[step], expected, rtol=0, atol=1e-12)
    np.testing.assert_allclose(measured, pose9_to_transform(anchor), rtol=0, atol=1e-12)
    np.testing.assert_array_equal(opening, np.full(40, 0.7))
    action[0, 10] = 1
    with pytest.raises(ValueError, match="Action128"):
        audit.independent_targets(anchor, action)


def test_independent_opening_uses_only_supported_future_and_fixed_threshold():
    raw, probability = np.full((2, 40), 0.8), np.full((2, 40), 0.8)
    raw[0, 16] = 0.2
    config = dict(closed_below=0.35, commanded_opening=0.1, support_future_steps=16, threshold=0.7)
    support, output = audit.independent_opening(raw, probability, config)
    assert support[0, 0] and not support[0, 16] and not support[1].any()
    assert output[0, 0] == 0.1 and output[0, 16] == 0.2 and output[1, 0] == 0.8


def test_scalar_metrics_match_and_reject_tampered_counts():
    rng = np.random.default_rng(42)
    raw, truth, candidate = (rng.uniform(size=(4, 40)) for _ in range(3))
    checked = audit.independent_summary(raw, truth, candidate, range(4), 16, 0.35)
    recorded = summarize(raw, truth, candidate, 16, closed_below=0.35)
    assert audit.compare_summary(checked, recorded) <= 1e-12
    changed = copy.deepcopy(recorded)
    changed["added_false_positives"] += 1
    with pytest.raises(ValueError, match="summary differs"):
        audit.compare_summary(checked, changed)
