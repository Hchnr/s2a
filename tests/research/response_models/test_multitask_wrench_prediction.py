"""Validate causal alignment and training-only scaling for wrench correction."""

import copy

import numpy as np

from scripts.tracking.response_models.evaluate_multitask_wrench import fit_ridge, predict_ridge, prepare_periods


def test_features_use_current_wrench_and_only_completed_prediction_error():
    trace = {
        "rows": [
            {
                "endpoint_step": step,
                "sim_time": float(step),
                "sim_state_sha256": str(step),
                "wrench": {"world_at_control_site": [step, 0, 0, 0, 0, 0]},
            }
            for step in range(4)
        ]
    }
    periods = [
        {
            "step": step,
            "decision": {"quality": {"first_tcp_error": [0, 0, 0, 0, 0, 0], "verification": {"passed": True}}},
            "tcp_error": [step + 1, 0, 0, 0, 0, 0],
            "endpoint_receipt": {
                "endpoint_step": step + 1,
                "sim_time": float(step + 1),
                "sim_state_sha256": str(step + 1),
            },
        }
        for step in range(3)
    ]
    rows = prepare_periods(periods, trace)
    np.testing.assert_array_equal(rows[0]["features"], [1, 0, 0, 1, 0, 0, 0, 0, 0])
    changed = copy.deepcopy(periods)
    changed[1]["tcp_error"][0] = 10
    updated = prepare_periods(changed, trace)
    np.testing.assert_array_equal(updated[0]["features"], rows[0]["features"])
    assert updated[1]["features"][0] == 10


def test_ridge_standardization_is_fixed_by_training_rows():
    features = np.array([[0.0, 10.0], [2.0, 10.0]])
    target = np.array([[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    model = fit_ridge(features, target)
    np.testing.assert_allclose(model["mean"], [1, 10])
    np.testing.assert_allclose(model["scale"], [1, 1])
    before = predict_ridge(model, [3, 10])
    _ = predict_ridge(model, [1000, -1000])
    np.testing.assert_array_equal(predict_ridge(model, [3, 10]), before)
