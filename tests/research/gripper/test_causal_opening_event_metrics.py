"""The training-only selector must honor miss, false-close and MAE constraints."""

import numpy as np

from s2a.research.gripper.causal_opening_event_metrics import evaluate, select_threshold


def config():
    return dict(
        closed_below=0.35,
        commanded_opening=0.1,
        mae_ratio_limit=1.05,
        mae_additive_limit=0.01,
        pooled_fpr_increase=0.01,
        task_fpr_increase=0.02,
        minimum_h8_added_true_positives=5,
        minimum_tasks_improved=2,
        thresholds=[0.5, 0.9],
    )


def data():
    raw = np.full((9, 40), 0.8)
    truth = raw.copy()
    truth[0, :3] = truth[3, :3] = 0.1
    probability = np.zeros_like(raw)
    probability[0, :3] = probability[3, :3] = 0.95
    return raw, truth, probability, np.repeat([0, 1, 2], 3)


def test_selects_highest_tied_threshold_without_false_closures():
    raw, truth, probability, tasks = data()
    selected = select_threshold(raw, truth, probability, tasks, config())
    assert selected["selected_threshold"] == 0.9
    result = evaluate(raw, truth, probability, tasks, 0.9, config())
    assert result["eligible"]
    assert result["pooled"]["8"]["added_true_positives"] == 6
    assert result["pooled"]["8"]["added_false_positives"] == 0
    assert result["tasks_improved"] == {"0": True, "1": True, "2": False}


def test_rejects_high_false_closure_rate_instead_of_selecting_a_candidate():
    raw, truth, probability, tasks = data()
    probability[:] = 1.0
    result = select_threshold(raw, truth, probability, tasks, config())
    assert result["selected_threshold"] is None
    assert not any(item["eligible"] for item in result["thresholds"])
