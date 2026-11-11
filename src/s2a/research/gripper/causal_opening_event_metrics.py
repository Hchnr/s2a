"""Offline grouped event scoring and fixed training-only threshold selection."""

import numpy as np

from s2a.geometry import finite
from s2a.research.gripper.causal_opening_event import apply_opening_event


def summarize(raw, truth, candidate, horizon, *, closed_below):
    raw, truth, candidate = (finite(value) for value in (raw, truth, candidate))
    if (
        raw.ndim != 2
        or raw.shape[1] != 40
        or truth.shape != raw.shape
        or candidate.shape != raw.shape
        or not len(raw)
        or horizon not in (8, 16, 40)
    ):
        raise ValueError("matched nonempty H40 opening sequences required")
    raw, truth, candidate = (value[:, :horizon].ravel() for value in (raw, truth, candidate))
    actual = truth < closed_below
    old, new = raw < closed_below, candidate < closed_below
    negatives = int((~actual).sum())
    if not negatives:
        raise ValueError("event false-positive rate requires observed negatives")
    return dict(
        endpoints=len(actual),
        positive_endpoints=int(actual.sum()),
        negative_endpoints=negatives,
        raw_true_positives=int((actual & old).sum()),
        candidate_true_positives=int((actual & new).sum()),
        added_true_positives=int((actual & ~old & new).sum()),
        raw_false_positives=int((~actual & old).sum()),
        candidate_false_positives=int((~actual & new).sum()),
        added_false_positives=int((~actual & ~old & new).sum()),
        raw_false_positive_rate=float((~actual & old).sum() / negatives),
        candidate_false_positive_rate=float((~actual & new).sum() / negatives),
        raw_mae=float(np.mean(abs(raw - truth))),
        candidate_mae=float(np.mean(abs(candidate - truth))),
    )


def evaluate(raw, truth, probability, tasks, threshold, config):
    raw, truth, probability = (finite(value) for value in (raw, truth, probability))
    tasks = np.asarray(tasks)
    if raw.shape != truth.shape or raw.shape != probability.shape or raw.ndim != 2 or tasks.shape != (len(raw),):
        raise ValueError("event scoring requires one task identity per H40 query")
    if set(tasks.tolist()) != {0, 1, 2}:
        raise ValueError("all three tasks required for the fixed gate")
    candidate = apply_opening_event(
        raw,
        probability,
        threshold,
        closed_below=config["closed_below"],
        commanded_opening=config["commanded_opening"],
    )
    pooled = {str(h): summarize(raw, truth, candidate, h, closed_below=config["closed_below"]) for h in (8, 16, 40)}
    by_task = {
        str(task): {
            str(h): summarize(
                raw[tasks == task],
                truth[tasks == task],
                candidate[tasks == task],
                h,
                closed_below=config["closed_below"],
            )
            for h in (8, 16, 40)
        }
        for task in (0, 1, 2)
    }
    components = {}
    for h in (8, 16, 40):
        item = pooled[str(h)]
        components[f"h{h}_mae"] = (
            item["candidate_mae"] <= config["mae_ratio_limit"] * item["raw_mae"]
            and item["candidate_mae"] <= item["raw_mae"] + config["mae_additive_limit"]
        )
        components[f"h{h}_pooled_fpr"] = (
            item["candidate_false_positive_rate"] <= item["raw_false_positive_rate"] + config["pooled_fpr_increase"]
        )
        if h in (8, 16):
            for task in (0, 1, 2):
                group = by_task[str(task)][str(h)]
                components[f"h{h}_task{task}_fpr"] = (
                    group["candidate_false_positive_rate"]
                    <= group["raw_false_positive_rate"] + config["task_fpr_increase"]
                )
    task_improved = {
        str(task): any(by_task[str(task)][str(h)]["added_true_positives"] > 0 for h in (8, 16)) for task in (0, 1, 2)
    }
    components["h8_added_true_positives"] = (
        pooled["8"]["added_true_positives"] >= config["minimum_h8_added_true_positives"]
    )
    components["tasks_improved"] = sum(task_improved.values()) >= config["minimum_tasks_improved"]
    return dict(
        threshold=float(threshold),
        eligible=all(components.values()),
        components=components,
        tasks_improved=task_improved,
        added_score=sum(pooled[str(h)]["added_true_positives"] for h in (8, 16, 40)),
        pooled=pooled,
        by_task=by_task,
    )


def select_threshold(raw, truth, probability, tasks, config):
    """Select only from registered thresholds and never fit on validation."""
    thresholds = config["thresholds"]
    if thresholds != sorted(set(thresholds)) or not thresholds:
        raise ValueError("ordered unique registered threshold grid required")
    results = [evaluate(raw, truth, probability, tasks, threshold, config) for threshold in thresholds]
    eligible = [result for result in results if result["eligible"]]
    selected = None if not eligible else max(eligible, key=lambda item: (item["added_score"], item["threshold"]))
    return dict(selected_threshold=None if selected is None else selected["threshold"], thresholds=results)
