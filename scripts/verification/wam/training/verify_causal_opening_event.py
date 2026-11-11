"""Independent feature, optimization and threshold checks for opening events."""

import numpy as np
from scipy.special import expit


def independent_features(anchor, current_opening, poses, opening):
    anchor, poses, opening = (np.asarray(value, dtype=float) for value in (anchor, poses, opening))
    if anchor.shape != (4, 4) or poses.shape != (40, 4, 4) or opening.shape != (40,):
        raise ValueError("independent event feature shapes differ")
    if not np.isfinite(anchor).all() or not np.isfinite(poses).all() or not np.isfinite(opening).all():
        raise ValueError("independent event feature inputs must be finite")
    opening = np.clip(opening, 0, 1)
    base = anchor[:3, :3]
    position = poses[:, :3, 3]
    current = float(current_opening)
    if not np.isfinite(current) or not 0 <= current <= 1:
        raise ValueError("independent measured opening differs")
    elapsed = np.arange(1, 41) / 40
    local = np.einsum("ij,tj->ti", base.T, position - anchor[:3, 3]) / 0.2
    before = np.concatenate((anchor[None, :3, 3], position[:-1]))
    velocity = np.einsum("ij,tj->ti", base.T, position - before) / 0.05
    remaining = np.einsum("ij,tj->ti", base.T, position[-1] - position) / 0.2
    minimum, final, span = opening.min(), opening[-1], opening.max() - opening.min()
    rows = []
    for i in range(40):
        z, raw, tau = position[i, 2] / 0.5, opening[i], elapsed[i]
        extra = [
            raw**2,
            current**2,
            raw * current,
            tau * raw,
            tau * current,
            z * raw,
            z**2,
            tau * local[i, 2],
            float(np.dot(local[i], local[i])),
            minimum * raw,
        ]
        rows.append(
            np.r_[
                1,
                anchor[:3, 3] / 0.5,
                base[:, 0],
                base[:, 1],
                position[i] / 0.5,
                poses[i, :3, 0],
                poses[i, :3, 1],
                local[i],
                velocity[i],
                raw,
                current,
                tau,
                minimum,
                final,
                span,
                remaining[i],
                extra,
            ]
        )
    result = np.asarray(rows)
    if result.shape != (40, 44) or not np.isfinite(result).all():
        raise ValueError("independent event feature reconstruction differs")
    return result


def verify_fit(features, closed, coefficients, config, reported):
    x, y, theta = (np.asarray(value, dtype=float) for value in (features, closed, coefficients))
    if x.ndim != 2 or x.shape[1] != 44 or y.shape != (len(x),) or theta.shape != (44,):
        raise ValueError("independent logistic fit shapes differ")
    weights = np.where(y == 1, config["positive_weight"], 1.0)

    def loss(t):
        z = x @ t
        return float(
            np.sum(weights * (np.logaddexp(0, z) - y * z)) / np.sum(weights)
            + 0.5 * config["ridge"] * np.sum(t[1:] ** 2)
        )

    probability = expit(x @ theta)
    gradient = np.sum((weights * (probability - y))[:, None] * x, axis=0) / np.sum(weights)
    gradient[1:] += config["ridge"] * theta[1:]
    maximum = float(np.max(abs(gradient)))
    checks = []
    for index in (0, 1, 11, 43):
        step = np.zeros(44)
        step[index] = 1e-6
        checks.append(abs(gradient[index] - (loss(theta + step) - loss(theta - step)) / 2e-6))
    difference = abs(loss(theta) - reported["loss"])
    if (
        not np.isfinite(gradient).all()
        or maximum > config["gradient_tolerance"]
        or abs(maximum - reported["max_abs_gradient"]) > 1e-10
        or difference > 1e-10
        or max(checks) > 1e-8
    ):
        raise ValueError("independent logistic optimum or numerical gradient differs")
    return dict(loss_difference=difference, max_abs_gradient=maximum, max_finite_difference_error=max(checks))


def verify_selection(raw, truth, probability, tasks, config, selection):
    """Recompute the registered gate using scalar counts and MAE, not producer metrics."""
    raw, truth, probability = (np.asarray(value, dtype=float) for value in (raw, truth, probability))
    tasks = np.asarray(tasks)
    if raw.shape != truth.shape or raw.shape != probability.shape or raw.ndim != 2 or raw.shape[1] != 40:
        raise ValueError("independent event score shapes differ")
    rows = []
    for threshold in config["thresholds"]:
        output = np.where(
            (probability >= threshold) & (raw >= config["closed_below"]), config["commanded_opening"], raw
        )
        pooled, groups, components = {}, {}, {}
        for task in (0, 1, 2):
            groups[str(task)] = {}
        for horizon in (8, 16, 40):

            def stats(mask):
                target, before, after = truth[mask, :horizon], raw[mask, :horizon], output[mask, :horizon]
                closed, old, new = (
                    target < config["closed_below"],
                    before < config["closed_below"],
                    after < config["closed_below"],
                )
                negatives = int(np.count_nonzero(~closed))
                if not negatives:
                    raise ValueError("independent FPR has no negative samples")
                return dict(
                    raw_mae=float(np.mean(abs(before - target))),
                    candidate_mae=float(np.mean(abs(after - target))),
                    raw_fpr=int(np.count_nonzero(~closed & old)) / negatives,
                    candidate_fpr=int(np.count_nonzero(~closed & new)) / negatives,
                    added_tp=int(np.count_nonzero(closed & ~old & new)),
                    added_fp=int(np.count_nonzero(~closed & ~old & new)),
                )

            item = pooled[str(horizon)] = stats(np.ones(len(raw), dtype=bool))
            components[f"h{horizon}_mae"] = (
                item["candidate_mae"] <= config["mae_ratio_limit"] * item["raw_mae"]
                and item["candidate_mae"] <= item["raw_mae"] + config["mae_additive_limit"]
            )
            components[f"h{horizon}_pooled_fpr"] = (
                item["candidate_fpr"] <= item["raw_fpr"] + config["pooled_fpr_increase"]
            )
            for task in (0, 1, 2):
                group = groups[str(task)][str(horizon)] = stats(tasks == task)
                if horizon in (8, 16):
                    components[f"h{horizon}_task{task}_fpr"] = (
                        group["candidate_fpr"] <= group["raw_fpr"] + config["task_fpr_increase"]
                    )
        improved = {str(task): any(groups[str(task)][str(h)]["added_tp"] > 0 for h in (8, 16)) for task in (0, 1, 2)}
        components["h8_added_true_positives"] = pooled["8"]["added_tp"] >= config["minimum_h8_added_true_positives"]
        components["tasks_improved"] = sum(improved.values()) >= config["minimum_tasks_improved"]
        rows.append(
            dict(
                threshold=float(threshold),
                eligible=all(components.values()),
                components=components,
                tasks_improved=improved,
                added_score=sum(pooled[str(h)]["added_tp"] for h in (8, 16, 40)),
                pooled=pooled,
                by_task=groups,
            )
        )
    if len(rows) != len(selection["thresholds"]):
        raise ValueError("independent threshold count differs")
    maximum = 0.0
    for checked, recorded in zip(rows, selection["thresholds"], strict=True):
        if any(
            checked[k] != recorded[k] for k in ("threshold", "eligible", "components", "tasks_improved", "added_score")
        ):
            raise ValueError("independent threshold gate differs")
        for location in ("pooled", "by_task"):
            for task in [None] if location == "pooled" else ["0", "1", "2"]:
                for horizon in ("8", "16", "40"):
                    left = checked[location][horizon] if task is None else checked[location][task][horizon]
                    right = recorded[location][horizon] if task is None else recorded[location][task][horizon]
                    for a, b in (
                        ("raw_mae", "raw_mae"),
                        ("candidate_mae", "candidate_mae"),
                        ("raw_fpr", "raw_false_positive_rate"),
                        ("candidate_fpr", "candidate_false_positive_rate"),
                        ("added_tp", "added_true_positives"),
                        ("added_fp", "added_false_positives"),
                    ):
                        maximum = max(maximum, abs(left[a] - right[b]))
    eligible = [row for row in rows if row["eligible"]]
    expected = (
        None if not eligible else max(eligible, key=lambda row: (row["added_score"], row["threshold"]))["threshold"]
    )
    if expected != selection["selected_threshold"] or maximum > 1e-12:
        raise ValueError("independent threshold selection differs")
    return dict(maximum_metric_difference=maximum, selected_threshold=expected, thresholds=len(rows))
