"""Fixed054 gates for the separately identified aggregate056 candidate."""

import numpy as np


def require(condition, message):
    if not condition:
        raise ValueError(message)


def numeric(values, message):
    result = np.asarray(values)
    require(result.dtype.kind in "biuf" and np.isfinite(result).all(), message)
    return result


def pool(rows):
    require(bool(rows), "empty state metric group")
    result = {}
    for horizon in (8, 16, 40):
        methods = {}
        for method in ("raw", "aggregate056", "hold", "constant_velocity"):
            selected = [r for r in rows if r["variant"] == (method if method in ("raw", "aggregate056") else "raw")]
            require(bool(selected), "missing raw/W2 metric rows")
            field = "wam" if method in ("raw", "aggregate056") else method
            errors = {}
            for name in ("position", "rotation", "opening"):
                sequences = [numeric(r["errors"][field][name], "nonfinite endpoint errors") for r in selected]
                require(all(a.shape == (40,) for a in sequences), "state scoring must retain every H40 endpoint")
                errors[name] = np.concatenate([a[:horizon] for a in sequences])
            p, r, g = (errors[k] for k in ("position", "rotation", "opening"))
            methods[method] = dict(
                valid_endpoints=len(p),
                position_rmse_m=float(np.sqrt(np.mean(p**2))),
                rotation_rmse_rad=float(np.sqrt(np.mean(r**2))),
                opening_mae=float(np.abs(g).mean()),
                joint_score=float(np.mean((p / 0.01) ** 2 + (r / 0.05) ** 2 + (g / 0.1) ** 2)),
            )
            require(all(np.isfinite(value) for value in methods[method].values()), "state metrics overflow")
        result[str(horizon)] = methods
    return result


def state_gate(groups):
    """Original028 thresholds with the separately identified aggregate056 candidate."""
    pooled = groups["pooled"]
    require(
        len(groups["by_episode"]) == 6 and set(groups["by_seed"]) == {"42", "43"},
        "state gate needs all six episodes and both seeds",
    )
    components = {}
    for horizon in (8, 16, 40):
        raw, w2 = pooled[str(horizon)]["raw"], pooled[str(horizon)]["aggregate056"]
        for name, additive in (("position_rmse_m", 0.002), ("rotation_rmse_rad", 0.01), ("opening_mae", 0.01)):
            components[f"h{horizon}_{name}"] = w2[name] <= 1.05 * raw[name] and w2[name] <= raw[name] + additive
    episodes = {
        episode: all(
            g[str(h)]["aggregate056"]["joint_score"] <= 1.05 * g[str(h)]["raw"]["joint_score"] for h in (8, 16)
        )
        for episode, g in groups["by_episode"].items()
    }
    checks = dict(
        short_joint_improvement=all(
            pooled[str(h)]["aggregate056"]["joint_score"] <= 0.90 * pooled[str(h)]["raw"]["joint_score"]
            for h in (8, 16)
        ),
        strictly_beats_hold_and_constant_velocity=all(
            pooled[str(h)]["aggregate056"]["joint_score"] < pooled[str(h)][b]["joint_score"]
            for h in (8, 16)
            for b in ("hold", "constant_velocity")
        ),
        all_components_protected=all(components.values()),
        at_least_five_episodes=sum(episodes.values()) >= 5,
        no_episode_short_outlier=all(
            g[str(h)]["aggregate056"]["joint_score"] <= 1.25 * g[str(h)]["raw"]["joint_score"]
            for g in groups["by_episode"].values()
            for h in (8, 16)
        ),
        both_seeds_protected=all(
            g[str(h)]["aggregate056"]["joint_score"] <= 1.05 * g[str(h)]["raw"]["joint_score"]
            for g in groups["by_seed"].values()
            for h in (8, 16)
        ),
    )
    return dict(passed=all(checks.values()), checks=checks, component_checks=components, episode_passes=episodes)


def diagnosis(rows):
    require(len(rows) == 96, "Full96 state scoring denominator")
    groups = {"pooled": pool(rows)}
    for label, field in [("by_episode", "episode"), ("by_task", "task_index"), ("by_seed", "sampling_seed")]:
        groups[label] = {str(v): pool([r for r in rows if r[field] == v]) for v in sorted({r[field] for r in rows})}
    require(
        len(groups["by_episode"]) == 6 and set(groups["by_task"]) == {"0", "1", "2"},
        "All six cases and three tasks required",
    )
    return groups, state_gate(groups)
