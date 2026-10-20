"""Task predicates and uncertainty that retain repeated-seed scene dependence."""

import numpy as np


def task_outcome(success, *, expected_steps, stable_steps):
    """Require the complete budget; transient success is not stable completion."""
    values = np.asarray(success)
    if values.dtype != np.bool_ or values.shape != (expected_steps,):
        raise ValueError("requires one boolean predicate for every budgeted step")
    if not 1 <= stable_steps <= expected_steps:
        raise ValueError("invalid stable tail length")
    trailing = 0
    for value in values[::-1]:
        if not value:
            break
        trailing += 1
    return dict(
        ever=bool(values.any()),
        terminal=bool(values[-1]),
        stable=bool(values[-stable_steps:].all()),
        successful_steps=int(values.sum()),
        trailing_success_steps=trailing,
    )


def scene_bootstrap(success, *, seed=42, replicates=20000):
    """Stratify by task; resample initial states with all their seeds together.

    Input is task x initial state x sampling seed, with equal counts per task.
    This small-sample percentile interval is descriptive, not a certification.
    """
    values = np.asarray(success)
    if values.ndim != 3 or min(values.shape) < 1 or values.dtype != np.bool_:
        raise ValueError("requires a complete boolean task x scene x seed grid")
    if replicates < 1:
        raise ValueError("replicates must be positive")
    rng = np.random.default_rng(seed)
    scene_rates = values.mean(axis=2)
    draws = np.zeros(replicates)
    for rates in scene_rates:
        draws += rates[rng.integers(len(rates), size=(replicates, len(rates)))].mean(axis=1)
    draws /= len(scene_rates)
    return dict(
        rate=float(values.mean()),
        scene_count=values.shape[0] * values.shape[1],
        trial_count=int(values.size),
        interval95=np.quantile(draws, [0.025, 0.975]).tolist(),
        method="task-stratified initial-state cluster percentile bootstrap",
        seed=seed,
        replicates=replicates,
    )
