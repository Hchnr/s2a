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


def chunk_opening_diagnostics(actions, *, execution_horizon):
    """Count closures confined to a skipped suffix; full execution has no suffix."""
    if not isinstance(execution_horizon, (int, np.integer)) or execution_horizon < 1:
        raise ValueError("requires a positive integer execution horizon")
    chunks = [np.asarray(chunk, dtype=float) for chunk in actions]
    if not chunks or any(a.ndim != 2 or a.shape[0] == 0 or a.shape[1] < 10 or not np.isfinite(a).all() for a in chunks):
        raise ValueError("requires nonempty finite action chunks with opening slot 9")
    return dict(
        chunks=len(chunks),
        executed_prefix_min=float(min(a[:execution_horizon, 9].min() for a in chunks)),
        full_prediction_min=float(min(a[:, 9].min() for a in chunks)),
        deferred_closure_chunks=sum(
            bool(
                len(a) > execution_horizon
                and a[:execution_horizon, 9].min() >= 0.35
                and a[execution_horizon:, 9].min() < 0.35
            )
            for a in chunks
        ),
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
