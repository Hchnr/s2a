"""Full-rank coupled probes and causal fits within the original command domain."""

import numpy as np

from s2a.geometry import finite

STEP = 1e-4
RADIUS = 0.005
MAX_CONDITION = 10.0


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def prefix_geometry(displacements):
    D = finite(displacements, (72, 72))
    records = []
    for period in range(12):
        design = D[:, : 6 * (period + 1)] / STEP
        rank = int(np.linalg.matrix_rank(design))
        condition = float(np.linalg.cond(design))
        _require(
            rank == design.shape[1] and condition <= MAX_CONDITION,
            "Every causal prefix must be full rank and well conditioned",
        )
        records.append(dict(period=period, rank=rank, condition=condition))
    return records


def select_interval_basis(anchor, previous, input_min, input_max, lower, upper):
    """Choose probes by command geometry only, in a fixed response-blind order."""
    x = finite(anchor, (12, 6))
    previous, lo, hi = (finite(v, (6,)) for v in (previous, input_min, input_max))
    lower, upper = (finite(v, (12, 6)) for v in (lower, upper))
    _require(np.all(lo < hi) and np.all(lower <= upper), "Ordered original command bounds")
    chosen, vectors = [], []
    admitted = 0
    for length in range(1, 13):
        for start in range(13 - length):
            for channel in range(6):
                for sign in (1, -1):
                    nominal = np.zeros((12, 6))
                    nominal[start : start + length, channel] = sign * STEP
                    command = x + nominal
                    step = command - x
                    valid = (
                        np.all(command >= lo - 1e-12)
                        and np.all(command <= hi + 1e-12)
                        and np.all(abs(np.diff(np.vstack((previous, command)), axis=0)) <= 0.25 + 1e-12)
                        and np.all(command >= lower - 1e-12)
                        and np.all(command <= upper + 1e-12)
                        and np.max(abs(step)) <= RADIUS + 1e-12
                        and np.any(step != 0)
                    )
                    if not valid:
                        continue
                    admitted += 1
                    if len(vectors) == 72:
                        continue
                    vector = step.ravel() / STEP
                    if np.linalg.matrix_rank(np.asarray(vectors + [vector])) > len(vectors):
                        vectors.append(vector)
                        chosen.append(
                            dict(
                                length=length,
                                start=start,
                                channel=channel,
                                sign=sign,
                                commands=command,
                                actual_displacement=step.ravel(),
                            )
                        )
    _require(len(chosen) == 72, "No complete admissible interval basis in original command domain")
    D = np.asarray([c["actual_displacement"] for c in chosen])
    return dict(
        chosen=chosen,
        displacements=D,
        prefix_geometry=prefix_geometry(D),
        condition=float(np.linalg.cond(D / STEP)),
        enumerated=936,
        admitted=admitted,
    )


def fit_causal_response(displacements, observed_changes, row_periods):
    """Fit directly in each row's causal parameter space; retain finite-scale residuals.

    This enforces a model structure, not a proof of physical system causality.
    Changes are probe-by-output; each row of the returned Jacobian is an output.
    """
    D = finite(displacements, (72, 72))
    geometry = prefix_geometry(D)
    periods = np.asarray(row_periods)
    _require(
        periods.ndim == 1 and periods.size > 0 and np.isin(periods, np.arange(12)).all(),
        "Explicit valid period for every output row",
    )
    changes = finite(observed_changes, (72, len(periods)))
    jacobian = np.zeros((len(periods), 72))
    residual = np.zeros_like(changes)
    for period in range(12):
        indices = np.flatnonzero(periods == period)
        if not len(indices):
            continue
        end = 6 * (period + 1)
        # Scale both sides together. Future parameters never enter the solve.
        coefficients = np.linalg.lstsq(D[:, :end] / STEP, changes[:, indices] / STEP, rcond=None)[0]
        jacobian[indices, :end] = coefficients.T
        residual[:, indices] = changes[:, indices] - D[:, :end] @ coefficients
    return dict(
        jacobian=jacobian,
        residual=residual,
        maximum_absolute_residual_by_row=np.max(abs(residual), axis=0),
        prefix_geometry=geometry,
    )
