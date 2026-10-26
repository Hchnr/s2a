"""202 minimal translation change bound with a finite piecewise-affine guard."""

import numpy as np


def guarded_rate_command(
    matrix, offset, target, previous, command, *, error_bound=0.005, min_change=0.125, max_change=0.25
):
    matrix = np.asarray(matrix, dtype=float)
    vectors = [np.asarray(v, dtype=float) for v in (offset, target, previous, command)]
    if (
        matrix.shape != (3, 3)
        or not np.isfinite(matrix).all()
        or any(v.shape != (3,) or not np.isfinite(v).all() for v in vectors)
    ):
        raise ValueError("Finite3x3 normalized-command position map and finite3-vectors required")
    offset, target, previous, command = vectors
    if (
        not all(np.isfinite(v) for v in (error_bound, min_change, max_change))
        or error_bound <= 0
        or not 0 < min_change <= max_change
    ):
        raise ValueError("Positive finite position bound and ordered command changes required")
    delta = command - previous
    if np.max(np.abs(delta)) > max_change + 1e-12:
        raise ValueError("Original command must satisfy original change bound")
    breaks = sorted(set([min_change, max_change, *[float(v) for v in np.abs(delta) if min_change < v < max_change]]))

    def candidate(change):
        return previous + np.clip(delta, -change, change)

    def error(change):
        return matrix @ candidate(change) + offset - target

    limit = error_bound**2

    def output(change, feasible):
        value = candidate(change) if feasible else command.copy()
        return dict(
            command=value,
            change=float(change),
            tracking_guard_feasible=feasible,
            predicted_error=float(np.linalg.norm(matrix @ value + offset - target)),
        )

    if float(error(min_change) @ error(min_change)) <= limit:
        return output(min_change, True)
    for left, right in zip(breaks[:-1], breaks[1:], strict=True):
        if right <= left:
            continue
        u = error(left)
        v = error(right) - u
        a = float(v @ v)
        b = 2 * float(u @ v)
        c = float(u @ u) - limit
        if c <= 0:
            return output(left, True)
        if a == 0 or b >= 0:
            continue
        minimizer = float(np.clip(-b / (2 * a), 0, 1))
        closest = u + minimizer * v
        if float(closest @ closest) > limit:
            continue
        discriminant = max(0.0, b * b - 4 * a * c)
        denominator = -b + float(np.sqrt(discriminant))
        if denominator <= 0:
            continue
        fraction = 2 * c / denominator
        if not -1e-12 <= fraction <= 1 + 1e-12:
            continue
        change = left + float(np.clip(fraction, 0, 1)) * (right - left)
        result = output(change, True)
        if result["predicted_error"] > error_bound + 1e-12:
            raise RuntimeError("Analytic tracking guard root violates position constraint")
        return result
    if np.linalg.norm(error(max_change)) <= error_bound:
        return output(max_change, True)
    return output(max_change, False)
