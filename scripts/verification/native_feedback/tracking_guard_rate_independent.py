"""202 independent polynomial feasible-point enumeration, no production solver."""

import numpy as np


def verify_guard(matrix, offset, target, previous, command, result):
    matrix, offset, target, previous, command = (
        np.asarray(v, dtype=float) for v in (matrix, offset, target, previous, command)
    )
    delta = command - previous
    boundaries = sorted(set([0.125, 0.25, *[float(v) for v in np.abs(delta) if 0.125 < v < 0.25]]))
    feasible = []
    for lo, hi in zip(boundaries[:-1], boundaries[1:], strict=True):
        middle = (lo + hi) / 2
        slope = np.sign(delta) * (np.abs(delta) > middle)
        intercept = previous + np.where(np.abs(delta) > middle, 0, delta)
        a = matrix @ slope
        b = offset + matrix @ intercept - target
        polynomial = [float(b @ b) - 0.005**2, 2 * float(a @ b), float(a @ a)]
        roots = np.polynomial.polynomial.polyroots(polynomial) if np.any(a) else []
        candidates = [
            lo,
            hi,
            *[float(r.real) for r in roots if abs(r.imag) < 1e-9 and lo - 1e-12 <= r.real <= hi + 1e-12],
        ]
        for rate in candidates:
            rate = min(hi, max(lo, rate))
            if np.linalg.norm(matrix @ (previous + np.clip(delta, -rate, rate)) + offset - target) <= 0.005 + 1e-12:
                feasible.append(rate)
    assert bool(feasible) == result["tracking_guard_feasible"]
    expected = min(feasible) if feasible else 0.25
    np.testing.assert_allclose(result["change"], expected, atol=1e-9, rtol=0)
    value = previous + np.minimum(np.maximum(delta, -expected), expected) if feasible else command
    np.testing.assert_allclose(result["command"], value, atol=1e-9, rtol=0)
    actual_error = float(np.linalg.norm(matrix @ np.asarray(result["command"]) + offset - target))
    np.testing.assert_allclose(result["predicted_error"], actual_error, atol=1e-12, rtol=0)
    if feasible:
        assert actual_error <= 0.005 + 1e-12
    return dict(minimal_change=expected, feasible=bool(feasible), predicted_error=actual_error)
