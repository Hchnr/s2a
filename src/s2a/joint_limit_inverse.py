"""Bounded one-period refinement candidate around an existing causal arm command."""

import numpy as np
from scipy.optimize import least_squares

from s2a.geometry import finite
from s2a.joint_limit_response import predict_joint_response


def refine_command(model, velocity, seed, previous, interface, joint_lower, joint_upper, target_response):
    """Refine only when the seed's free joint response crosses a joint boundary.

    The caller supplies a causal 12D world-tangent pose/twist target. This is a
    one-period alternative to the original four-period preview, not a proof of
    equivalent multi-period optimality. No object state or future label enters.
    """
    seed, previous = finite(seed, (6,)), finite(previous, (6,))
    target = finite(target_response, (12,))
    if not np.isclose(model["dt"] * 25, 0.05, atol=1e-12, rtol=0):
        raise ValueError("Fixed25 substeps at2ms required")
    lo = np.maximum(interface.input_min, previous - 0.25)
    hi = np.minimum(interface.input_max, previous + 0.25)
    if np.any(lo >= hi) or np.any(seed < lo) or np.any(seed > hi):
        raise ValueError("Feasible seed within fixed command box/rate required")
    joint_lower = finite(joint_lower, np.asarray(model["q"]).shape)
    joint_upper = finite(joint_upper, np.asarray(model["q"]).shape)
    free = predict_joint_response(
        model, velocity, interface.decode(seed), joint_lower, joint_upper, substeps=25, enforce_limits=False
    )
    at_risk = (
        np.any(model["q"] < joint_lower)
        or np.any(model["q"] > joint_upper)
        or np.any(free["joint_position"] < joint_lower)
        or np.any(free["joint_position"] > joint_upper)
    )
    if not at_risk:
        return dict(command=seed.copy(), route="unchanged_free_feasible", evaluations=0)
    weights = np.array([100.0] * 3 + [20.0] * 3 + [5.0] * 3 + [1.0] * 3)
    evaluations = 0

    def residual(command):
        nonlocal evaluations
        evaluations += 1
        if evaluations > 300:
            raise RuntimeError("Fixed candidate response-evaluation budget exhausted")
        prediction = predict_joint_response(
            model, velocity, interface.decode(command), joint_lower, joint_upper, substeps=25, enforce_limits=True
        )
        response = np.r_[prediction["task_displacement"], prediction["task_velocity"]]
        return weights * (response - target)

    # Finite differences on an outward-impact plateau can be identically zero.
    # Include feasible previous/corner probes, using the same objective for all.
    probes = [seed, np.clip(previous, lo, hi), lo, hi]
    residuals = [residual(p) for p in probes]
    scores = [float(r @ r) for r in residuals]
    best = min(range(len(probes)), key=lambda i: (scores[i], -i))
    result = least_squares(
        residual, probes[best], bounds=(lo, hi), jac="3-point", max_nfev=20, ftol=1e-8, xtol=1e-8, gtol=1e-8
    )
    command = probes[best].copy()
    score = scores[best]
    proposed = np.clip(result.x, lo, hi)
    proposed_residual = residual(proposed)
    proposed_score = float(proposed_residual @ proposed_residual)
    if result.success and proposed_score < score:
        command, score = proposed, proposed_score
    if score >= scores[0] - 1e-12:
        command, score = seed.copy(), scores[0]
    if np.any(command < lo) or np.any(command > hi) or not np.isfinite(command).all():
        raise RuntimeError("Candidate violates command feasibility")
    return dict(
        command=command,
        route="joint_limit_refinement",
        evaluations=evaluations,
        seed_objective=scores[0],
        candidate_objective=score,
        optimizer_success=bool(result.success),
        scope="ideal one-period model objective only",
    )
