"""Exact TCP pose safeguards for the separate joint-limit model inverse candidate."""

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from s2a.geometry import finite, pose_error, transform, validate_transform


def predicted_tcp(site, site_from_tcp, response, innovation):
    """Compose world SO(3) increments before applying the fixed TCP transform."""
    site, site_from_tcp = validate_transform(site), validate_transform(site_from_tcp)
    response, innovation = finite(response, (12,)), finite(innovation, (12,))
    position = site[:3, 3] + response[:3] + innovation[:3]
    rotation = (
        Rotation.from_rotvec(innovation[3:6]) * Rotation.from_rotvec(response[3:6]) * Rotation.from_matrix(site[:3, :3])
    ).as_matrix()
    return transform(position, rotation) @ site_from_tcp


def guard_command(seed, proposal, lower, upper, response_fn, score_fn, site, site_from_tcp, desired_tcp, innovation):
    """Search with the original score, constrained by seed TCP position/rotation.

    Feasibility is rechecked on the actual returned point, independent of solver
    success. Seed fallback preserves a feasible reference; no global optimality
    or real-robot error guarantee follows from these model constraints.
    """
    seed, proposal = finite(seed, (6,)), finite(proposal, (6,))
    lower, upper = finite(lower, (6,)), finite(upper, (6,))
    desired_tcp = validate_transform(desired_tcp)
    if np.any(lower >= upper) or any(np.any(p < lower) or np.any(p > upper) for p in (seed, proposal)):
        raise ValueError("Feasible seed/proposal and strict command bounds required")
    cache = {}

    def evaluate(command):
        command = np.asarray(command, dtype=float)
        key = tuple(command)
        if key not in cache:
            if len(cache) >= 300:
                raise RuntimeError("Pose-guard response evaluation budget exhausted")
            response = finite(response_fn(command), (12,))
            error = pose_error(desired_tcp, predicted_tcp(site, site_from_tcp, response, innovation))
            score = float(score_fn(response))
            if not np.isfinite(score):
                raise ValueError("Finite original model score required")
            cache[key] = (score, error, response)
        return cache[key]

    seed_score, seed_error, _ = evaluate(seed)
    seed_limits = np.array([np.dot(seed_error[:3], seed_error[:3]), np.dot(seed_error[3:], seed_error[3:])])
    # Scaling only conditions the constraints; it does not change the feasible set.
    scales = np.array([100.0**2, 20.0**2])

    def constraint(command):
        error = evaluate(command)[1]
        return (seed_limits - [np.dot(error[:3], error[:3]), np.dot(error[3:], error[3:])]) * scales

    def feasible(command):
        if np.any(command < lower) or np.any(command > upper):
            return False
        error = evaluate(command)[1]
        return bool(
            np.linalg.norm(error[:3]) <= np.linalg.norm(seed_error[:3]) + 1e-10
            and np.linalg.norm(error[3:]) <= np.linalg.norm(seed_error[3:]) + 1e-10
        )

    proposal_feasible = feasible(proposal)
    start = proposal if proposal_feasible and evaluate(proposal)[0] < seed_score else seed
    result = minimize(
        lambda c: evaluate(c)[0],
        start,
        method="SLSQP",
        bounds=list(zip(lower, upper, strict=True)),
        constraints=[dict(type="ineq", fun=constraint)],
        options=dict(maxiter=20, ftol=1e-9, eps=1e-5),
    )
    candidates = [seed]
    if proposal_feasible:
        candidates.append(proposal)
    candidate = np.clip(result.x, lower, upper)
    if feasible(candidate):
        candidates.append(candidate)
    command = min(candidates, key=lambda c: evaluate(c)[0]).copy()
    score, error, response = evaluate(command)
    return dict(
        command=command,
        nominal_response=response,
        seed_tcp_error=seed_error,
        chosen_tcp_error=error,
        seed_objective=seed_score,
        candidate_objective=score,
        proposal_pose_feasible=proposal_feasible,
        optimizer_success=bool(result.success),
        optimizer_message=str(result.message),
        response_evaluations=len(cache),
        route="pose_guard_retained_seed" if np.array_equal(command, seed) else "pose_guard_refined",
        scope="model TCP no-worse pose constraints; no physical qualification",
    )
