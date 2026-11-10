"""Independently audit support certificates and a scalar first-step response."""

import numpy as np
from scipy.spatial.transform import Rotation

from s2a.geometry import finite, pose_error, validate_transform
from s2a.research.response_models.forward.joint_limit_response import predict_joint_response


def check_support_certificate(direction, scope, certificate, *, tolerance=1e-9):
    """Recompute a box-corrected dual upper bound from retained sparse duals."""
    lower, upper, seed = (finite(scope[key]) for key in ("lower", "upper", "seed"))
    direction = finite(direction, lower.shape)
    a, b = finite(scope["a"]), finite(scope["b"])
    primal = finite(certificate["primal"], lower.shape)
    if a.shape != (len(b), len(lower)) or np.any(lower > upper):
        raise ValueError("Matching finite inequality and box dimensions required")
    violation = max(0.0, float(np.max(lower - primal)), float(np.max(primal - upper)))
    if len(a):
        violation = max(violation, float(np.max(a @ primal - b)))
        indices = np.asarray(certificate["dual_indices"], dtype=int)
        values = finite(certificate["dual_values"])
        if (
            indices.ndim != 1
            or values.shape != indices.shape
            or np.any(indices < 0)
            or np.any(indices >= len(a))
            or len(set(indices.tolist())) != len(indices)
            or np.any(values < 0)
        ):
            raise ValueError("Nonnegative sparse dual multipliers must have unique row indices")
        dual = np.zeros(len(a))
        dual[indices] = values
        residual = direction - a.T @ dual
        np.testing.assert_allclose(residual, certificate["dual_residual"], atol=tolerance, rtol=0)
        upper_bound = float(b @ dual + np.maximum(residual * lower, residual * upper).sum() - direction @ seed)
    else:
        upper_bound = float(np.maximum(direction * (lower - seed), direction * (upper - seed)).sum())
    primal_support = float(direction @ (primal - seed))
    if violation > 1e-7 or certificate["support_upper"] + tolerance < upper_bound:
        raise ValueError("Support certificate does not bound its feasible linear domain")
    if certificate["support_upper"] + tolerance < primal_support:
        raise ValueError("Support upper bound is below its retained primal")
    np.testing.assert_allclose(certificate["primal_support"], primal_support, atol=tolerance, rtol=0)
    return dict(verified=True, maximum_primal_violation=violation, recomputed_support_upper=upper_bound)


def check_candidate(
    model,
    velocity,
    interface,
    site,
    tool,
    target,
    lower,
    upper,
    seed_joint_position,
    joint_command_jacobian,
    candidate,
    previous_error,
    linearization,
    scopes,
    bounds,
    *,
    tolerance=1e-9,
):
    """Check one saved attempt with scalar dynamics and every dual certificate."""
    site, tool, target = (validate_transform(value) for value in (site, tool, target))
    candidate = finite(candidate, (8, 6))
    previous_error = finite(previous_error, (6,))
    q0 = finite(model["q"], (7,))
    jacobian = finite(model["jacobian"], (6, 7))
    velocity, lower, upper = (finite(value, (7,)) for value in (velocity, lower, upper))
    seed_joint_position = finite(seed_joint_position, (7,))
    joint_command_jacobian = finite(joint_command_jacobian, (7, 6))
    response = predict_joint_response(
        model, velocity, interface.decode(candidate[0]), lower, upper, substeps=25, enforce_limits=False
    )
    affine_q = seed_joint_position + joint_command_jacobian @ (candidate[0] - linearization["seed_command"])
    actual_q = finite(response["joint_position"], (7,))
    max_q_error = float(np.max(abs(actual_q - affine_q)))
    if max_q_error > tolerance:
        raise ValueError("Independent scalar joint response differs from the affine prediction")
    movement = jacobian @ (actual_q - q0)
    predicted_site = site.copy()
    predicted_site[:3, 3] += movement[:3]
    predicted_site[:3, :3] = Rotation.from_rotvec(movement[3:]).as_matrix() @ site[:3, :3]
    tcp = predicted_site @ tool
    actual_error = pose_error(target, tcp)
    np.testing.assert_allclose(actual_error, previous_error, atol=tolerance, rtol=0)
    n, d = finite(bounds["position_direction"], (3,)), finite(bounds["rotation_direction"], (3, 3))
    rotation_cap = float(bounds["rotation_chord_cap"])
    p = float(n @ actual_error[:3] / 0.01)
    r = float(np.sum(d * (target[:3, :3] - tcp[:3, :3])) / rotation_cap)
    p_jac = n @ linearization["position_jacobian"] / 0.01
    r_jac = np.einsum("ij,ijk->k", d, linearization["rotation_jacobian"]) / rotation_cap
    verified, membership = 0, {}
    for name, scope in scopes.items():
        x = candidate.ravel() if len(scope["lower"]) == 48 else candidate[0]
        a, b = scope["a"], scope["b"]
        violation = max(0.0, float(np.max(scope["lower"] - x)), float(np.max(x - scope["upper"])))
        if len(a):
            violation = max(violation, float(np.max(a @ x - b)))
        membership[name] = dict(within_relaxation=violation <= 1e-8, maximum_violation=violation)
        for row in bounds["scopes"][name]["rows"]:
            weight = float(row["weight"])
            direction = np.zeros(len(scope["lower"]))
            direction[:6] = weight * p_jac + (1 - weight) * r_jac
            check_support_certificate(direction, scope, row["certificate"], tolerance=tolerance)
            expected = (
                row["seed_projection"] - row["certificate"]["support_upper"] - row["remainder"] - bounds["allowance"]
            )
            np.testing.assert_allclose(row["lower_bound"], expected, atol=tolerance, rtol=0)
            observed = weight * p + (1 - weight) * r
            if membership[name]["within_relaxation"] and observed + tolerance < row["lower_bound"]:
                raise ValueError("Saved candidate violates its applicable claimed lower bound")
            verified += 1
    return dict(
        passed=True,
        first_error=actual_error,
        maximum_joint_affine_difference_rad=max_q_error,
        candidate_membership=membership,
        support_certificates_verified=verified,
        independent_first_responses=1,
        independent_transition_steps=25,
        scope="one saved attempt and corrected LP duals; frozen-model diagnostics only",
    )
