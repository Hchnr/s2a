"""Floating-point outer bounds for joint-constrained first-pose tracking."""

import numpy as np
from scipy.optimize import linprog
from scipy.spatial.transform import Rotation

from s2a.geometry import finite, validate_transform
from s2a.research.joint_control.inverse.full_coupled_pose_inverse import hat, left_jacobian


def first_pose_linearization(
    site, tool, jacobian, joint_displacement, joint_command_jacobian, seed, lower, upper, target
):
    """Linearize frozen first TCP pose and bound its rotation-exponential remainder."""
    site, tool, target = (validate_transform(value) for value in (site, tool, target))
    if any(value.shape != (4, 4) for value in (site, tool, target)):
        raise ValueError("Individual SITE, TCP transform and target are required")
    displacement = finite(joint_displacement)
    if displacement.ndim != 1 or not len(displacement):
        raise ValueError("A nonempty joint displacement is required")
    jacobian = finite(jacobian, (6, len(displacement)))
    joint_command_jacobian = finite(joint_command_jacobian, (len(displacement), 6))
    seed, lower, upper = (finite(value, (6,)) for value in (seed, lower, upper))
    if np.any(lower > upper) or np.any(seed < lower - 1e-12) or np.any(seed > upper + 1e-12):
        raise ValueError("Seed must be in the first-command box")
    movement = jacobian @ displacement
    motion_jac = jacobian @ joint_command_jacobian
    rotation = Rotation.from_rotvec(movement[3:]).as_matrix()
    tcp_rotation = rotation @ site[:3, :3] @ tool[:3, :3]
    offset = site[:3, :3] @ tool[:3, 3]
    rotated_offset = rotation @ offset
    tcp_position = site[:3, 3] + movement[:3] + rotated_offset
    angular_jac = left_jacobian(movement[3:]) @ motion_jac[3:]
    position_jac = motion_jac[:3] - hat(rotated_offset) @ angular_jac
    rotation_jac = np.stack([hat(angular_jac[:, i]) @ tcp_rotation for i in range(6)], axis=-1)

    delta_lo, delta_hi = lower - seed, upper - seed
    angular_lo = np.minimum(motion_jac[3:] * delta_lo, motion_jac[3:] * delta_hi).sum(axis=1)
    angular_hi = np.maximum(motion_jac[3:] * delta_lo, motion_jac[3:] * delta_hi).sum(axis=1)
    angular_abs = np.maximum(abs(angular_lo), abs(angular_hi))
    angular_squared_bound = float(angular_abs @ angular_abs)
    return dict(
        seed_command=seed.copy(),
        command_lower=lower.copy(),
        command_upper=upper.copy(),
        seed_tcp_position=tcp_position,
        seed_tcp_rotation=tcp_rotation,
        target_position=target[:3, 3].copy(),
        target_rotation=target[:3, :3].copy(),
        position_jacobian=position_jac,
        rotation_jacobian=rotation_jac,
        angular_component_bound_rad=angular_abs,
        position_remainder_bound_m=0.5 * float(np.linalg.norm(offset)) * angular_squared_bound,
        rotation_remainder_bound_frobenius=0.5 * np.sqrt(3.0) * angular_squared_bound,
    )


def error_directions(target_rotation, previous_error):
    """Form unit translation and rotation-matrix directions from a prior attempt."""
    target_rotation = finite(target_rotation, (3, 3))
    error = finite(previous_error, (6,))
    position_norm, rotation_norm = np.linalg.norm(error[:3]), np.linalg.norm(error[3:])
    if position_norm <= 1e-15 or rotation_norm <= 1e-15:
        raise ValueError("The recorded position and rotation directions must be nonzero")
    candidate_rotation = Rotation.from_rotvec(-error[3:]).as_matrix() @ target_rotation
    return error[:3] / position_norm, hat(error[3:] / rotation_norm) @ candidate_rotation / np.sqrt(2.0)


def safety_scopes(domain, seed, *, rate_limit=0.25):
    """Build nested box, first-period and eight-period outer relaxations.

    Terminal directions and the native rotation latch are deliberately omitted.
    First-period projection uses the future command box to relax any tiny
    future coefficients rather than silently discarding them.
    """
    seed = finite(seed, (8, 6)).ravel()
    lo, hi = finite(domain["lo"], (48,)), finite(domain["hi"], (48,))
    rate, center = finite(domain["rate"], (48, 48)), finite(domain["center"], (48,))
    a = finite(domain["joint_a"], (1456, 48))
    joint_lo, joint_hi = finite(domain["joint_lo"], (1456,)), finite(domain["joint_hi"], (1456,))
    if not np.isfinite(rate_limit) or rate_limit <= 0 or np.any(lo > hi):
        raise ValueError("A positive rate limit and ordered input box are required")
    if not np.array_equal(rate, np.eye(48) - np.eye(48, k=-6)) or not np.array_equal(center[6:], np.zeros(42)):
        raise ValueError("Original consecutive-command rate rows are required")
    first_lo = np.maximum(lo[:6], center[:6] - rate_limit)
    first_hi = np.minimum(hi[:6], center[:6] + rate_limit)
    if np.any(first_lo > first_hi) or np.any(seed < lo - 1e-12) or np.any(seed > hi + 1e-12):
        raise ValueError("The seed and first-command box must be valid")
    selected = np.r_[np.arange(25 * 7), np.arange(200 * 7, 201 * 7)]
    selected_a = np.vstack((a[selected], -a[selected]))
    selected_b = np.r_[joint_hi[selected], -joint_lo[selected]]
    # Existence of future commands in their box implies these relaxed rows.
    future_minimum = np.minimum(selected_a[:, 6:] * lo[6:], selected_a[:, 6:] * hi[6:]).sum(axis=1)
    first_a, first_b = selected_a[:, :6], selected_b - future_minimum
    full_a = np.vstack((a, -a, rate, -rate))
    full_b = np.r_[joint_hi, -joint_lo, center + rate_limit, -center + rate_limit]
    return {
        "box": dict(lower=first_lo, upper=first_hi, a=np.empty((0, 6)), b=np.empty(0), seed=seed[:6]),
        "first_interval_safety": dict(lower=first_lo, upper=first_hi, a=first_a, b=first_b, seed=seed[:6]),
        "eight_interval_safety": dict(lower=lo, upper=hi, a=full_a, b=full_b, seed=seed),
    }


def support_upper_bound(direction, scope):
    """Upper-bound linear support using nonnegative LP duals and box correction."""
    lower, upper, seed = (finite(scope[key]) for key in ("lower", "upper", "seed"))
    if lower.ndim != 1 or upper.shape != lower.shape or seed.shape != lower.shape or np.any(lower > upper):
        raise ValueError("Ordered finite box and seed dimensions required")
    direction = finite(direction, lower.shape)
    a, b = finite(scope["a"]), finite(scope["b"])
    if a.ndim != 2 or a.shape[1] != len(lower) or b.shape != (len(a),):
        raise ValueError("One finite upper bound per linear row is required")
    if len(a):
        result = linprog(-direction, A_ub=a, b_ub=b, bounds=list(zip(lower, upper, strict=True)), method="highs")
        if not result.success:
            raise ValueError(f"Support LP failed: {result.status}: {result.message}")
        primal = finite(result.x, lower.shape)
        violation = max(
            0.0, float(np.max(a @ primal - b)), float(np.max(lower - primal)), float(np.max(primal - upper))
        )
        if violation > 1e-7:
            raise ValueError("Support LP returned an infeasible primal point")
        dual = np.maximum(0.0, -finite(result.ineqlin.marginals, (len(a),)))
        residual = direction - a.T @ dual
        box_terms = np.maximum(residual * lower, residual * upper)
        upper_absolute = float(b @ dual + box_terms.sum())
        absolute_scale = float(np.abs(b * dual).sum() + np.abs(box_terms).sum())
        indices = np.flatnonzero(dual)
        certificate = dict(
            status=int(result.status),
            message=str(result.message),
            primal=primal,
            max_primal_violation=violation,
            dual_indices=indices,
            dual_values=dual[indices],
            dual_residual=residual,
        )
    else:
        primal = np.where(direction >= 0, upper, lower)
        box_terms = direction * primal
        upper_absolute = float(box_terms.sum())
        absolute_scale = float(np.abs(box_terms).sum())
        certificate = dict(status=0, message="analytic_box", primal=primal, max_primal_violation=0.0)
    rounding = 64 * np.finfo(float).eps * (1 + absolute_scale + float(np.abs(direction * seed).sum()))
    support = upper_absolute + rounding - float(direction @ seed)
    primal_support = float(direction @ (primal - seed))
    if support < primal_support - 1e-10:
        raise ValueError("Corrected support upper bound fell below the primal objective")
    certificate.update(support_upper=support, primal_support=primal_support, support_gap=support - primal_support)
    return certificate


def weighted_pose_bounds(
    linearization, position_direction, rotation_direction, scopes, weights, *, allowance=1e-7, on_support=None
):
    """Bound a convex combination of normalized position and orientation errors."""
    n, d = finite(position_direction, (3,)), finite(rotation_direction, (3, 3))
    if not np.isclose(np.linalg.norm(n), 1, atol=1e-12, rtol=0) or not np.isclose(
        np.linalg.norm(d), 1, atol=1e-12, rtol=0
    ):
        raise ValueError("Position and Frobenius rotation directions must be unit length")
    weights = finite(weights)
    if weights.ndim != 1 or not len(weights) or np.any(weights < 0) or np.any(weights > 1):
        raise ValueError("One or more position weights in [0, 1] are required")
    if not np.isfinite(allowance) or allowance < 0:
        raise ValueError("Nonnegative finite normalized allowance required")
    rotation_cap = 2 * np.sqrt(2.0) * np.sin(0.05 / 2)
    position_error = linearization["target_position"] - linearization["seed_tcp_position"]
    rotation_error = linearization["target_rotation"] - linearization["seed_tcp_rotation"]
    p_gap, r_gap = float(n @ position_error / 0.01), float(np.sum(d * rotation_error) / rotation_cap)
    p_jac = n @ linearization["position_jacobian"] / 0.01
    r_jac = np.einsum("ij,ijk->k", d, linearization["rotation_jacobian"]) / rotation_cap
    p_remainder = linearization["position_remainder_bound_m"] / 0.01
    r_remainder = linearization["rotation_remainder_bound_frobenius"] / rotation_cap
    result = {}
    for name, scope in scopes.items():
        rows = []
        for weight in weights:
            weight = float(weight)
            direction = np.zeros(len(scope["lower"]))
            direction[:6] = weight * p_jac + (1 - weight) * r_jac
            if on_support is not None:
                on_support(name, "attempted")
            certificate = support_upper_bound(direction, scope)
            if on_support is not None:
                on_support(name, "completed")
            seed_gap = weight * p_gap + (1 - weight) * r_gap
            remainder = weight * p_remainder + (1 - weight) * r_remainder
            lower_bound = seed_gap - certificate["support_upper"] - remainder - allowance
            rows.append(
                dict(
                    weight=weight,
                    seed_projection=seed_gap,
                    remainder=remainder,
                    lower_bound=lower_bound,
                    certificate=certificate,
                )
            )
        result[name] = dict(rows=rows, best_lower_bound=max(row["lower_bound"] for row in rows))
    return dict(
        scopes=result, position_direction=n, rotation_direction=d, rotation_chord_cap=rotation_cap, allowance=allowance
    )
