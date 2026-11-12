"""Linear box-overlap constraints for offline symmetric gripper closing."""

import numpy as np
from scipy.optimize import linprog, minimize

from s2a.research.native_runtime.box_separation import box_separation


def pair_constraints(pad, target, jaw_axis, direction, margin):
    """Constrain translation t and scalar closing d for one pad/target pair."""
    if direction not in (-1, 1) or not np.isfinite(margin) or margin <= 0:
        raise ValueError("A signed closing direction and positive overlap margin are required")
    center, rotation, half = (
        np.asarray(pad[k], dtype=float) for k in ("center_world_m", "rotation_world", "half_size_m")
    )
    other, other_rotation, other_half = (
        np.asarray(target[k], dtype=float) for k in ("center_world_m", "rotation_world", "half_size_m")
    )
    jaw = np.asarray(jaw_axis, dtype=float)
    if jaw.shape != (3,) or not np.isfinite(jaw).all() or not np.isclose(np.linalg.norm(jaw), 1, atol=1e-10, rtol=0):
        raise ValueError("A finite unit jaw axis is required")
    box_separation(center, rotation, half, other, other_rotation, other_half)
    axes = [rotation[:, i] for i in range(3)] + [other_rotation[:, i] for i in range(3)]
    axes += [np.cross(rotation[:, i], other_rotation[:, j]) for i in range(3) for j in range(3)]
    rows, bounds = [], []
    for raw in axes:
        norm = float(np.linalg.norm(raw))
        if norm <= 1e-12:
            continue
        axis = raw / norm
        radius = float(np.abs(rotation.T @ axis) @ half + np.abs(other_rotation.T @ axis) @ other_half)
        base = float((other - center) @ axis)
        move = np.r_[axis, direction * float(jaw @ axis)]
        rows.extend((-move, move))
        bounds.extend((radius - margin - base, radius - margin + base))
    return np.asarray(rows), np.asarray(bounds)


def solve_infinity(left_constraints, right_constraints, reference_error, maximum_close):
    """Minimize the infinity-norm position error, returning solver and witness details."""
    e = np.asarray(reference_error, dtype=float)
    if e.shape != (3,) or not np.isfinite(e).all() or not np.isfinite(maximum_close) or maximum_close < 0:
        raise ValueError("Finite reference error and nonnegative joint closing range are required")
    rows = np.vstack((left_constraints[0], right_constraints[0]))
    bounds = np.r_[left_constraints[1], right_constraints[1]]
    a = np.column_stack((rows, np.zeros(len(rows))))
    pose_rows = np.zeros((6, 5))
    for index in range(3):
        pose_rows[2 * index, index], pose_rows[2 * index, 4] = 1, -1
        pose_rows[2 * index + 1, index], pose_rows[2 * index + 1, 4] = -1, -1
    a = np.vstack((a, pose_rows))
    b = np.r_[bounds, np.column_stack((e, -e)).ravel()]
    result = linprog(
        [0, 0, 0, 0, 1], A_ub=a, b_ub=b, bounds=[(None, None)] * 3 + [(0, maximum_close), (0, None)], method="highs"
    )
    value = dict(status=int(result.status), iterations=int(result.nit), success=bool(result.success))
    if result.success:
        x = np.asarray(result.x, dtype=float)
        violation = max(0.0, float(np.max(a @ x - b)), float(-x[3]), float(x[3] - maximum_close))
        if not np.isfinite(x).all() or violation > 1e-8:
            raise ValueError("A finite, constraint-verified linear solution is required")
        value.update(
            translation_world_m=x[:3],
            closing_m=float(x[3]),
            infinity_error_m=float(x[4]),
            euclidean_error_m=float(np.linalg.norm(x[:3] - e)),
            maximum_constraint_violation_m=violation,
        )
    return value


def refine_euclidean(left_constraints, right_constraints, reference_error, maximum_close, initial):
    """Seek a Euclidean position-gate witness; a failed solve is inconclusive."""
    e = np.asarray(reference_error, dtype=float)
    rows = np.vstack((left_constraints[0], right_constraints[0]))
    bounds = np.r_[left_constraints[1], right_constraints[1]]
    x0 = np.r_[initial["translation_world_m"], initial["closing_m"]]

    def objective(x):
        return float(np.sum((x[:3] - e) ** 2))

    def jac(x):
        return np.r_[2 * (x[:3] - e), 0.0]

    constraint = dict(type="ineq", fun=lambda x: bounds - rows @ x, jac=lambda _x: -rows)
    result = minimize(
        objective,
        x0,
        jac=jac,
        constraints=[constraint],
        bounds=[(None, None)] * 3 + [(0, maximum_close)],
        method="SLSQP",
        options=dict(ftol=1e-12, maxiter=200),
    )
    x = np.asarray(result.x, dtype=float)
    violation = max(0.0, float(np.max(rows @ x - bounds)), float(-x[3]), float(x[3] - maximum_close))
    return dict(
        solver_success=bool(result.success),
        status=int(result.status),
        iterations=int(result.nit),
        translation_world_m=x[:3],
        closing_m=float(x[3]),
        euclidean_error_m=float(np.linalg.norm(x[:3] - e)),
        maximum_constraint_violation_m=violation,
    )
