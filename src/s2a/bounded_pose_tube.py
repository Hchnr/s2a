"""Nearest bounded joint configuration in a fixed pose-error tube, geometry only."""

import numpy as np
from scipy.optimize import minimize

from s2a.bounded_pose_ik import solve_pose_ik
from s2a.full_coupled_pose_inverse import right_inverse_jacobian
from s2a.geometry import finite, pose_error, validate_transform


def closest_pose_in_tube(
    fk_jacobian,
    desired,
    previous,
    lower,
    upper,
    *,
    position_tolerance=0.01,
    rotation_tolerance=0.05,
    max_nfev=200,
    max_iterations=60,
):
    """Prefer small joint changes while preserving explicit geometric tolerances.

    This does not generate controller commands or certify collision-free timed
    motion. The pose tube is checked again on the actual returned configuration.
    """
    previous = np.asarray(previous, dtype=float)
    n = len(previous)
    previous, lower, upper = finite(previous, (n,)), finite(lower, (n,)), finite(upper, (n,))
    desired = validate_transform(desired)
    tolerances = np.asarray([position_tolerance, rotation_tolerance], dtype=float)
    if (
        desired.shape != (4, 4)
        or np.any(lower >= upper)
        or not np.isfinite(tolerances).all()
        or np.any(tolerances <= 0)
        or rotation_tolerance >= np.pi
        or type(max_iterations) is not int
        or max_iterations < 1
    ):
        raise ValueError("Strict bounds, positive finite pose tube and iteration budget required")
    cache = {}

    def evaluate(q):
        key = tuple(q)
        if key not in cache:
            pose, jacobian = fk_jacobian(q)
            error = pose_error(desired, pose)
            jacobian = finite(jacobian, (6, n))
            derivative = np.vstack((-jacobian[:3], -right_inverse_jacobian(error[3:]) @ jacobian[3:]))
            cache[key] = dict(pose=pose.copy(), jacobian=jacobian.copy(), error=error, derivative=derivative)
        return cache[key]

    def valid(q):
        error = evaluate(q)["error"]
        return bool(
            np.all(q >= lower - 1e-12)
            and np.all(q <= upper + 1e-12)
            and np.linalg.norm(error[:3]) <= position_tolerance + 1e-10
            and np.linalg.norm(error[3:]) <= rotation_tolerance + 1e-10
        )

    start = np.clip(previous, lower, upper)
    route, success, message, iterations, ik_nfev = (
        "inside_tube",
        True,
        "Previous bounded configuration satisfies tube",
        0,
        0,
    )
    chosen = start.copy()
    if not valid(start):

        def cached_fk(q):
            item = evaluate(q)
            return item["pose"], item["jacobian"]

        cold = solve_pose_ik(cached_fk, desired, start, lower, upper, max_nfev=max_nfev)
        chosen, ik_nfev = cold["joints"].copy(), cold["nfev"]
        route, success, message = "cold_pose_only", False, "No verified tube point from fixed cold solve"
        if valid(chosen):
            feasible_start = chosen.copy()

            def constraints(q):
                error = evaluate(q)["error"]
                return 1 - np.array([np.linalg.norm(error[:3]), np.linalg.norm(error[3:])]) / tolerances

            def constraint_jacobian(q):
                item = evaluate(q)
                rows = []
                for group, (a, b) in enumerate(((0, 3), (3, 6))):
                    error = item["error"][a:b]
                    norm = np.linalg.norm(error)
                    rows.append(
                        -(error / norm if norm > 1e-15 else np.zeros(3)) @ item["derivative"][a:b] / tolerances[group]
                    )
                return np.asarray(rows)

            result = minimize(
                lambda q: 0.5 * float((q - previous) @ (q - previous)),
                feasible_start,
                jac=lambda q: q - previous,
                method="SLSQP",
                bounds=list(zip(lower, upper, strict=True)),
                constraints=[dict(type="ineq", fun=constraints, jac=constraint_jacobian)],
                options=dict(maxiter=max_iterations, ftol=1e-11),
            )
            success, message, iterations = bool(result.success), str(result.message), int(result.nit)
            proposal = np.clip(result.x, lower, upper)
            original_cost = float((feasible_start - previous) @ (feasible_start - previous))
            route = "cold_feasible_point"
            for step in (1.0, 0.999, 0.99, 0.9, 0.5, 0.25, 0.125, 0.0625):
                trial = np.clip(feasible_start + step * (proposal - feasible_start), lower, upper)
                if valid(trial) and float((trial - previous) @ (trial - previous)) <= original_cost:
                    chosen, route = trial, "nearest_tube_candidate"
                    break
    final = evaluate(chosen)
    return dict(
        joints=chosen,
        pose=final["pose"],
        error=final["error"],
        within_tube=valid(chosen),
        solver_success=success,
        solver_message=message,
        optimizer_iterations=iterations,
        cold_ik_nfev=ik_nfev,
        fk_configurations=len(cache),
        route=route,
        joint_change_norm_rad=float(np.linalg.norm(chosen - previous)),
        initial_clip_max_abs_rad=float(np.max(abs(start - previous))),
        scope="bounded pose tube only; no action or dynamic trajectory qualification",
    )
