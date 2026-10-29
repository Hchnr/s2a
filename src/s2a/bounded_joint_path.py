"""Jointly smooth a bounded geometric pose tube anchored at the actual current q."""

import numpy as np
from scipy.optimize import Bounds, minimize

from s2a.full_coupled_pose_inverse import right_inverse_jacobian
from s2a.geometry import finite, pose_error, validate_transform


class _PathBudget(RuntimeError):
    pass


def plan_joint_path(
    fk_jacobian,
    targets,
    actual_q,
    seed,
    lower,
    upper,
    *,
    position_tolerance=0.01,
    rotation_tolerance=0.05,
    max_iterations=80,
    max_evaluations=160,
):
    """Minimize all adjacent squared joint changes, including q1 minus actual_q.

    Future q are optimization variables; actual_q is never clipped or optimized.
    Seeds must already satisfy the pose tube. No velocity/torque/contact model is
    imposed, so a returned geometric path is not an executable control policy.
    """
    actual_q = np.asarray(actual_q, dtype=float)
    n = len(actual_q)
    actual_q, lower, upper = finite(actual_q, (n,)), finite(lower, (n,)), finite(upper, (n,))
    seed = finite(seed)
    targets = validate_transform(targets)
    tolerances = np.array([position_tolerance, rotation_tolerance], dtype=float)
    if (
        seed.ndim != 2
        or seed.shape[1] != n
        or not 1 <= len(seed) <= 40
        or targets.shape != (len(seed), 4, 4)
        or np.any(lower >= upper)
        or not np.isfinite(tolerances).all()
        or np.any(tolerances <= 0)
        or rotation_tolerance >= np.pi
        or type(max_iterations) is not int
        or max_iterations < 1
        or type(max_evaluations) is not int
        or max_evaluations < 1
    ):
        raise ValueError("Finite aligned path, actual anchor, strict ranges/tube and positive budgets required")
    horizon, width = len(seed), seed.size
    lo, hi = np.tile(lower, horizon), np.tile(upper, horizon)
    fk_cache, path_cache = {}, {}

    def energy(x):
        path = np.asarray(x).reshape(horizon, n)
        difference = np.diff(np.vstack((actual_q, path)), axis=0)
        gradient = difference.copy()
        gradient[:-1] -= difference[1:]
        return 0.5 * float(np.sum(difference**2)), gradient.ravel()

    def evaluate(x):
        key = tuple(np.asarray(x))
        if key not in path_cache:
            if len(path_cache) >= max_evaluations:
                raise _PathBudget("trajectory_evaluation_budget")
            path = np.asarray(x).reshape(horizon, n)
            errors, poses = [], []
            constraints = np.zeros(2 * horizon)
            jacobian = np.zeros((2 * horizon, width))
            for k, q in enumerate(path):
                qkey = tuple(q)
                if qkey not in fk_cache:
                    pose, j = fk_jacobian(q)
                    fk_cache[qkey] = (validate_transform(pose).copy(), finite(j, (6, n)).copy())
                pose, j = fk_cache[qkey]
                error = pose_error(targets[k], pose)
                derivative = np.vstack((-j[:3], -right_inverse_jacobian(error[3:]) @ j[3:]))
                for group, (a, b) in enumerate(((0, 3), (3, 6))):
                    norm = np.linalg.norm(error[a:b])
                    constraints[2 * k + group] = 1 - norm / tolerances[group]
                    direction = error[a:b] / norm if norm > 1e-15 else np.zeros(3)
                    jacobian[2 * k + group, k * n : (k + 1) * n] = -direction @ derivative[a:b] / tolerances[group]
                errors.append(error)
                poses.append(pose)
            path_cache[key] = dict(
                errors=np.asarray(errors), poses=np.asarray(poses), constraints=constraints, jacobian=jacobian
            )
        return path_cache[key]

    def feasible(x):
        if np.any(x < lo - 1e-12) or np.any(x > hi + 1e-12):
            return False
        errors = evaluate(x)["errors"]
        return bool(
            np.all(np.linalg.norm(errors[:, :3], axis=1) <= position_tolerance + 1e-10)
            and np.all(np.linalg.norm(errors[:, 3:], axis=1) <= rotation_tolerance + 1e-10)
        )

    if not feasible(seed.ravel()):
        raise ValueError("Verified feasible whole-path seed required; no branch substitution")
    chosen = seed.copy()
    seed_energy = energy(seed.ravel())[0]
    step, success, iterations, message = 0.0, False, 0, "No optimization result"
    try:
        result = minimize(
            energy,
            seed.ravel(),
            jac=True,
            method="SLSQP",
            bounds=Bounds(lo, hi),
            constraints=[
                dict(type="ineq", fun=lambda x: evaluate(x)["constraints"], jac=lambda x: evaluate(x)["jacobian"])
            ],
            options=dict(maxiter=max_iterations, ftol=1e-10),
        )
        success, iterations, message = bool(result.success), int(result.nit), str(result.message)
        proposal = np.clip(result.x, lo, hi)
        # One shared step for the entire trajectory; never select per-point
        # winners from incompatible IK branches.
        for alpha in (1.0, 0.999, 0.99, 0.9, 0.5, 0.25, 0.125, 0.0625):
            trial = np.clip(seed.ravel() + alpha * (proposal - seed.ravel()), lo, hi)
            if feasible(trial) and energy(trial)[0] < seed_energy:
                chosen, step = trial.reshape(horizon, n).copy(), alpha
                break
    except (_PathBudget, ValueError) as error:
        success = False
        message = f"Original feasible path retained: {error!r}"
    final = evaluate(chosen.ravel())
    difference = np.diff(np.vstack((actual_q, chosen)), axis=0)
    return dict(
        joints=chosen,
        poses=final["poses"],
        errors=final["errors"],
        actual_anchor=actual_q.copy(),
        seed_energy=seed_energy,
        energy=energy(chosen.ravel())[0],
        accepted_step=step,
        optimizer_success=success,
        optimizer_iterations=iterations,
        optimizer_message=message,
        trajectory_evaluations=len(path_cache),
        fk_configurations=len(fk_cache),
        first_joint_difference_max_rad=float(np.max(abs(difference[0]))),
        maximum_joint_difference_rad=float(np.max(abs(difference))),
        actual_initial_penetration_rad=float(max(0, np.max(lower - actual_q), np.max(actual_q - upper))),
        scope="whole-path bounded geometry with actual anchor; no dynamic or controller qualification",
    )
