"""Construct bounded geometric pose witnesses; never certify dynamic reachability."""

import numpy as np
from scipy.optimize import least_squares

from s2a.full_coupled_pose_inverse import right_inverse_jacobian
from s2a.geometry import finite, pose_error, validate_transform


def solve_pose_ik(fk_jacobian, desired, initial, lower, upper, *, max_nfev=200):
    """Solve one bounded pose from a fixed initial guess and report actual residual.

    fk_jacobian(q) returns the TCP pose and its world linear/angular Jacobian.
    A small residual constructs a pose witness. Failure is not a global proof
    of unreachability; neither result proves a collision-free timed trajectory.
    """
    initial = np.asarray(initial, dtype=float)
    n = len(initial)
    initial, lower, upper = finite(initial, (n,)), finite(lower, (n,)), finite(upper, (n,))
    desired = validate_transform(desired)
    if desired.shape != (4, 4) or np.any(lower >= upper) or type(max_nfev) is not int or max_nfev < 1:
        raise ValueError("One target, strict joint ranges and positive evaluation budget required")
    start = np.clip(initial, lower, upper)
    weights = np.array([100.0] * 3 + [20.0] * 3)
    cache = {}

    def evaluate(q):
        key = tuple(q)
        if key not in cache:
            pose, jacobian = fk_jacobian(q)
            error = pose_error(desired, pose)
            jacobian = finite(jacobian, (6, n))
            derivative = np.vstack((-jacobian[:3], -right_inverse_jacobian(error[3:]) @ jacobian[3:]))
            cache[key] = (weights * error, weights[:, None] * derivative, error, validate_transform(pose).copy())
        return cache[key]

    result = least_squares(
        lambda q: evaluate(q)[0],
        start,
        jac=lambda q: evaluate(q)[1],
        bounds=(lower, upper),
        max_nfev=max_nfev,
        ftol=1e-10,
        xtol=1e-10,
        gtol=1e-10,
    )
    _, _, error, pose = evaluate(result.x)
    in_bounds = bool(np.all(result.x >= lower - 1e-12) and np.all(result.x <= upper + 1e-12))
    witness = bool(in_bounds and np.linalg.norm(error[:3]) <= 1e-5 and np.linalg.norm(error[3:]) <= 1e-4)
    return dict(
        joints=result.x.copy(),
        pose=pose,
        error=error.copy(),
        geometric_witness=witness,
        in_bounds=in_bounds,
        solver_success=bool(result.success),
        solver_message=str(result.message),
        nfev=int(result.nfev),
        initial_clip_max_abs_rad=float(np.max(abs(start - initial))),
        scope="bounded kinematics only; no collision, velocity, torque or controller feasibility guarantee",
    )
