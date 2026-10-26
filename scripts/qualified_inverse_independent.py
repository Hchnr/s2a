"""Independent affine prediction and KKT audit; never calls the inverse solver."""

import numpy as np
from arm_joint_wrench_independent import prediction


def verify_inverse(data, parameters, contract, previous, desired_motion, desired_velocity, command):
    lo, hi, outlo, outhi = (
        np.asarray(contract[k], dtype=float)[:3] for k in ("input_min", "input_max", "output_min", "output_max")
    )
    previous, command = np.asarray(previous, dtype=float), np.asarray(command, dtype=float)
    assert previous.shape == command.shape == (3,)
    assert np.isfinite(previous).all() and np.isfinite(command).all()
    assert np.all(previous >= lo) and np.all(previous <= hi)
    lower, upper = np.maximum(lo, previous - 0.25), np.minimum(hi, previous + 0.25)
    assert np.all(command >= lower - 1e-12) and np.all(command <= upper + 1e-12)
    scale = (outhi - outlo) / (hi - lo)

    def evaluate(value):
        row = {k: np.asarray(v).copy() for k, v in data.items()}
        row["action_command"][0, :3] = value
        row["action_goal_delta"][0, :3] = outlo + (value - lo) * scale
        return prediction(row, parameters)[0]

    baseline = evaluate(lo)
    matrix = np.empty((12, 3))
    for j in range(3):
        value = lo.copy()
        value[j] = hi[j]
        matrix[:, j] = (evaluate(value) - baseline) / (hi[j] - lo[j])
    constant = baseline - matrix @ lo
    full = evaluate(command)
    np.testing.assert_allclose(full, constant + matrix @ command, rtol=1e-8, atol=1e-9)
    target = np.r_[np.asarray(desired_motion, dtype=float), 0.05 * np.asarray(desired_velocity, dtype=float)]
    assert target.shape == (6,) and np.isfinite(target).all()
    rows = [0, 1, 2, 6, 7, 8]
    residual = full[rows] - target
    gradient = matrix[rows].T @ residual
    tolerance = 1e-9 * (1 + np.linalg.norm(matrix[rows]) * np.linalg.norm(residual))
    for j in range(3):
        if upper[j] - lower[j] <= 1e-12:
            continue
        if command[j] <= lower[j] + 1e-10:
            assert gradient[j] >= -tolerance, "Lower-bound KKT violation"
        elif command[j] >= upper[j] - 1e-10:
            assert gradient[j] <= tolerance, "Upper-bound KKT violation"
        else:
            assert abs(gradient[j]) <= tolerance, "Interior KKT violation"
    return dict(
        full_prediction=full,
        gradient=gradient,
        objective=float(residual @ residual),
        physical_delta=outlo + (command - lo) * scale,
        lower=lower,
        upper=upper,
    )
