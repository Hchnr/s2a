"""Independent five-point SO(3) derivative and tighter DOP853 response integration."""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import nnls
from scipy.spatial.transform import Rotation


def orientation_error(goal, current):
    return sum((np.cross(current[:, i], goal[:, i]) for i in range(3)), np.zeros(3)) / 2


def response(mass, jacobian, kp, kd, relative_goal):
    mass, j, kp, kd, a = (np.asarray(value, dtype=float) for value in (mass, jacobian, kp, kd, relative_goal))
    inverse_inertia = j @ np.linalg.inv(mass) @ j.T
    decoupling = np.zeros((6, 6))
    decoupling[:3, :3] = np.linalg.pinv(inverse_inertia[:3, :3])
    decoupling[3:, 3:] = np.linalg.pinv(inverse_inertia[3:, 3:])
    coupling = inverse_inertia @ decoupling
    forcing = orientation_error(a, np.eye(3))
    stiffness = np.empty((3, 3))
    for axis in range(3):
        h = 1e-3
        direction = np.eye(3)[axis]

        def value(scale):
            return orientation_error(a, Rotation.from_rotvec(scale * direction).as_matrix())

        stiffness[:, axis] = -(value(-2 * h) - 8 * value(-h) + 8 * value(h) - value(2 * h)) / (12 * h)
    spring = np.diag(kp)
    spring[3:, 3:] = np.diag(kp[3:]) @ stiffness
    constant = np.zeros((12, 12))
    constant[6:, 6:] = coupling @ np.diag(kp)
    initial = np.zeros((12, 12))
    initial[6:, :6] = np.eye(6)

    def rhs(_, flattened):
        state = flattened.reshape(12, 12)
        derivative = np.zeros_like(state)
        derivative[:6] = state[6:]
        derivative[6:] = -coupling @ (spring @ state[:6] + kd[:, None] * state[6:])
        return (derivative + constant).ravel()

    result = solve_ivp(rhs, (0, 0.05), initial.ravel(), method="DOP853", rtol=5e-14, atol=1e-15)
    assert result.success
    end = result.y[:, -1].reshape(12, 12)
    return dict(
        initial_velocity_response=end[:, :6],
        goal_forcing_response=end[:, 6:],
        rotation_forcing=forcing,
        rotation_stiffness=stiffness,
        acceleration_coupling=coupling,
    )


def verify_inverse(
    solution, position, twist, desired_position, desired_velocity, previous, contract, matrices, innovation
):
    position, twist, target, reference_velocity, previous, bias = (
        np.asarray(value, dtype=float)
        for value in (position, twist, desired_position, desired_velocity, previous, innovation)
    )
    lo, hi, outlo, outhi = (
        np.asarray(contract[key], dtype=float)[:3] for key in ("input_min", "input_max", "output_min", "output_max")
    )
    f, g = matrices["initial_velocity_response"], matrices["goal_forcing_response"]
    rot = matrices["rotation_forcing"]
    slope = (outhi - outlo) / (hi - lo)
    offset = outlo - slope * lo
    base = f @ twist + g @ np.r_[offset, rot]
    design = np.vstack([g[:3, :3], 0.05 * g[6:9, :3]]) * slope[None, :] / 0.01
    desired = np.r_[target - position - base[:3] - bias[0], 0.05 * (reference_velocity - base[6:9] - bias[1])] / 0.01
    command = np.asarray(solution["command"], dtype=float)
    lower, upper = np.maximum(lo, previous - 0.25), np.minimum(hi, previous + 0.25)
    assert command.shape == (3,) and np.all(command >= lower - 1e-9) and np.all(command <= upper + 1e-9)
    physical = outlo + (command - lo) * (outhi - outlo) / (hi - lo)
    nominal = f @ twist + g @ np.r_[physical, rot]
    predicted = nominal.copy()
    predicted[:3] += bias[0]
    predicted[6:9] += bias[1]
    for key, value in dict(
        physical_delta=physical,
        nominal_response=nominal,
        predicted_response=predicted,
        innovation=bias,
        lower=lower,
        upper=upper,
    ).items():
        np.testing.assert_allclose(solution[key], value, atol=1e-8, rtol=0)
    error = design @ command - desired
    np.testing.assert_allclose(solution["objective"], 0.5 * float(error @ error), atol=1e-7, rtol=0)
    gradient = design.T @ error
    constraints = np.vstack([np.eye(3), -np.eye(3)])
    slack = np.r_[upper, -lower] - constraints @ command
    active = slack <= 2e-7
    if np.any(active):
        multipliers, residual = nnls(constraints[active].T, -gradient, maxiter=1000)
        assert np.max(np.abs(multipliers * slack[active])) < 1e-6
    else:
        residual = np.linalg.norm(gradient)
    assert residual < 2e-6, f"Coupled inverse stationarity {residual}"
    return dict(independent_kkt_verified=True, max_stationarity_residual=float(residual))
