"""218 scalar response rollout and independent convex KKT certificate."""

import numpy as np
from pose_velocity_arm_independent_replay import PoseVelocityArmReplay
from scipy.optimize import nnls

from s2a.osc import OSCInterface


def verify_plan(plan, position, velocity, references, reference_velocities, previous, contract, innovation):
    interface = OSCInterface(*(contract[k] for k in ("input_min", "input_max", "output_min", "output_max")))
    response = PoseVelocityArmReplay(interface, contract["kp"], contract["kd"], 0.05)
    u = np.asarray(plan["planned_commands"])
    horizon = len(references)
    assert u.shape == (horizon, 3)
    np.testing.assert_allclose(plan["command"], u[0], atol=1e-12, rtol=0)
    maximum = 0.0
    for axis in range(3):

        def rollout(sequence):
            p, v = float(position[axis]), float(velocity[axis])
            result = []
            for value in sequence:
                physical = interface.output_min[axis] + (value - interface.input_min[axis]) * (
                    interface.output_max[axis] - interface.output_min[axis]
                ) / (interface.input_max[axis] - interface.input_min[axis])
                next_p = p + response.momentum[axis] * v + response.response[axis] * physical + innovation[0, axis]
                next_v = (
                    (1 - response.response[axis] - response.kd[axis] * response.momentum[axis]) * v
                    + response.kp[axis] * response.momentum[axis] * physical
                    + innovation[1, axis]
                )
                p, v = next_p, next_v
                result.extend([p, 0.05 * v])
            return np.array(result)

        baseline = rollout(np.zeros(horizon))
        matrix = np.column_stack([rollout(np.eye(horizon)[i]) - baseline for i in range(horizon)]) / 0.01
        target = (
            np.array([[p[axis], 0.05 * v[axis]] for p, v in zip(references, reference_velocities, strict=True)]).ravel()
            / 0.01
        )
        prediction = rollout(u[:, axis])
        np.testing.assert_allclose(plan["predicted_positions"][:, axis], prediction[::2], atol=1e-12, rtol=0)
        np.testing.assert_allclose(plan["predicted_velocities"][:, axis], prediction[1::2] / 0.05, atol=1e-12, rtol=0)
        error = prediction / 0.01 - target
        np.testing.assert_allclose(plan["axes"][axis]["objective"], 0.5 * float(error @ error), atol=1e-9, rtol=0)
        gradient = matrix.T @ error
        difference = np.eye(horizon) - np.eye(horizon, k=-1)
        high, low = np.full(horizon, 0.25), np.full(horizon, 0.25)
        high[0] += previous[axis]
        low[0] -= previous[axis]
        g = np.vstack([np.eye(horizon), -np.eye(horizon), difference, -difference])
        h = np.r_[np.full(horizon, interface.input_max[axis]), np.full(horizon, -interface.input_min[axis]), high, low]
        slack = h - g @ u[:, axis]
        assert np.min(slack) >= -1e-9
        active = slack <= 2e-7
        if np.any(active):
            dual, residual = nnls(g[active].T, -gradient, maxiter=1000)
            assert np.max(np.abs(dual * slack[active])) < 1e-6
        else:
            residual = np.linalg.norm(gradient)
        assert residual < 2e-6, f"QP stationarity residual {residual}"
        maximum = max(maximum, float(residual))
    return dict(convex_kkt_verified=True, max_stationarity_residual=maximum)
