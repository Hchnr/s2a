"""Independent ODE propagation and finite-difference KKT for tangent preview proposals."""

import numpy as np
from current_coupled_stable_independent import response
from scipy.optimize import nnls
from scipy.spatial.transform import Rotation


def verify_preview(
    saved, measured, twist, references, reference_twists, previous_command, mass, jacobian, contract, *, innovation
):
    measured, twist, references, reference_twists, previous, bias = (
        np.asarray(v, dtype=float)
        for v in (measured, twist, references, reference_twists, previous_command, innovation)
    )
    horizon = len(references)
    assert 1 <= horizon <= 4 and saved["horizon"] == horizon
    assert contract["control_freq"] == 20 and contract["control_frame"] == "MuJoCo world"
    assert saved["native_decode_and_actual_commit_required"] and not saved["nonlinear_optimality_claim"]
    assert saved["innovation_policy"] == "hold_completed_innovation_per_period"
    np.testing.assert_array_equal(saved["innovation"], bias)
    model = response(mass, jacobian, contract["kp"], contract["kd"], np.eye(3))
    lo, hi, outlo, outhi = (np.asarray(contract[k]) for k in ("input_min", "input_max", "output_min", "output_max"))
    targets = np.c_[
        references[:, :3, 3] - measured[:3, 3],
        Rotation.from_matrix(references[:, :3, :3] @ measured[:3, :3].T).as_rotvec(),
        reference_twists,
    ]
    weights = np.array([100.0] * 3 + [20.0] * 3 + [5.0] * 3 + [1.0] * 3)

    def evaluate(flat):
        position, velocity = np.zeros(6), twist.copy()
        states = []
        for u in flat.reshape(horizon, 6):
            # Do not clip finite-difference probes; differentiate the unconstrained quadratic.
            delta = (u - (hi + lo) / 2) * (outhi - outlo) / (hi - lo) + (outhi + outlo) / 2
            propagated = model["initial_velocity_response"] @ velocity + model["goal_forcing_response"] @ delta
            position += propagated[:6] + bias[:6]
            velocity = propagated[6:] + bias[6:] / 0.05
            states.append(np.r_[position, velocity])
        states = np.asarray(states)
        residual = (states - targets) * weights
        return 0.5 * float(np.sum(residual**2)), states

    commands = np.asarray(saved["planned_commands"], dtype=float)
    assert commands.shape == (horizon, 6) and np.isfinite(commands).all()
    np.testing.assert_array_equal(saved["command"], commands[0])
    flat = commands.ravel()
    value, states = evaluate(flat)
    np.testing.assert_allclose(saved["predicted_tangent_states"], states, atol=1e-8, rtol=0)
    np.testing.assert_allclose(saved["objective"], value, atol=1e-7, rtol=0)
    identity = np.eye(len(flat))
    difference = identity - np.eye(len(flat), k=-6)
    constraints = np.vstack([identity, -identity, difference, -difference])
    center = np.r_[previous, np.zeros(6 * (horizon - 1))]
    bounds = np.r_[np.tile(hi, horizon), -np.tile(lo, horizon), 0.25 + center, 0.25 - center]
    slack = bounds - constraints @ flat
    assert slack.min() >= -1e-9
    # Two fixed finite-difference steps reject cancellation-sensitive certificates.
    gradients = []
    for epsilon in (1e-4, 1e-5):
        gradients.append(
            np.array(
                [
                    (evaluate(flat + axis * epsilon)[0] - evaluate(flat - axis * epsilon)[0]) / (2 * epsilon)
                    for axis in identity
                ]
            )
        )
    np.testing.assert_allclose(gradients[0], gradients[1], atol=2e-7, rtol=0)
    active = constraints[slack < 1e-8]
    gradient = gradients[1]
    multiplier = nnls(active.T, -gradient, maxiter=10000)[0] if len(active) else np.zeros(0)
    stationarity = float(np.max(np.abs(gradient + active.T @ multiplier)))
    complementarity = float(np.max(np.abs(multiplier * slack[slack < 1e-8]))) if len(active) else 0.0
    assert stationarity < 2e-6 and complementarity < 1e-8
    return dict(
        max_independent_stationarity=stationarity,
        max_complementarity=complementarity,
        maximum_prediction_difference=float(np.max(np.abs(np.asarray(saved["predicted_tangent_states"]) - states))),
        horizon=horizon,
    )
