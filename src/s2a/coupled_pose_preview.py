"""Four-period world-tangent preview proposal; actual native goal commit remains mandatory."""

import numpy as np
from scipy.spatial.transform import Rotation

from s2a.current_coupled_response import build_response
from s2a.geometry import finite, validate_transform
from s2a.osc import OSCInterface
from s2a.strict_convex_qp import solve_qp


def plan_pose(measured, twist, references, reference_twists, previous_command, mass, jacobian, contract, *, innovation):
    """Use only current robot dynamics, completed innovation and already generated references.

    M/J and the identity-relative-goal spring are frozen across at most four periods.
    Completed innovation is held as a per-period additive tangent disturbance.
    This convex proposal does not certify nonlinear or native reset/latch optimality.
    """
    measured = validate_transform(measured)
    references = validate_transform(references)
    if measured.shape != (4, 4) or references.ndim != 3 or not 1 <= len(references) <= 4:
        raise ValueError("One current pose and one to four generated SITE references required")
    if contract["control_freq"] != 20 or contract["control_frame"] != "MuJoCo world":
        raise ValueError("Fixed20Hz world contract required")
    horizon = len(references)
    twist = finite(twist, (6,))
    targets_twist = finite(reference_twists, (horizon, 6))
    previous = finite(previous_command, (6,))
    bias = finite(innovation, (12,))
    interface = OSCInterface(*(contract[k] for k in ("input_min", "input_max", "output_min", "output_max")))
    lo, hi = interface.input_min, interface.input_max
    if np.any(previous < lo) or np.any(previous > hi):
        raise ValueError("Feasible previous actual command required")
    rotations = Rotation.from_matrix(references[:, :3, :3] @ measured[:3, :3].T).as_rotvec()
    if np.any(np.linalg.norm(rotations, axis=1) >= np.pi - 1e-6):
        raise ValueError("Generated target reaches ambiguous SO3 logarithm branch")
    model = build_response(mass, jacobian, contract["kp"], contract["kd"], np.eye(3))
    a = np.zeros((12, 12))
    a[:6, :6] = np.eye(6)
    a[:, 6:] = model["initial_velocity_response"]
    scale = (interface.output_max - interface.output_min) / (hi - lo)
    offset = (interface.output_max + interface.output_min) / 2 - scale * (hi + lo) / 2
    b = model["goal_forcing_response"] * scale[None, :]
    disturbance = model["goal_forcing_response"] @ offset + np.r_[bias[:6], bias[6:] / model["dt"]]
    state = np.r_[np.zeros(6), twist]
    targets = np.c_[references[:, :3, 3] - measured[:3, 3], rotations, targets_twist]
    response = np.zeros((12 * horizon, 6 * horizon))
    bases = []
    for k in range(horizon):
        state = a @ state + disturbance
        bases.append(state.copy())
        for j in range(k + 1):
            response[12 * k : 12 * (k + 1), 6 * j : 6 * (j + 1)] = np.linalg.matrix_power(a, k - j) @ b
    base = np.concatenate(bases)
    weight = np.tile([100.0] * 3 + [20.0] * 3 + [5.0] * 3 + [1.0] * 3, horizon)
    weighted = weight[:, None] * response
    hessian = weighted.T @ weighted
    linear = weighted.T @ (weight * (base - targets.ravel()))
    identity = np.eye(6 * horizon)
    difference = identity.copy()
    for k in range(1, horizon):
        difference[6 * k : 6 * (k + 1), 6 * (k - 1) : 6 * k] = -np.eye(6)
    constraints = np.vstack((identity, -identity, difference, -difference))
    center = np.r_[previous, np.zeros(6 * (horizon - 1))]
    bounds = np.r_[np.tile(hi, horizon), -np.tile(lo, horizon), 0.25 + center, 0.25 - center]
    solution = solve_qp(hessian, linear, constraints, bounds, np.tile(previous, horizon))
    commands = solution["solution"].reshape(horizon, 6)
    prediction = (base + response @ commands.ravel()).reshape(horizon, 12)
    residual = (prediction - targets).ravel() * weight
    return dict(
        command=commands[0].copy(),
        planned_commands=commands.copy(),
        predicted_tangent_states=prediction,
        objective=0.5 * float(residual @ residual),
        horizon=horizon,
        solver_max_stationarity=solution["max_stationarity"],
        solver_iterations=solution["iterations"],
        innovation=bias.copy(),
        innovation_policy="hold_completed_innovation_per_period",
        model_scope="frozen M/J identity-goal world tangent",
        native_decode_and_actual_commit_required=True,
        nonlinear_optimality_claim=False,
    )
