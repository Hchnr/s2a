"""Strictly feasible reset-mode preview commands with post-projection KKT checks."""

import numpy as np
from scipy.optimize import nnls
from scipy.spatial.transform import Rotation

from s2a.coupled_pose_preview import plan_pose as original_plan
from s2a.current_coupled_response import build_response
from s2a.osc import OSCInterface


def plan_pose(measured, twist, references, reference_twists, previous_command, mass, jacobian, contract, *, innovation):
    result = original_plan(
        measured, twist, references, reference_twists, previous_command, mass, jacobian, contract, innovation=innovation
    )
    raw = np.asarray(result["planned_commands"])
    horizon = len(raw)
    lo, hi = (np.asarray(contract[k]) for k in ("input_min", "input_max"))
    previous = np.asarray(previous_command).copy()
    commands = raw.copy()
    interface = OSCInterface(*(contract[k] for k in ("input_min", "input_max", "output_min", "output_max")))
    reset_representative = False
    if np.all(interface.decode(commands[0])[3:] == 0):
        lower, upper = np.maximum(lo, previous - 0.25), np.minimum(hi, previous + 0.25)
        for axis in range(3, 6):
            for sign in (1, -1):
                candidate = commands[0].copy()
                candidate[axis] += sign * 1e-10
                if lower[axis] <= candidate[axis] <= upper[axis] and np.any(interface.decode(candidate)[3:] != 0):
                    commands[0] = candidate
                    reset_representative = True
                    break
            if reset_representative:
                break
        if not reset_representative:
            raise ValueError("No feasible nonzero native reset representative")
    for index in range(horizon):
        commands[index] = np.clip(commands[index], np.maximum(lo, previous - 0.25), np.minimum(hi, previous + 0.25))
        previous = commands[index]
    if np.all(interface.decode(commands[0])[3:] == 0):
        raise ValueError("Feasibility projection erased the native reset representative")
    change = float(np.max(np.abs(commands - raw)))
    if change > 1e-9:
        raise ValueError("QP requires more than numerical feasibility correction")
    model = build_response(mass, jacobian, contract["kp"], contract["kd"], np.eye(3))
    a = np.zeros((12, 12))
    a[:6, :6] = np.eye(6)
    a[:, 6:] = model["initial_velocity_response"]
    b = (
        model["goal_forcing_response"]
        * ((np.asarray(contract["output_max"]) - contract["output_min"]) / (hi - lo))[None, :]
    )
    response = np.zeros((12 * horizon, 6 * horizon))
    for k in range(horizon):
        for j in range(k + 1):
            response[12 * k : 12 * (k + 1), 6 * j : 6 * (j + 1)] = np.linalg.matrix_power(a, k - j) @ b
    prediction = (np.asarray(result["predicted_tangent_states"]).ravel() + response @ (commands - raw).ravel()).reshape(
        horizon, 12
    )
    measured, references = np.asarray(measured), np.asarray(references)
    targets = np.c_[
        references[:, :3, 3] - measured[:3, 3],
        Rotation.from_matrix(references[:, :3, :3] @ measured[:3, :3].T).as_rotvec(),
        reference_twists,
    ]
    weight = np.tile([100.0] * 3 + [20.0] * 3 + [5.0] * 3 + [1.0] * 3, horizon)
    residual = (prediction - targets).ravel() * weight
    gradient = (response * weight[:, None]).T @ residual
    identity = np.eye(6 * horizon)
    difference = identity - np.eye(6 * horizon, k=-6)
    constraints = np.vstack((identity, -identity, difference, -difference))
    center = np.r_[previous_command, np.zeros(6 * (horizon - 1))]
    bounds = np.r_[np.tile(hi, horizon), -np.tile(lo, horizon), 0.25 + center, 0.25 - center]
    slack = bounds - constraints @ commands.ravel()
    active = constraints[slack < 1e-8]
    multipliers = nnls(active.T, -gradient, maxiter=10000)[0] if len(active) else np.zeros(0)
    residual_kkt = float(np.max(np.abs(gradient + active.T @ multipliers)))
    if residual_kkt >= 2e-6 or slack.min() < -1e-9:
        raise ValueError("Numerically bounded plan fails unchanged KKT/feasibility gate")
    result.update(
        command=commands[0].copy(),
        planned_commands=commands,
        predicted_tangent_states=prediction,
        objective=0.5 * float(residual @ residual),
        raw_solver_max_stationarity=result["solver_max_stationarity"],
        solver_max_stationarity=residual_kkt,
        command_projection_max_abs=change,
        reset_representative_adjusted=reset_representative,
        goal_mode="reset",
        command_projection_policy="sequential box/rate correction and nonzero reset representative at most 1e-9",
    )
    return result
