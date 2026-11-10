"""Eight-period joint and TCP constraints in a frozen causal preview."""

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, minimize

from s2a.geometry import finite
from s2a.research.joint_control.limits.joint_limit_preview_objective import preview_objective
from s2a.research.joint_control.preview.interior_joint_preview import rollout_joint_preview


def pose_quality_constraints(score, horizon, scale):
    """Return normalized caps and analytic command Jacobian for every period."""
    if type(horizon) is not int or not 1 <= horizon <= 8:
        raise ValueError("One to eight generated targets required")
    scale = finite(scale, (6 * horizon,))
    residual = finite(score["residual"], (12 * horizon,)).reshape(horizon, 12)
    jacobian = finite(score["jacobian"], (12 * horizon, 6 * horizon)).reshape(horizon, 12, 6 * horizon)
    values, rows, errors = [], [], []
    for k in range(horizon):
        error = residual[k, :6] / np.array([100.0] * 3 + [20.0] * 3)
        derivative = jacobian[k, :6] * scale[None, :] / np.array([100.0] * 3 + [20.0] * 3)[:, None]
        errors.append(error)
        for start, stop, cap in ((0, 3, 0.01), (3, 6, 0.05)):
            norm = np.linalg.norm(error[start:stop])
            direction = error[start:stop] / norm if norm > 1e-15 else np.zeros(3)
            values.append(1.0 - norm / cap)
            rows.append(-direction @ derivative[start:stop] / cap)
    return np.asarray(values), np.asarray(rows), np.asarray(errors)


def solve_full_horizon(
    model,
    velocity,
    seed,
    previous,
    interface,
    site,
    tool,
    targets,
    twists,
    lower,
    upper,
    *,
    initial_guess=None,
    past_inner_error_rad=None,
    terminal_normals=None,
    braking_time_s=0.1,
):
    """Search for a model plan satisfying all original quality and safety caps."""
    seed, previous = finite(seed), finite(previous, (6,))
    if seed.ndim != 2 or seed.shape[1] != 6 or not 1 <= len(seed) <= 8:
        raise ValueError("One to eight seed commands required")
    h, width = len(seed), seed.size
    lower, upper = finite(lower, (7,)), finite(upper, (7,))
    error = np.zeros((25, 7)) if past_inner_error_rad is None else finite(past_inner_error_rad, (25, 7))
    if np.any(error < 0) or np.any(np.diff(error, axis=0) < 0):
        raise ValueError("Nonnegative nondecreasing completed inner error required")
    if not np.isclose(braking_time_s, 0.1, atol=0, rtol=0):
        raise ValueError("Registered 0.1 s braking condition required")
    peak = error.max(axis=0)
    if np.any(lower + 0.05 + peak >= upper - 0.05 - peak):
        raise ValueError("Joint limits do not contain the requested tightened interior")
    margin = np.tile(0.05 + peak, (25 * h, 1))
    margin[:25] = 0.05 + error
    lo, hi = np.tile(interface.input_min, h), np.tile(interface.input_max, h)
    rate = np.eye(width) - np.eye(width, k=-6)
    center = np.r_[previous, np.zeros(width - 6)]

    def check_box_rate(commands):
        x = finite(commands, seed.shape).ravel()
        if np.any(x < lo - 1e-12) or np.any(x > hi + 1e-12) or np.max(abs(rate @ x - center)) > 0.25 + 1e-12:
            raise ValueError("Seed and warm plan must satisfy original box/rate")
        if np.any(np.all(np.asarray([interface.decode(row) for row in x.reshape(h, 6)])[:, 3:] == 0, axis=1)):
            raise ValueError("Native rotation latch is unsupported in the frozen model")
        return x

    seed_x = check_box_rate(seed)
    start = seed_x if initial_guess is None else check_box_rate(initial_guess)
    scale = np.tile((interface.output_max - interface.output_min) / (interface.input_max - interface.input_min), h)
    cache = {}

    class Stop(RuntimeError):
        pass

    def evaluate(x):
        key = tuple(x)
        if key not in cache:
            if len(cache) >= 320:
                raise Stop("response_budget")
            delta = np.asarray([interface.decode(row) for row in np.asarray(x).reshape(h, 6)])
            if np.any(np.all(delta[:, 3:] == 0, axis=1)):
                raise Stop("unsupported_rotation_latch")
            rollout = rollout_joint_preview(model, velocity, delta, site, lower, upper, enforce_limits=False)
            cache[key] = {"rollout": rollout, "score": preview_objective(rollout, tool, targets, twists)}
        return cache[key]

    original = evaluate(seed_x)
    rollout = original["rollout"]
    q = rollout["inner_joint_position"].ravel()
    a = rollout["inner_joint_position_jacobian"].reshape(len(q), width) * scale
    offset = q - a @ seed_x
    joint_lo, joint_hi = (lower + margin).ravel() - offset, (upper - margin).ravel() - offset
    brake = rollout["joint_position"] + braking_time_s * rollout["joint_velocity"]
    brake_jac = rollout["joint_position_jacobian"] + braking_time_s * rollout["joint_velocity_jacobian"]
    brake_a = brake_jac.reshape(brake.size, width) * scale
    brake_offset = brake.ravel() - brake_a @ seed_x
    a = np.vstack((a, brake_a))
    joint_lo = np.r_[joint_lo, np.tile(lower + 0.05, h) - brake_offset]
    joint_hi = np.r_[joint_hi, np.tile(upper - 0.05, h) - brake_offset]
    if terminal_normals is None:
        inner, identity = rollout["inner_joint_position"], np.eye(7)
        terminal_normals = np.vstack(
            (identity[np.any(inner >= upper - 0.05, axis=0)], -identity[np.any(inner <= lower + 0.05, axis=0)])
        )
    normals = finite(terminal_normals)
    if normals.ndim != 2 or normals.shape[1] != 7:
        raise ValueError("Finite terminal joint direction rows required")
    terminal_a = normals @ rollout["joint_velocity_jacobian"][-1] * scale[None, :]
    terminal_offset = normals @ rollout["joint_velocity"][-1] - terminal_a @ seed_x
    base = {
        "seed_commands": seed.copy(),
        "initial_guess_used": initial_guess is not None,
        "terminal_normals": normals.copy(),
        "past_inner_error_rad": error.copy(),
        "joint_margin_rad": 0.05,
        "braking_time_s": 0.1,
        "position_cap_m": 0.01,
        "rotation_cap_rad": 0.05,
        "pose_horizon": h,
        "initial_margin_rad": float(np.min(np.minimum(model["q"] - lower, upper - model["q"]))),
        "scope": "frozen model only; all generated TCP points constrained; no physical qualification",
        "seed_tcp_errors": pose_quality_constraints(original["score"], h, scale)[2],
    }
    if base["initial_margin_rad"] < 0.05 - 1e-12:
        return dict(**base, status="initial_outside_requested_interior", witness_found=False, response_evaluations=1)
    lp = linprog(
        np.zeros(width),
        A_ub=np.vstack((a, -a, rate, -rate, terminal_a)),
        b_ub=np.r_[joint_hi, -joint_lo, center + 0.25, -center + 0.25, -terminal_offset],
        bounds=list(zip(lo, hi, strict=True)),
        method="highs",
    )
    base.update(linear_status=int(lp.status), linear_message=str(lp.message))
    if not lp.success:
        return dict(**base, status="linear_solver_no_witness", witness_found=False, response_evaluations=len(cache))
    if initial_guess is not None:
        base["initial_guess_tcp_errors"] = pose_quality_constraints(evaluate(start)["score"], h, scale)[2]

    def values(x):
        return pose_quality_constraints(evaluate(x)["score"], h, scale)[0]

    def jac(x):
        return pose_quality_constraints(evaluate(x)["score"], h, scale)[1]

    def objective(x):
        score = evaluate(x)["score"]
        return score["objective"], score["gradient"] * scale

    success, iterations, message = False, 0, "not_run"
    try:
        optimized = minimize(
            objective,
            start,
            jac=True,
            method="SLSQP",
            bounds=Bounds(lo, hi),
            constraints=[
                LinearConstraint(rate, center - 0.25, center + 0.25),
                LinearConstraint(a, joint_lo, joint_hi),
                {"type": "ineq", "fun": values, "jac": jac},
            ]
            + ([LinearConstraint(terminal_a, -np.inf, -terminal_offset)] if len(normals) else []),
            options={"maxiter": 120, "ftol": 1e-9},
        )
        success, iterations, message = bool(optimized.success), int(optimized.nit), str(optimized.message)
    except Stop as exc:
        message = str(exc)
    best = None
    for key, item in list(cache.items()):
        x = np.asarray(key)
        _values, _jacobian, errors = pose_quality_constraints(item["score"], h, scale)
        if (
            np.all(x >= lo - 1e-12)
            and np.all(x <= hi + 1e-12)
            and np.max(abs(rate @ x - center)) <= 0.25 + 1e-12
            and np.all(a @ x >= joint_lo - 1e-10)
            and np.all(a @ x <= joint_hi + 1e-10)
            and np.all(terminal_a @ x + terminal_offset <= 1e-10)
            and np.all(np.linalg.norm(errors[:, :3], axis=1) <= 0.01 + 1e-10)
            and np.all(np.linalg.norm(errors[:, 3:], axis=1) <= 0.05 + 1e-10)
            and (best is None or item["score"]["objective"] < best[1]["score"]["objective"])
        ):
            best = (x, item, errors)
    base.update(
        optimizer_success=success,
        iterations=iterations,
        optimizer_message=message,
        response_evaluations=len(cache),
        witness_found=best is not None,
    )
    if best is None:
        return dict(**base, status="nonlinear_search_no_witness")
    x, item, errors = best
    return dict(
        **base,
        status="model_witness",
        planned_commands=x.reshape(h, 6).copy(),
        first_tcp_error=errors[0].copy(),
        tcp_errors=errors.copy(),
        rollout=item["rollout"],
        objective=item["score"]["objective"],
        first_pose_caps_enforced=True,
    )
