"""Diagnostic reentry search: initial state may be below buffer; all future inner endpoints retain .05."""

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, minimize

from s2a.geometry import finite
from s2a.interior_joint_preview import rollout_joint_preview
from s2a.joint_limit_preview_objective import preview_objective


def solve_interior_plan(
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
    terminal_normals=None,
):
    """Find a model witness; original first-pose caps are recorded, never imposed.

    Joint positions are affine in actions for frozen free M/J, including anchor
    resets. All 2ms endpoints, not only 50ms control endpoints, are constrained.
    This diagnostic does not model contacts, torque saturation or nonlinear M/J.
    """
    seed, previous = finite(seed), finite(previous, (6,))
    if seed.ndim != 2 or seed.shape[1] != 6 or not 1 <= len(seed) <= 8:
        raise ValueError("One to eight seed commands required")
    lower, upper = finite(lower, (len(model["q"]),)), finite(upper, (len(model["q"]),))
    if np.any(lower + 0.05 >= upper - 0.05):
        raise ValueError("Joint ranges must contain requested .05rad interior")
    h, width = len(seed), seed.size
    lo, hi = np.tile(interface.input_min, h), np.tile(interface.input_max, h)
    d = np.eye(width) - np.eye(width, k=-6)
    center = np.r_[previous, np.zeros(width - 6)]
    if (
        np.any(seed.ravel() < lo - 1e-12)
        or np.any(seed.ravel() > hi + 1e-12)
        or np.max(abs(d @ seed.ravel() - center)) > 0.25 + 1e-12
    ):
        raise ValueError("Box/rate feasible seed required")
    scale = np.tile((interface.output_max - interface.output_min) / (interface.input_max - interface.input_min), h)
    cache = {}

    class Stop(RuntimeError):
        pass

    def evaluate(x):
        key = tuple(x)
        if key not in cache:
            if len(cache) >= 160:
                raise Stop("response_budget")
            delta = np.asarray([interface.decode(c) for c in np.asarray(x).reshape(h, 6)])
            if np.any(np.all(delta[:, 3:] == 0, axis=1)):
                raise Stop("unsupported_rotation_latch")
            rollout = rollout_joint_preview(model, velocity, delta, site, lower, upper, enforce_limits=False)
            cache[key] = dict(rollout=rollout, score=preview_objective(rollout, tool, targets, twists))
        return cache[key]

    original = evaluate(seed.ravel())
    q = original["rollout"]["inner_joint_position"].ravel()
    a = original["rollout"]["inner_joint_position_jacobian"].reshape(len(q), width) * scale
    offset = q - a @ seed.ravel()
    joint_lo = np.tile(lower + 0.05, 25 * h) - offset
    joint_hi = np.tile(upper - 0.05, 25 * h) - offset
    first = original["score"]["residual"][:6] / np.array([100.0] * 3 + [20.0] * 3)
    caps = np.maximum([np.linalg.norm(first[:3]), np.linalg.norm(first[3:])], [0.01, 0.05])
    if terminal_normals is None:
        inner = original["rollout"]["inner_joint_position"]
        identity = np.eye(len(lower))
        terminal_normals = np.vstack(
            (identity[np.any(inner >= upper - 0.05, axis=0)], -identity[np.any(inner <= lower + 0.05, axis=0)])
        )
    normals = finite(terminal_normals)
    if normals.ndim != 2 or normals.shape[1] != len(lower):
        raise ValueError("Finite terminal joint direction rows required")
    terminal_jac = normals @ original["rollout"]["joint_velocity_jacobian"][-1] * scale[None, :]
    terminal_offset = normals @ original["rollout"]["joint_velocity"][-1] - terminal_jac @ seed.ravel()
    base = dict(
        seed_commands=seed.copy(),
        first_pose_caps=caps,
        joint_margin_rad=0.05,
        terminal_normals=normals.copy(),
        initial_margin_rad=float(np.min(np.minimum(model["q"] - lower, upper - model["q"]))),
        scope="future2ms .05 interior reentry witness only; initial buffer not qualified and no physical guarantee",
        initial_inside_requested_buffer=bool(np.all(model["q"] >= lower + 0.05) and np.all(model["q"] <= upper - 0.05)),
    )
    if base["initial_margin_rad"] < 0.0:
        return dict(
            **base,
            status="initial_outside_xml",
            witness_found=False,
            response_evaluations=len(cache),
            optimizer_success=False,
        )
    lp = linprog(
        np.zeros(width),
        A_ub=np.vstack((a, -a, d, -d, terminal_jac)),
        b_ub=np.r_[joint_hi, -joint_lo, center + 0.25, -center + 0.25, -terminal_offset],
        bounds=list(zip(lo, hi, strict=True)),
        method="highs",
    )
    base.update(linear_status=int(lp.status), linear_message=str(lp.message))
    if not lp.success:
        return dict(
            **base,
            status="linear_solver_no_witness",
            witness_found=False,
            response_evaluations=len(cache),
            optimizer_success=False,
        )
    base["linear_witness_commands"] = lp.x.reshape(h, 6).copy()

    def objective(x):
        score = evaluate(x)["score"]
        return score["objective"], score["gradient"] * scale

    start = seed.ravel().copy()
    if initial_guess is not None:
        guess = finite(initial_guess, seed.shape)
        x = guess.ravel()
        if np.any(x < lo - 1e-12) or np.any(x > hi + 1e-12) or np.max(abs(d @ x - center)) > 0.25 + 1e-12:
            raise ValueError("Warm start must satisfy current command box/rate")
        try:
            evaluate(x)  # Preserve a checked feasible warm candidate even if later search exhausts budget.
        except Stop as error:
            raise ValueError("Warm start uses unsupported latch mode") from error
        start = x.copy()
    base["initial_guess_used"] = initial_guess is not None
    success, iterations, message = False, 0, "not_run"
    try:
        # Original TCP caps remain comparison metrics. The original whole-plan
        # tracking objective and joint/terminal constraints are unchanged.
        result = minimize(
            objective,
            start,
            jac=True,
            method="SLSQP",
            bounds=Bounds(lo, hi),
            constraints=[
                LinearConstraint(d, center - 0.25, center + 0.25),
                LinearConstraint(a, joint_lo, joint_hi),
            ]
            + ([LinearConstraint(terminal_jac, -np.inf, -terminal_offset)] if len(normals) else []),
            options=dict(maxiter=60, ftol=1e-9),
        )
        success, iterations, message = bool(result.success), int(result.nit), str(result.message)
    except Stop as error:
        message = str(error)
    best = None
    for key, item in list(cache.items()):
        x = np.asarray(key)
        if (
            np.all(x >= lo - 1e-12)
            and np.all(x <= hi + 1e-12)
            and np.max(abs(d @ x - center)) <= 0.25 + 1e-12
            and np.all(a @ x >= joint_lo - 1e-10)
            and np.all(a @ x <= joint_hi + 1e-10)
            and np.all(terminal_jac @ x + terminal_offset <= 1e-10)
        ):
            if best is None or item["score"]["objective"] < best[1]["score"]["objective"]:
                best = (x, item)
    base.update(
        optimizer_success=success,
        iterations=iterations,
        optimizer_message=message,
        response_evaluations=len(cache),
        witness_found=best is not None,
    )
    if best is None:
        return dict(**base, status="nonlinear_search_no_witness")
    x, item = best
    actual = item["rollout"]["inner_joint_position"]
    np.testing.assert_allclose(actual.ravel(), a @ x + offset, atol=1e-10, rtol=0)
    return dict(
        **base,
        status="model_witness",
        planned_commands=x.reshape(h, 6).copy(),
        first_tcp_error=item["score"]["residual"][:6] / np.array([100.0] * 3 + [20.0] * 3),
        original_first_pose_caps_satisfied=bool(
            np.linalg.norm(item["score"]["residual"][:3]) / 100.0 <= caps[0] + 1e-10
            and np.linalg.norm(item["score"]["residual"][3:6]) / 20.0 <= caps[1] + 1e-10
        ),
        first_pose_caps_enforced=False,
        objective=item["score"]["objective"],
        seed_objective=original["score"]["objective"],
        minimum_inner_margin_rad=float(np.min(np.minimum(actual - lower, upper - actual))),
        terminal_signed_velocities_rad_s=normals @ item["rollout"]["joint_velocity"][-1],
        rollout=item["rollout"],
    )
