"""Explicit interior joint constraints under the frozen free OSC model."""

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, minimize

from s2a.geometry import finite
from s2a.interior_joint_preview import rollout_joint_preview
from s2a.joint_limit_preview_objective import preview_objective


def solve_interior_plan(model, velocity, seed, previous, interface, site, tool, targets, twists, lower, upper):
    """Find a model witness, never interpret failed nonlinear search as impossibility.

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
    base = dict(
        seed_commands=seed.copy(),
        first_pose_caps=caps,
        joint_margin_rad=0.05,
        initial_margin_rad=float(np.min(np.minimum(model["q"] - lower, upper - model["q"]))),
        scope="frozen free model interior feasibility only; no physical control qualification",
    )
    if base["initial_margin_rad"] < 0.05 - 1e-12:
        return dict(
            **base,
            status="initial_outside_requested_interior",
            witness_found=False,
            response_evaluations=len(cache),
            optimizer_success=False,
        )
    lp = linprog(
        np.zeros(width),
        A_ub=np.vstack((a, -a, d, -d)),
        b_ub=np.r_[joint_hi, -joint_lo, center + 0.25, -center + 0.25],
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

    def pose_values(x):
        e = evaluate(x)["score"]["residual"][:6] / np.array([100.0] * 3 + [20.0] * 3)
        return np.array([1 - np.linalg.norm(e[:3]) / caps[0], 1 - np.linalg.norm(e[3:]) / caps[1]])

    def pose_jac(x):
        item = evaluate(x)["score"]
        e = item["residual"][:6] / np.array([100.0] * 3 + [20.0] * 3)
        jac = item["jacobian"][:6] * scale[None, :] / np.array([100.0] * 3 + [20.0] * 3)[:, None]
        rows = []
        for g, (start, end) in enumerate(((0, 3), (3, 6))):
            norm = np.linalg.norm(e[start:end])
            rows.append(-(e[start:end] / norm if norm > 1e-15 else np.zeros(3)) @ jac[start:end] / caps[g])
        return np.asarray(rows)

    def objective(x):
        score = evaluate(x)["score"]
        return score["objective"], score["gradient"] * scale

    success, iterations, message = False, 0, "not_run"
    try:
        # The linear witness may use latch commands. Start the supported seed;
        # the linear constraints still guide search toward the feasible region.
        result = minimize(
            objective,
            seed.ravel(),
            jac=True,
            method="SLSQP",
            bounds=Bounds(lo, hi),
            constraints=[
                LinearConstraint(d, center - 0.25, center + 0.25),
                LinearConstraint(a, joint_lo, joint_hi),
                dict(type="ineq", fun=pose_values, jac=pose_jac),
            ],
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
            and np.all(pose_values(x) >= -1e-8)
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
        objective=item["score"]["objective"],
        seed_objective=original["score"]["objective"],
        minimum_inner_margin_rad=float(np.min(np.minimum(actual - lower, upper - actual))),
        rollout=item["rollout"],
    )
