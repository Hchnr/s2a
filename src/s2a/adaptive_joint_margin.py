"""Search a retained model margin without relaxing original first-pose limits."""

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, minimize

from s2a.geometry import finite
from s2a.interior_joint_preview import rollout_joint_preview
from s2a.joint_limit_preview_objective import preview_objective


def maximize_margin(
    model, velocity, seed, previous, interface, site, tool, targets, twists, lower, upper, *, initial_guess=None
):
    """Return a feasible margin witness, not a global nonlinear optimality claim."""
    seed, previous = finite(seed), finite(previous, (6,))
    if seed.ndim != 2 or seed.shape[1] != 6 or not 1 <= len(seed) <= 8:
        raise ValueError("One to eight seed commands required")
    n = len(model["q"])
    lower, upper = finite(lower, (n,)), finite(upper, (n,))
    if np.any(lower >= upper):
        raise ValueError("Strict joint ranges required")
    h, width = len(seed), seed.size
    lo, hi = np.tile(interface.input_min, h), np.tile(interface.input_max, h)
    d = np.eye(width) - np.eye(width, k=-6)
    center = np.r_[previous, np.zeros(width - 6)]

    def command_feasible(x):
        return bool(np.all(x >= lo - 1e-12) and np.all(x <= hi + 1e-12) and np.max(abs(d @ x - center)) <= 0.25 + 1e-12)

    if not command_feasible(seed.ravel()):
        raise ValueError("Feasible original seed required")
    scale = np.tile((interface.output_max - interface.output_min) / (interface.input_max - interface.input_min), h)
    cache = {}
    trials = []

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
            r = rollout_joint_preview(model, velocity, delta, site, lower, upper, enforce_limits=False)
            cache[key] = dict(rollout=r, score=preview_objective(r, tool, targets, twists))
        return cache[key]

    original = evaluate(seed.ravel())
    first = original["score"]["residual"][:6] / np.array([100.0] * 3 + [20.0] * 3)
    caps = np.maximum([np.linalg.norm(first[:3]), np.linalg.norm(first[3:])], [0.01, 0.05])
    current_margin = float(np.min(np.minimum(model["q"] - lower, upper - model["q"])))
    base = dict(
        first_pose_caps=caps,
        requested_margin_rad=0.05,
        current_margin_rad=current_margin,
        scope="frozen free model margin witness with original first-pose caps; no global optimum or physical guarantee",
    )
    if current_margin < 0:
        return dict(**base, status="initial_outside_xml", witness_found=False)
    max_margin = min(0.05, current_margin)
    q = original["rollout"]["inner_joint_position"].ravel()
    a = original["rollout"]["inner_joint_position_jacobian"].reshape(len(q), width) * scale
    offset = q - a @ seed.ravel()
    joint = np.vstack((np.c_[a, np.ones(len(q))], np.c_[-a, np.ones(len(q))]))
    bound = np.r_[np.tile(upper, 25 * h) - offset, offset - np.tile(lower, 25 * h)]
    rate = np.c_[d, np.zeros(width)]
    bounds = Bounds(np.r_[lo, 0.0], np.r_[hi, max_margin])
    lp = linprog(
        np.r_[np.zeros(width), -1.0],
        A_ub=np.vstack((joint, rate, -rate)),
        b_ub=np.r_[bound, center + 0.25, -center + 0.25],
        bounds=list(zip(bounds.lb, bounds.ub, strict=True)),
        method="highs",
    )
    base.update(linear_status=int(lp.status), linear_message=str(lp.message))
    if not lp.success:
        return dict(**base, status="linear_search_no_witness", witness_found=False)
    base["linear_margin_upper_bound_rad"] = float(lp.x[-1])
    if initial_guess is None:
        start = lp.x.copy()
    else:
        x = finite(initial_guess, seed.shape).ravel()
        if not command_feasible(x):
            raise ValueError("Warm command box/rate invalid")
        item = evaluate(x)
        inner = item["rollout"]["inner_joint_position"]
        margin = float(np.min(np.minimum(inner - lower, upper - inner)))
        start = np.r_[x, max(0.0, min(max_margin, margin))]

    def pose_values(z):
        trials.append(tuple(z))
        e = evaluate(z[:-1])["score"]["residual"][:6] / np.array([100.0] * 3 + [20.0] * 3)
        return np.asarray([1 - np.linalg.norm(e[:3]) / caps[0], 1 - np.linalg.norm(e[3:]) / caps[1]])

    def pose_jac(z):
        item = evaluate(z[:-1])["score"]
        e = item["residual"][:6] / np.array([100.0] * 3 + [20.0] * 3)
        jac = item["jacobian"][:6] * scale[None, :] / np.array([100.0] * 3 + [20.0] * 3)[:, None]
        rows = []
        for g, (start, end) in enumerate(((0, 3), (3, 6))):
            norm = np.linalg.norm(e[start:end])
            rows.append(np.r_[-(e[start:end] / norm if norm > 1e-15 else np.zeros(3)) @ jac[start:end] / caps[g], 0.0])
        return np.asarray(rows)

    def objective(z):
        trials.append(tuple(z))
        return -float(z[-1]), np.r_[np.zeros(width), -1.0]

    success, iterations, message = False, 0, "not_run"
    try:
        evaluate(start[:-1])
        trials.append(tuple(start))
        result = minimize(
            objective,
            start,
            jac=True,
            method="SLSQP",
            bounds=bounds,
            constraints=[
                LinearConstraint(joint, -np.inf, bound),
                LinearConstraint(rate, center - 0.25, center + 0.25),
                dict(type="ineq", fun=pose_values, jac=pose_jac),
            ],
            options=dict(maxiter=60, ftol=1e-10),
        )
        success, iterations, message = bool(result.success), int(result.nit), str(result.message)
    except Stop as error:
        message = str(error)
    best = None
    for entry in sorted(set(trials)):
        z = np.asarray(entry)
        if tuple(z[:-1]) not in cache:
            continue
        if (
            not command_feasible(z[:-1])
            or z[-1] < -1e-12
            or z[-1] > max_margin + 1e-12
            or np.any(joint @ z > bound + 1e-10)
        ):
            continue
        item = cache[tuple(z[:-1])]
        e = item["score"]["residual"][:6] / np.array([100.0] * 3 + [20.0] * 3)
        if np.linalg.norm(e[:3]) > caps[0] + 1e-10 or np.linalg.norm(e[3:]) > caps[1] + 1e-10:
            continue
        if (
            best is None
            or z[-1] > best[0][-1] + 1e-12
            or (abs(z[-1] - best[0][-1]) <= 1e-12 and item["score"]["objective"] < best[1]["score"]["objective"])
        ):
            best = (z, item)
    base.update(
        optimizer_success=success, iterations=iterations, optimizer_message=message, response_evaluations=len(cache)
    )
    if best is None:
        return dict(**base, status="nonlinear_search_no_witness", witness_found=False)
    z, item = best
    inner = item["rollout"]["inner_joint_position"]
    actual_margin = float(np.min(np.minimum(inner - lower, upper - inner)))
    return dict(
        **base,
        status="model_margin_witness",
        witness_found=True,
        planned_commands=z[:-1].reshape(h, 6),
        retained_margin_rad=float(z[-1]),
        minimum_inner_margin_rad=actual_margin,
        requested_margin_satisfied=bool(actual_margin >= 0.05 - 1e-10),
        tracking_objective=item["score"]["objective"],
        rollout=item["rollout"],
    )
