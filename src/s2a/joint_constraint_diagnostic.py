"""Frozen affine constraint attribution; removed constraints are diagnostic only."""

import numpy as np
from scipy.optimize import linprog

from s2a.geometry import finite
from s2a.interior_joint_preview import rollout_joint_preview


def diagnose(model, velocity, seed, previous, interface, site, lower, upper, error):
    """Separate constraint families and bound best first-period model clearance.

    Feasible relaxed LPs are not executable plans. Infeasibility and clearance
    apply only to this frozen model, without a physical safety claim.
    """
    seed = finite(seed)
    previous = finite(previous, (6,))
    lower, upper, error = (finite(v, (len(model["q"]),)) for v in (lower, upper, error))
    if seed.ndim != 2 or seed.shape[1] != 6 or not 1 <= len(seed) <= 8 or np.any(error < 0):
        raise ValueError("Valid seed and nonnegative error required")
    h, width = len(seed), seed.size
    delta = np.asarray([interface.decode(c) for c in seed])
    if np.any(np.all(delta[:, 3:] == 0, axis=1)):
        raise ValueError("Supported reset seed required")
    rollout = rollout_joint_preview(model, velocity, delta, site, lower, upper, enforce_limits=False)
    scale = np.tile((interface.output_max - interface.output_min) / (interface.input_max - interface.input_min), h)
    q = rollout["inner_joint_position"].ravel()
    a = rollout["inner_joint_position_jacobian"].reshape(len(q), width) * scale
    offset = q - a @ seed.ravel()
    margin = 0.05 + error
    groups = {
        "position": (
            np.vstack((a, -a)),
            np.r_[np.tile(upper - margin, 25 * h) - offset, offset - np.tile(lower + margin, 25 * h)],
        )
    }
    braking = rollout["joint_position"] + 0.1 * rollout["joint_velocity"]
    ba = (rollout["joint_position_jacobian"] + 0.1 * rollout["joint_velocity_jacobian"]).reshape(-1, width) * scale
    boffset = braking.ravel() - ba @ seed.ravel()
    groups["braking"] = (
        np.vstack((ba, -ba)),
        np.r_[np.tile(upper - 0.05, h) - boffset, boffset - np.tile(lower + 0.05, h)],
    )
    d = np.eye(width) - np.eye(width, k=-6)
    center = np.r_[previous, np.zeros(width - 6)]
    groups["rate"] = (np.vstack((d, -d)), np.r_[center + 0.25, -center + 0.25])
    identity = np.eye(len(lower))
    normals = np.vstack(
        (
            identity[np.any(rollout["inner_joint_position"] >= upper - 0.05, axis=0)],
            -identity[np.any(rollout["inner_joint_position"] <= lower + 0.05, axis=0)],
        )
    )
    ta = normals @ rollout["joint_velocity_jacobian"][-1] * scale
    toffset = normals @ rollout["joint_velocity"][-1] - ta @ seed.ravel()
    groups["terminal"] = (ta, -toffset)
    bounds = list(zip(np.tile(interface.input_min, h), np.tile(interface.input_max, h), strict=True))

    def solve(names):
        active = [groups[name] for name in names]
        lp = linprog(
            np.zeros(width),
            A_ub=np.vstack([v[0] for v in active]),
            b_ub=np.concatenate([v[1] for v in active]),
            bounds=bounds,
            method="highs",
        )
        return dict(status=int(lp.status), success=bool(lp.success), message=str(lp.message))

    checks = {"all": solve(groups)}
    for name in groups:
        checks["without_" + name] = solve([k for k in groups if k != name])
    checks["position_and_rate"] = solve(["position", "rate"])
    checks["position_only"] = solve(["position"])

    # Even discard every future-period, braking and terminal constraint here.
    # The optimum bounds attainable FIRST-period clearance under box and rate.
    n = 25 * len(lower)
    fa, fo = a[:n], offset[:n]
    rate_a, rate_b = groups["rate"]
    ma = np.vstack(
        (
            np.c_[fa, np.ones(n)],
            np.c_[-fa, np.ones(n)],
            np.c_[rate_a, np.zeros(len(rate_b))],
        )
    )
    mb = np.r_[np.tile(upper - error, 25) - fo, fo - np.tile(lower + error, 25), rate_b]
    lp = linprog(np.r_[np.zeros(width), -1.0], A_ub=ma, b_ub=mb, bounds=bounds + [(None, None)], method="highs")
    best = dict(status=int(lp.status), success=bool(lp.success), message=str(lp.message))
    if lp.success:
        violation = float(np.max(ma @ lp.x - mb))
        if violation > 1e-8:
            raise ValueError("Clearance LP witness exceeds diagnostic tolerance")
        best.update(
            maximum_first_period_error_adjusted_margin_rad=float(lp.x[-1]),
            maximum_linear_violation=violation,
            commands=lp.x[:-1].reshape(h, 6),
            executable_plan=False,
        )
    return dict(
        constraint_checks=checks,
        first_period_clearance=best,
        scope="Frozen affine diagnostic only; no removed constraint authorized for execution",
    )
