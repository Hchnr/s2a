"""Supply a checked linear-feasible start to the frozen first-hard solver."""

from types import FunctionType

import numpy as np
from scipy.optimize import linprog

from s2a.geometry import finite
from s2a.research.joint_control.preview.interior_joint_preview import rollout_joint_preview
from s2a.research.trajectory_search import first_hard_restoration as original

_ORIGINAL_SOLVER = original.solve_first_hard_restoration
_ORIGINAL_ROLLOUT = rollout_joint_preview


def linear_domain(model, seed, previous, interface, lower, upper, error, rollout, terminal_normals=None):
    """Reconstruct exactly the frozen solver's affine joint and action rows."""
    seed, previous = finite(seed, (8, 6)), finite(previous, (6,))
    lower, upper, error = finite(lower, (7,)), finite(upper, (7,)), finite(error, (25, 7))
    if np.any(error < 0) or np.any(np.diff(error, axis=0) < 0):
        raise ValueError("Only nonnegative completed inner error is allowed")
    peak = error.max(axis=0)
    if np.any(lower + 0.05 + peak >= upper - 0.05 - peak):
        raise ValueError("Joint ranges must contain the original tightened interior")
    margin = np.tile(0.05 + peak, (200, 1))
    margin[:25] = 0.05 + error
    seed_x = seed.ravel()
    lo, hi = np.tile(interface.input_min, 8), np.tile(interface.input_max, 8)
    rate = np.eye(48) - np.eye(48, k=-6)
    center = np.r_[previous, np.zeros(42)]
    scale = np.tile((interface.output_max - interface.output_min) / (interface.input_max - interface.input_min), 8)
    q = finite(rollout["inner_joint_position"], (200, 7)).ravel()
    joint_a = finite(rollout["inner_joint_position_jacobian"], (200, 7, 48)).reshape(len(q), 48) * scale
    joint_offset = q - joint_a @ seed_x
    joint_lo, joint_hi = (lower + margin).ravel() - joint_offset, (upper - margin).ravel() - joint_offset
    brake = finite(rollout["joint_position"], (8, 7)) + 0.1 * finite(rollout["joint_velocity"], (8, 7))
    brake_jac = finite(rollout["joint_position_jacobian"], (8, 7, 48)) + 0.1 * finite(
        rollout["joint_velocity_jacobian"], (8, 7, 48)
    )
    brake_a = brake_jac.reshape(brake.size, 48) * scale
    brake_offset = brake.ravel() - brake_a @ seed_x
    joint_a = np.vstack((joint_a, brake_a))
    joint_lo = np.r_[joint_lo, np.tile(lower + 0.05, 8) - brake_offset]
    joint_hi = np.r_[joint_hi, np.tile(upper - 0.05, 8) - brake_offset]
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
    return dict(
        lo=lo,
        hi=hi,
        rate=rate,
        center=center,
        joint_a=joint_a,
        joint_lo=joint_lo,
        joint_hi=joint_hi,
        terminal_a=terminal_a,
        terminal_offset=terminal_offset,
        terminal_normals=normals,
    )


def domain_check(commands, domain, interface):
    """Check the same tolerances as the original solver's candidate filter."""
    x = finite(commands, (8, 6)).ravel()
    lo, hi, rate, center = (domain[key] for key in ("lo", "hi", "rate", "center"))
    a, joint_lo, joint_hi = (domain[key] for key in ("joint_a", "joint_lo", "joint_hi"))
    terminal_a, terminal_offset = domain["terminal_a"], domain["terminal_offset"]
    box = bool(np.all(x >= lo - 1e-12) and np.all(x <= hi + 1e-12))
    rate_ok = bool(np.max(abs(rate @ x - center)) <= 0.25 + 1e-12)
    joint_ok = bool(np.all(a @ x >= joint_lo - 1e-10) and np.all(a @ x <= joint_hi + 1e-10))
    terminal_ok = bool(np.all(terminal_a @ x + terminal_offset <= 1e-10))
    latch = [index for index, command in enumerate(x.reshape(8, 6)) if np.all(interface.decode(command)[3:] == 0)]
    return dict(
        box=box,
        rate=rate_ok,
        joint_and_braking=joint_ok,
        terminal=terminal_ok,
        zero_rotation_periods=latch,
        accepted=bool(box and rate_ok and joint_ok and terminal_ok and not latch),
    )


def nearest_linear_plan(seed, domain, interface, *, max_lp=2):
    """L1-project to original linear rows, with one optional latch repair LP."""
    seed = finite(seed, (8, 6))
    if max_lp != 2:
        raise ValueError("The registered initializer permits at most two LPs")
    if domain_check(seed, domain, interface)["accepted"]:
        return seed.copy(), dict(seed_linear_safe=True, start_lp_attempts=0, latch_repair=False, warm_start_used=False)
    x0 = seed.ravel()
    a, lo, hi, rate, center = (domain[key] for key in ("joint_a", "joint_lo", "joint_hi", "rate", "center"))
    terminal_a, terminal_offset = domain["terminal_a"], domain["terminal_offset"]
    base_a = np.vstack((a, -a, rate, -rate, terminal_a))
    base_b = np.r_[hi, -lo, center + 0.25, -center + 0.25, -terminal_offset]
    identity = np.eye(48)
    all_a = np.vstack(
        (np.c_[base_a, np.zeros((len(base_a), 48))], np.c_[identity, -identity], np.c_[-identity, -identity])
    )
    all_b = np.r_[base_b, x0, -x0]
    objective = np.r_[np.zeros(48), np.ones(48)]
    bounds = list(zip(domain["lo"], domain["hi"], strict=True))
    attempts = []

    def solve(current_bounds):
        result = linprog(
            objective,
            A_ub=all_a,
            b_ub=all_b,
            bounds=current_bounds + [(0, None)] * 48,
            method="highs",
            options={"primal_feasibility_tolerance": 1e-10, "dual_feasibility_tolerance": 1e-10},
        )
        attempts.append(dict(status=int(result.status), message=str(result.message), success=bool(result.success)))
        return result

    first = solve(bounds)
    if not first.success:
        return None, dict(
            seed_linear_safe=False,
            start_lp_attempts=len(attempts),
            lp_results=attempts,
            latch_repair=False,
            warm_start_used=False,
            reason="linear_start_lp_failed",
        )
    candidate = np.asarray(first.x[:48]).reshape(8, 6)
    check = domain_check(candidate, domain, interface)
    repaired, sign_anchors = False, []
    if check["zero_rotation_periods"]:
        repaired = True
        modified = list(bounds)
        physical_scale = (interface.output_max - interface.output_min) / (interface.input_max - interface.input_min)
        for period, seed_command in enumerate(seed):
            physical = interface.decode(seed_command)[3:]
            axis = int(np.argmax(abs(physical)))
            value = float(physical[axis])
            threshold = min(1e-8, abs(value) * 0.5)
            if threshold == 0:
                return None, dict(
                    seed_linear_safe=False,
                    start_lp_attempts=len(attempts),
                    lp_results=attempts,
                    latch_repair=True,
                    warm_start_used=False,
                    reason="seed_rotation_anchor_unavailable",
                )
            component, sign = 3 + axis, 1 if value > 0 else -1
            encoded = (
                interface.input_min[component]
                + (sign * threshold - interface.output_min[component]) / physical_scale[component]
            )
            index = 6 * period + component
            lower, upper = modified[index]
            modified[index] = (max(lower, encoded), upper) if sign > 0 else (lower, min(upper, encoded))
            if modified[index][0] > modified[index][1]:
                return None, dict(
                    seed_linear_safe=False,
                    start_lp_attempts=len(attempts),
                    lp_results=attempts,
                    latch_repair=True,
                    warm_start_used=False,
                    reason="rotation_anchor_outside_box",
                )
            sign_anchors.append(
                dict(period=period, component=component, sign=sign, minimum_physical_magnitude=threshold)
            )
        second = solve(modified)
        if not second.success:
            return None, dict(
                seed_linear_safe=False,
                start_lp_attempts=len(attempts),
                lp_results=attempts,
                latch_repair=True,
                sign_anchors=sign_anchors,
                warm_start_used=False,
                reason="rotation_repair_lp_failed",
            )
        candidate = np.asarray(second.x[:48]).reshape(8, 6)
        check = domain_check(candidate, domain, interface)
    if not check["accepted"]:
        return None, dict(
            seed_linear_safe=False,
            start_lp_attempts=len(attempts),
            lp_results=attempts,
            latch_repair=repaired,
            sign_anchors=sign_anchors,
            warm_start_used=False,
            check=check,
            reason="start_failed_strict_domain_check",
        )
    return candidate.copy(), dict(
        seed_linear_safe=False,
        start_lp_attempts=len(attempts),
        lp_results=attempts,
        latch_repair=repaired,
        sign_anchors=sign_anchors,
        warm_start_used=True,
        check=check,
        warm_start_commands=candidate.copy(),
    )


def solve_feasible_start_restoration(
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
    feasibility_restoration=True,
    enforce_braking=True,
):
    """Add a linear-feasible start without increasing the 320-response budget."""
    if initial_guess is not None or feasibility_restoration is not True or enforce_braking is not True:
        raise ValueError("Only registered restoration with an internally chosen start is supported")
    if not np.isclose(braking_time_s, 0.1, atol=0, rtol=0):
        raise ValueError("The original braking constraint is required")
    if (
        original.solve_first_hard_restoration is not _ORIGINAL_SOLVER
        or _ORIGINAL_SOLVER.__globals__.get("rollout_joint_preview") is not _ORIGINAL_ROLLOUT
    ):
        raise ValueError("The frozen first-hard solver and rollout must remain unchanged")
    seed = finite(seed, (8, 6))
    velocity, previous = finite(velocity, (7,)), finite(previous, (6,))
    lower, upper = finite(lower, (7,)), finite(upper, (7,))
    lo, hi = np.tile(interface.input_min, 8), np.tile(interface.input_max, 8)
    rate = np.eye(48) - np.eye(48, k=-6)
    center = np.r_[previous, np.zeros(42)]
    if (
        np.any(seed.ravel() < lo - 1e-12)
        or np.any(seed.ravel() > hi + 1e-12)
        or np.max(abs(rate @ seed.ravel() - center)) > 0.25 + 1e-12
    ):
        raise ValueError("The original seed must satisfy command box and rate")
    error = np.zeros((25, 7)) if past_inner_error_rad is None else finite(past_inner_error_rad, (25, 7))
    physical = np.asarray([interface.decode(row) for row in seed])
    if np.any(np.all(physical[:, 3:] == 0, axis=1)):
        raise ValueError("The original seed uses an unsupported rotation latch")
    seed_rollout = rollout_joint_preview(model, velocity, physical, site, lower, upper, enforce_limits=False)
    domain = linear_domain(model, seed, previous, interface, lower, upper, error, seed_rollout, terminal_normals)
    if np.min(np.minimum(model["q"] - lower, upper - model["q"])) < 0.05 - 1e-12:
        start, info = (
            seed.copy(),
            dict(
                seed_linear_safe=False,
                start_lp_attempts=0,
                latch_repair=False,
                warm_start_used=False,
                reason="initial_outside_requested_interior",
            ),
        )
    else:
        start, info = nearest_linear_plan(seed, domain, interface)
    if start is None:
        info.update(seed_rollout_reused=False, actual_model_rollouts=1, solver_lp_attempted=False)
        return dict(
            status="feasible_start_unavailable", witness_found=False, response_evaluations=1, initialization=info
        )

    calls = []

    def cached_rollout(
        current_model, current_velocity, current_delta, current_site, current_lower, current_upper, *, enforce_limits
    ):
        if not calls:
            if not (
                current_model is model
                and np.array_equal(current_velocity, velocity)
                and np.array_equal(current_delta, physical)
                and np.array_equal(current_site, site)
                and np.array_equal(current_lower, lower)
                and np.array_equal(current_upper, upper)
                and enforce_limits is False
            ):
                raise ValueError("The first frozen response must match the cached seed exactly")
            calls.append("seed_reused")
            return seed_rollout
        calls.append("computed")
        return rollout_joint_preview(
            current_model,
            current_velocity,
            current_delta,
            current_site,
            current_lower,
            current_upper,
            enforce_limits=enforce_limits,
        )

    namespace = _ORIGINAL_SOLVER.__globals__.copy()
    namespace["rollout_joint_preview"] = cached_rollout
    solver = FunctionType(
        _ORIGINAL_SOLVER.__code__,
        namespace,
        _ORIGINAL_SOLVER.__name__,
        _ORIGINAL_SOLVER.__defaults__,
        _ORIGINAL_SOLVER.__closure__,
    )
    solver.__kwdefaults__ = dict(_ORIGINAL_SOLVER.__kwdefaults__ or {})
    result = solver(
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
        initial_guess=start if info["warm_start_used"] else None,
        past_inner_error_rad=error,
        terminal_normals=terminal_normals,
        braking_time_s=braking_time_s,
        feasibility_restoration=True,
        enforce_braking=True,
    )
    actual_rollouts = 1 + sum(item == "computed" for item in calls)
    if calls[:1] != ["seed_reused"] or actual_rollouts != result["response_evaluations"] or actual_rollouts > 320:
        raise ValueError("Cached seed and actual unique model response accounting disagree")
    if (
        original.solve_first_hard_restoration is not _ORIGINAL_SOLVER
        or _ORIGINAL_SOLVER.__globals__.get("rollout_joint_preview") is not _ORIGINAL_ROLLOUT
    ):
        raise ValueError("The original solver globals changed during the isolated call")
    info.update(
        seed_rollout_reused=True, actual_model_rollouts=actual_rollouts, solver_lp_attempted="linear_status" in result
    )
    result["initialization"] = info
    return result
