"""Experimental same-state search for a second-pose refresh margin.

This diagnostic does not dispatch commands or establish physical safety. Its
separate response budget is additional to the original baseline search.
"""

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize

from s2a.geometry import finite
from s2a.research.joint_control.limits.joint_limit_preview_objective import preview_objective
from s2a.research.joint_control.preview.interior_joint_preview import rollout_joint_preview
from s2a.research.trajectory_search.feasible_start_restoration import domain_check, linear_domain
from s2a.research.trajectory_search.full_horizon_quality import pose_quality_constraints
from s2a.research.trajectory_search.margin_two_step_search import (
    MAX_FUTURE_SLACK_INCREASE,
    NORMALIZED_MARGIN,
    POSITION_CAP_M,
    REFRESH_POSITION_LIMIT_M,
    REFRESH_ROTATION_LIMIT_RAD,
    ROTATION_CAP_RAD,
    SELECTION_TOLERANCE,
    margin_prefix_satisfied,
)

MAX_RESPONSES = 320
MAX_ITERATIONS = 120


class ResponseBudgetExhausted(RuntimeError):
    """The separately registered local-model response budget is exhausted."""


def targeted_constraints(values, jacobian, baseline_future_slack):
    """Apply the original first pose, tightened second pose, and future bound."""
    values, jacobian = finite(values, (16,)), finite(jacobian)
    if jacobian.ndim != 2 or jacobian.shape[0] != 16:
        raise ValueError("Sixteen normalized quality rows and their Jacobian are required")
    if not np.isfinite(baseline_future_slack) or baseline_future_slack < 0:
        raise ValueError("A finite nonnegative baseline future slack is required")
    floor = np.array(
        [
            NORMALIZED_MARGIN[0],
            NORMALIZED_MARGIN[1],
            1.0 - REFRESH_POSITION_LIMIT_M / POSITION_CAP_M,
            1.0 - REFRESH_ROTATION_LIMIT_RAD / ROTATION_CAP_RAD,
        ]
    )
    return np.r_[values[:4] - floor, values[4:] + baseline_future_slack + MAX_FUTURE_SLACK_INCREASE], jacobian.copy()


def candidate_is_eligible(errors, future_slack, baseline_future_slack, safety):
    """Recheck actual evaluated errors and the original linear safety domain."""
    errors = finite(errors, (8, 6))
    if (
        not np.isfinite(future_slack)
        or future_slack < 0
        or not np.isfinite(baseline_future_slack)
        or baseline_future_slack < 0
    ):
        raise ValueError("Finite nonnegative future quality values are required")
    return bool(
        safety.get("accepted") is True
        and margin_prefix_satisfied(errors)
        and np.linalg.norm(errors[1, :3]) <= REFRESH_POSITION_LIMIT_M + SELECTION_TOLERANCE
        and np.linalg.norm(errors[1, 3:]) <= REFRESH_ROTATION_LIMIT_RAD + SELECTION_TOLERANCE
        and future_slack <= baseline_future_slack + MAX_FUTURE_SLACK_INCREASE + SELECTION_TOLERANCE
    )


def solve_targeted_refresh(
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
    past_inner_error_rad,
    expected_seed_tcp_errors,
    expected_baseline_future_slack,
):
    """Search one frozen state; return only evaluated, rechecked model candidates."""
    seed = finite(seed, (8, 6)).copy()
    previous = finite(previous, (6,))
    lower, upper = finite(lower, (7,)), finite(upper, (7,))
    past = finite(past_inner_error_rad, (25, 7))
    expected_errors = finite(expected_seed_tcp_errors, (8, 6))
    targets = finite(targets, (8, 4, 4))
    seed_x = seed.ravel()
    scale = np.tile((interface.output_max - interface.output_min) / (interface.input_max - interface.input_min), 8)
    cache = {}
    attempts = 0

    def evaluate(x):
        nonlocal attempts
        x = finite(x, (48,))
        key = tuple(x)
        if key not in cache:
            if attempts >= MAX_RESPONSES:
                raise ResponseBudgetExhausted("The extra response budget is exhausted")
            commands = x.reshape(8, 6).copy()
            physical = np.asarray([interface.decode(row) for row in commands])
            if np.any(np.all(physical[:, 3:] == 0, axis=1)):
                raise ValueError("The frozen preview cannot evaluate a zero rotation latch")
            attempts += 1
            rollout = rollout_joint_preview(model, velocity, physical, site, lower, upper, enforce_limits=False)
            score = preview_objective(rollout, tool, targets, twists)
            values, jacobian, errors = pose_quality_constraints(score, 8, scale)
            cache[key] = dict(commands=commands, rollout=rollout, values=values, jacobian=jacobian, errors=errors)
        return cache[key]

    initial = evaluate(seed_x)
    np.testing.assert_allclose(initial["errors"], expected_errors, atol=1e-10, rtol=0)
    baseline_future = max(0.0, float(-np.min(initial["values"][4:])))
    np.testing.assert_allclose(baseline_future, expected_baseline_future_slack, atol=1e-10, rtol=0)
    domain = linear_domain(model, seed, previous, interface, lower, upper, past, initial["rollout"])
    initial_safety = domain_check(seed, domain, interface)
    if initial_safety["accepted"] is not True or not margin_prefix_satisfied(initial["errors"]):
        raise ValueError("The recorded starting plan must satisfy the original model domain")

    def objective(x):
        delta = x - seed_x
        return 0.5 * float(delta @ delta), delta

    def constraint(x):
        row = evaluate(x)
        return targeted_constraints(row["values"], row["jacobian"], baseline_future)

    constraints = [
        LinearConstraint(domain["rate"], domain["center"] - 0.25, domain["center"] + 0.25),
        LinearConstraint(domain["joint_a"], domain["joint_lo"], domain["joint_hi"]),
        {"type": "ineq", "fun": lambda x: constraint(x)[0], "jac": lambda x: constraint(x)[1]},
    ]
    if len(domain["terminal_normals"]):
        constraints.append(LinearConstraint(domain["terminal_a"], -np.inf, -domain["terminal_offset"]))
    try:
        result = minimize(
            objective,
            seed_x,
            jac=True,
            method="SLSQP",
            bounds=Bounds(domain["lo"], domain["hi"]),
            constraints=constraints,
            options={"maxiter": MAX_ITERATIONS, "ftol": 1e-10},
        )
        optimizer = dict(
            success=bool(result.success),
            status=int(result.status),
            message=str(result.message),
            iterations=int(result.nit),
        )
        if not 0 <= optimizer["iterations"] <= MAX_ITERATIONS:
            raise ValueError("The targeted search exceeded its iteration budget")
    except ResponseBudgetExhausted as error:
        optimizer = dict(success=False, status=None, message=str(error), iterations=None)

    eligible = []
    for key, row in cache.items():
        future = max(0.0, float(-np.min(row["values"][4:])))
        safety = domain_check(row["commands"], domain, interface)
        if candidate_is_eligible(row["errors"], future, baseline_future, safety):
            eligible.append((objective(np.asarray(key))[0], key, row, future, safety))
    selected = min(eligible, key=lambda item: (item[0], item[1])) if eligible else None
    candidate = None
    if selected is not None:
        distance, _key, row, future, safety = selected
        candidate = dict(
            witness_found=True,
            planned_commands=row["commands"],
            terminal_normals=domain["terminal_normals"],
            rollout=row["rollout"],
            first_tcp_error=row["errors"][0],
            tcp_errors=row["errors"],
            future_quality_slack=future,
            command_distance_squared=2 * distance,
            linear_safety=safety,
        )
    return dict(
        optimizer=optimizer,
        response_attempts=attempts,
        response_completions=len(cache),
        evaluated_candidates=len(cache),
        eligible_candidates=len(eligible),
        baseline_future_slack=baseline_future,
        baseline_tcp_errors=initial["errors"],
        baseline_linear_safety=initial_safety,
        candidate=candidate,
    )
