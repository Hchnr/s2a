"""One local linear program for a refresh-margin model diagnostic."""

import numpy as np
from scipy.optimize import linprog

from s2a.geometry import finite
from s2a.research.joint_control.limits.joint_limit_preview_objective import preview_objective
from s2a.research.joint_control.preview.interior_joint_preview import rollout_joint_preview
from s2a.research.trajectory_search.feasible_start_restoration import domain_check, linear_domain
from s2a.research.trajectory_search.full_horizon_quality import pose_quality_constraints
from s2a.research.trajectory_search.margin_two_step_search import margin_prefix_satisfied
from s2a.research.trajectory_search.refresh_margin_targeted_search import (
    candidate_is_eligible,
    targeted_constraints,
)

TRUST_REGION = 0.1
MAX_RESPONSES = 2


def linearized_problem(seed, domain, values, jacobian, baseline_future_slack):
    """Build one L1-nearest change subject to fixed linearized constraints."""
    seed = finite(seed, (8, 6)).ravel()
    values, jacobian = finite(values, (16,)), finite(jacobian, (16, 48))
    quality, quality_jacobian = targeted_constraints(values, jacobian, baseline_future_slack)
    rate, center = domain["rate"], domain["center"]
    joints, low, high = domain["joint_a"], domain["joint_lo"], domain["joint_hi"]
    terminal, offset = domain["terminal_a"], domain["terminal_offset"]
    matrix = np.vstack((rate, -rate, joints, -joints, terminal, -quality_jacobian))
    limits = np.r_[
        center + 0.25 - rate @ seed,
        -center + 0.25 + rate @ seed,
        high - joints @ seed,
        -low + joints @ seed,
        -offset - terminal @ seed,
        quality,
    ]
    identity = np.eye(48)
    matrix = np.vstack(
        (np.c_[matrix, np.zeros((len(matrix), 48))], np.c_[identity, -identity], np.c_[-identity, -identity])
    )
    limits = np.r_[limits, np.zeros(96)]
    lower = np.maximum(domain["lo"] - seed, -TRUST_REGION)
    upper = np.minimum(domain["hi"] - seed, TRUST_REGION)
    if np.any(lower > upper):
        raise ValueError("The starting command box and trust region must intersect")
    return dict(
        objective=np.r_[np.zeros(48), np.ones(48)],
        matrix=matrix,
        limits=limits,
        bounds=list(zip(lower, upper, strict=True)) + [(0.0, None)] * 48,
    )


def solve_linearized_refresh(
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
    on_response=None,
    on_lp=None,
):
    """Evaluate the original plan and at most one LP-proposed plan."""
    seed, previous = finite(seed, (8, 6)).copy(), finite(previous, (6,))
    targets, past = finite(targets, (8, 4, 4)), finite(past_inner_error_rad, (25, 7))
    lower, upper = finite(lower, (7,)), finite(upper, (7,))
    expected = finite(expected_seed_tcp_errors, (8, 6))
    if any(callback is not None and not callable(callback) for callback in (on_response, on_lp)):
        raise ValueError("Optional accounting callbacks must be callable")
    scale = np.tile((interface.output_max - interface.output_min) / (interface.input_max - interface.input_min), 8)
    attempts, completed = 0, 0

    def evaluate(commands):
        nonlocal attempts, completed
        commands = finite(commands, (8, 6))
        if attempts >= MAX_RESPONSES:
            raise ValueError("The two-response diagnostic budget is exhausted")
        physical = np.asarray([interface.decode(row) for row in commands])
        if np.any(np.all(physical[:, 3:] == 0, axis=1)):
            raise ValueError("The frozen preview cannot evaluate a zero rotation latch")
        attempts += 1
        if on_response is not None:
            on_response("attempted")
        rollout = rollout_joint_preview(model, velocity, physical, site, lower, upper, enforce_limits=False)
        score = preview_objective(rollout, tool, targets, twists)
        values, jacobian, errors = pose_quality_constraints(score, 8, scale)
        completed += 1
        if on_response is not None:
            on_response("completed")
        return dict(rollout=rollout, values=values, jacobian=jacobian, errors=errors)

    initial = evaluate(seed)
    np.testing.assert_allclose(initial["errors"], expected, atol=1e-10, rtol=0)
    baseline = max(0.0, float(-np.min(initial["values"][4:])))
    np.testing.assert_allclose(baseline, expected_baseline_future_slack, atol=1e-10, rtol=0)
    domain = linear_domain(model, seed, previous, interface, lower, upper, past, initial["rollout"])
    baseline_safety = domain_check(seed, domain, interface)
    if baseline_safety["accepted"] is not True or not margin_prefix_satisfied(initial["errors"]):
        raise ValueError("The recorded starting plan must satisfy the original model domain")
    problem = linearized_problem(seed, domain, initial["values"], initial["jacobian"], baseline)
    if on_lp is not None:
        on_lp("attempted")
    result = linprog(
        problem["objective"], A_ub=problem["matrix"], b_ub=problem["limits"], bounds=problem["bounds"], method="highs"
    )
    if on_lp is not None:
        on_lp("completed")
    optimizer = dict(
        success=bool(result.success), status=int(result.status), message=str(result.message), iterations=int(result.nit)
    )
    candidate, tested = None, None
    if result.success:
        delta = finite(result.x[:48], (48,))
        if np.max(np.abs(delta)) > TRUST_REGION + 1e-10:
            raise ValueError("The linear program exceeded the fixed command trust region")
        commands = (seed.ravel() + delta).reshape(8, 6)
        row = evaluate(commands)
        future = max(0.0, float(-np.min(row["values"][4:])))
        safety = domain_check(commands, domain, interface)
        eligible = candidate_is_eligible(row["errors"], future, baseline, safety)
        linear_quality, _ = targeted_constraints(
            initial["values"] + initial["jacobian"] @ delta, initial["jacobian"], baseline
        )
        actual_quality, _ = targeted_constraints(row["values"], row["jacobian"], baseline)
        tested = dict(
            eligible=eligible,
            command_change_l1=float(np.sum(np.abs(delta))),
            command_change_max=float(np.max(np.abs(delta))),
            tcp_errors=row["errors"],
            future_quality_slack=future,
            linear_safety=safety,
            minimum_linearized_quality=float(np.min(linear_quality)),
            minimum_actual_quality=float(np.min(actual_quality)),
        )
        if eligible:
            candidate = dict(
                witness_found=True,
                planned_commands=commands,
                terminal_normals=domain["terminal_normals"],
                rollout=row["rollout"],
                first_tcp_error=row["errors"][0],
                tcp_errors=row["errors"],
                future_quality_slack=future,
                linear_safety=safety,
            )
    return dict(
        optimizer=optimizer,
        response_attempts=attempts,
        response_completions=completed,
        evaluated_candidates=completed,
        eligible_candidates=int(candidate is not None),
        baseline_future_slack=baseline,
        baseline_tcp_errors=initial["errors"],
        baseline_linear_safety=baseline_safety,
        trust_region=TRUST_REGION,
        tested_candidate=tested,
        candidate=candidate,
    )
