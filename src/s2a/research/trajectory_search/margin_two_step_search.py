"""Search and select two-pose plans with a fixed model-quality margin."""

import numpy as np

from s2a.geometry import finite
from s2a.research.trajectory_search import guarded_two_step_search as guarded
from s2a.research.trajectory_search import two_step_viability as original

POSITION_CAP_M = 0.01
ROTATION_CAP_RAD = 0.05
POSITION_MARGIN_M = 0.0005
ROTATION_MARGIN_RAD = 0.003
SELECTION_TOLERANCE = 1e-10
NORMALIZED_MARGIN = np.array([POSITION_MARGIN_M / POSITION_CAP_M, ROTATION_MARGIN_RAD / ROTATION_CAP_RAD] * 2)


def margin_prefix_objective(values, jacobian):
    """Penalize a shortfall from the fixed two-pose model margins."""
    values, jacobian = finite(values), finite(jacobian)
    if values.shape != (16,) or jacobian.ndim != 2 or jacobian.shape[0] != 16:
        raise ValueError("Eight position/rotation rows and matching Jacobian required")
    violation = np.minimum(values[:4] - NORMALIZED_MARGIN, 0.0)
    return 0.5 * guarded.OBJECTIVE_SCALE * float(violation @ violation), guarded.OBJECTIVE_SCALE * (
        violation @ jacobian[:4]
    )


def margin_prefix_constraints(values, jacobian, slack):
    """Apply the same hard margins during future-quality optimization."""
    values = finite(values).copy()
    if values.shape != (16,):
        raise ValueError("Eight position/rotation rows required")
    values[:4] -= NORMALIZED_MARGIN
    return original.split_prefix_constraints(values, jacobian, slack)


def margin_prefix_satisfied(errors):
    """Check both points using physical units and the registered tolerance."""
    errors = finite(errors, (8, 6))
    return bool(
        np.all(np.linalg.norm(errors[:2, :3], axis=1) <= POSITION_CAP_M - POSITION_MARGIN_M + SELECTION_TOLERANCE)
        and np.all(
            np.linalg.norm(errors[:2, 3:], axis=1) <= ROTATION_CAP_RAD - ROTATION_MARGIN_RAD + SELECTION_TOLERANCE
        )
    )


def solve_margin_two_step(
    model, velocity, seed, previous, interface, site, tool, targets, twists, lower, upper, **kwargs
):
    """Select only evaluated, joint-safe plans that satisfy both fixed margins.

    The original solver runs in an isolated namespace and owns its response
    cache and budget. Instrumentation reads those already-computed responses;
    it does not issue new model calls or change the original solver globals.
    """
    targets = finite(targets, (8, 4, 4))
    solver = guarded.isolated_solver()
    namespace = solver.__globals__
    original_rollout = namespace["rollout_joint_preview"]
    original_score = namespace["preview_objective"]
    original_check = namespace["domain_check"]
    rollouts, scores, safe = {}, {}, {}
    rollout_calls = 0

    def physical_key(commands):
        return np.asarray([interface.decode(row) for row in commands], dtype=float).tobytes()

    def observed_rollout(
        current_model, current_velocity, physical, current_site, current_lower, current_upper, *, enforce_limits
    ):
        nonlocal rollout_calls
        result = original_rollout(
            current_model,
            current_velocity,
            physical,
            current_site,
            current_lower,
            current_upper,
            enforce_limits=enforce_limits,
        )
        rollout_calls += 1
        rollouts[np.asarray(physical, dtype=float).tobytes()] = result
        return result

    def observed_score(rollout, current_tool, current_targets, current_twists):
        result = original_score(rollout, current_tool, current_targets, current_twists)
        scores[id(rollout)] = result
        return result

    def observed_check(commands, domain, current_interface):
        result = original_check(commands, domain, current_interface)
        if result["accepted"]:
            candidate = finite(commands, (8, 6)).copy()
            safe[tuple(candidate.ravel())] = candidate
        return result

    namespace.update(
        prefix_violation_objective=margin_prefix_objective,
        split_prefix_constraints=margin_prefix_constraints,
        rollout_joint_preview=observed_rollout,
        preview_objective=observed_score,
        domain_check=observed_check,
    )
    result = solver(
        model, velocity, seed, previous, interface, site, tool, targets.copy(), twists, lower, upper, **kwargs
    )
    if rollout_calls != result["response_evaluations"] or len(safe) != result.get("joint_safe_evaluations", 0):
        raise ValueError("Recorded cached responses and safe candidates must match the isolated solve")
    scale = np.tile((interface.output_max - interface.output_min) / (interface.input_max - interface.input_min), 8)
    candidates = []
    for key, commands in safe.items():
        physical = physical_key(commands)
        if physical not in rollouts or id(rollouts[physical]) not in scores:
            raise ValueError("A safe command must have an already-scored rollout")
        rollout = rollouts[physical]
        score = scores[id(rollout)]
        values, _, errors = namespace["pose_quality_constraints"](score, 8, scale)
        if margin_prefix_satisfied(errors):
            future = max(0.0, float(-np.min(values[4:])))
            overall = max(0.0, float(-np.min(values)))
            candidates.append((future, float(score["objective"]), key, commands, rollout, errors, overall))

    primary_status = result["status"]
    result.pop("best_joint_safe", None)
    result["margin_search"] = dict(
        model_position_margin_m=POSITION_MARGIN_M,
        model_rotation_margin_rad=ROTATION_MARGIN_RAD,
        selection_tolerance=SELECTION_TOLERANCE,
        normalized_margin=NORMALIZED_MARGIN.copy(),
        objective_scale=guarded.OBJECTIVE_SCALE,
        original_acceptance_caps_changed=False,
        primary_status=primary_status,
        primary_nominal_prefix_evaluations=result.get("prefix_hard_evaluations", 0),
        recorded_rollout_calls=rollout_calls,
        captured_joint_safe_evaluations=len(safe),
        margin_prefix_evaluations=len(candidates),
        extra_model_responses=0,
    )
    if candidates:
        future, _objective, _key, commands, rollout, errors, overall = min(
            candidates, key=lambda row: (row[0], row[1], row[2])
        )
        result["best_joint_safe"] = dict(
            witness_found=True,
            planned_commands=commands.copy(),
            terminal_normals=np.asarray(result["terminal_normals"]).copy(),
            rollout=rollout,
            first_tcp_error=errors[0].copy(),
            tcp_errors=errors.copy(),
            quality_slack=overall,
            future_quality_slack=future,
            hard_pose_prefix=2,
            prefix_selection_hard=True,
            first_seen="evaluated_margin_search_cache",
        )
        result["status"] = "two_step_margin_candidate"
    elif primary_status in ("two_step_candidate", "two_step_no_witness"):
        result["status"] = "two_step_margin_no_witness"
    return result
