"""Opt-in quality solver with a narrowly bounded rate-rounding repair."""

import inspect
from types import FunctionType

import numpy as np

from s2a.geometry import pose_error
from s2a.research.joint_control.limits.joint_preview_constraint_audit import audit_joint_preview_constraints
from s2a.research.trajectory_search import quality_capped_inner_joint_plan as quality
from s2a.research.trajectory_search.rate_boundary_repair import repair_nearby_rate_plan


def _original_with_final_cache(*args, **kwargs):
    """Run the pinned solver in an isolated namespace and retain its final cache entry."""
    original = quality.solve_interior_plan
    namespace = original.__globals__.copy()
    minimize = namespace["minimize"]
    captured = {}

    def observed(objective, *positional, **options):
        result = minimize(objective, *positional, **options)
        evaluate = inspect.getclosurevars(objective).nonlocals["evaluate"]
        cache = inspect.getclosurevars(evaluate).nonlocals["cache"]
        key = tuple(result.x)
        if result.success and key in cache:
            captured.update(commands=np.asarray(result.x).reshape(-1, 6).copy(), rollout=cache[key]["rollout"])
        return result

    namespace["minimize"] = observed
    solver = FunctionType(original.__code__, namespace, original.__name__, original.__defaults__, original.__closure__)
    solver.__kwdefaults__ = dict(original.__kwdefaults__ or {})
    return solver(*args, **kwargs), captured


def _checks(rollout, commands, previous, interface, lower, upper, past, normals, braking_time_s, first_target, tool):
    """Check the original model constraints and absolute first-pose caps."""
    audit = audit_joint_preview_constraints(
        rollout,
        commands,
        previous,
        interface,
        lower,
        upper,
        past_inner_error_rad=past,
        terminal_normals=normals,
        braking_time_s=braking_time_s,
    )
    failed = [name for name, group in audit["groups"].items() if len(group["violated_indices"])]
    error = pose_error(first_target, np.asarray(rollout["pose"])[0] @ tool)
    if np.linalg.norm(error[:3]) > 0.01 + 1e-10:
        failed.append("first_position")
    if np.linalg.norm(error[3:]) > 0.05 + 1e-10:
        failed.append("first_rotation")
    return failed, error


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
    braking_time_s=0.1,
    past_inner_error_rad=None,
):
    """Try the unchanged solver first; expose only a fully rechecked repair.

    This is opt-in. Its model witness still needs the caller's independent
    audit before any physical command may be staged or dispatched.
    """
    args = (model, velocity, seed, previous, interface, site, tool, targets, twists, lower, upper)
    options = dict(
        initial_guess=initial_guess,
        terminal_normals=terminal_normals,
        braking_time_s=braking_time_s,
        past_inner_error_rad=past_inner_error_rad,
    )
    result, final = _original_with_final_cache(*args, **options)
    if result["witness_found"] is True:
        return result
    result = dict(result)
    result["rate_repair"] = {"attempted": False, "reason": "no_eligible_cached_final_candidate"}
    used = int(result["response_evaluations"])
    if not final or result.get("optimizer_success") is not True or used >= 160:
        return result
    past = np.zeros((25, len(lower))) if past_inner_error_rad is None else np.asarray(past_inner_error_rad)
    normals = np.asarray(result["terminal_normals"]).reshape(-1, len(lower))
    failed, _ = _checks(
        final["rollout"],
        final["commands"],
        previous,
        interface,
        lower,
        upper,
        past,
        normals,
        braking_time_s,
        targets[0],
        tool,
    )
    if not failed or any(name not in ("rate_lower", "rate_upper") for name in failed):
        result["rate_repair"] = {
            "attempted": False,
            "reason": "original_candidate_fails_other_constraints",
            "failed": failed,
        }
        return result
    try:
        repair = repair_nearby_rate_plan(
            final["commands"],
            previous,
            interface.input_min,
            interface.input_max,
            max_rate=0.25,
            interior=1e-10,
            max_correction=1e-8,
        )
    except ValueError as error:
        result["rate_repair"] = {"attempted": True, "reason": "correction_rejected", "detail": str(error)}
        return result
    plan = repair["planned_commands"]
    deltas = np.asarray([interface.decode(command) for command in plan])
    result["response_evaluations"] = used + 1
    rollout = quality.rollout_joint_preview(model, velocity, deltas, site, lower, upper, enforce_limits=False)
    failed, error = _checks(
        rollout, plan, previous, interface, lower, upper, past, normals, braking_time_s, targets[0], tool
    )
    result["rate_repair"] = {
        "attempted": True,
        "primary_response_evaluations": used,
        "additional_model_responses": 1,
        "maximum_command_change": repair["maximum_command_change"],
        "maximum_rate_after_repair": repair["maximum_rate_after_repair"],
        "failed_after_repair": failed,
        "requires_independent_audit": True,
    }
    if failed:
        return result
    result.update(
        status="model_witness_rate_repaired",
        witness_found=True,
        planned_commands=plan,
        first_tcp_error=error,
        first_pose_caps_enforced=True,
        original_first_pose_caps_satisfied=True,
        rollout=rollout,
        scope="frozen model witness only; caller must independently audit before dispatch",
    )
    return result
