"""Inspect already-evaluated guarded-search plans for first-pose feasibility."""

import numpy as np

from s2a.geometry import finite, pose_error
from s2a.research.trajectory_search import guarded_two_step_search as guarded


def collect_cached_first_pose(
    model, velocity, seed, previous, interface, site, tool, targets, twists, lower, upper, **kwargs
):
    """Keep the primary solve unchanged and inspect its existing safe cache.

    The isolated function's own globals are instrumented; the frozen solver's
    module namespace is never changed. No model response is added here.
    """
    targets = finite(targets, (8, 4, 4))
    solver = guarded.isolated_solver()
    namespace = solver.__globals__
    original_rollout, original_check = namespace["rollout_joint_preview"], namespace["domain_check"]
    rollouts, safe = {}, {}
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

    def observed_check(commands, domain, current_interface):
        result = original_check(commands, domain, current_interface)
        if result["accepted"]:
            candidate = finite(commands, (8, 6)).copy()
            safe[tuple(candidate.ravel())] = candidate
        return result

    namespace["rollout_joint_preview"] = observed_rollout
    namespace["domain_check"] = observed_check
    result = solver(
        model, velocity, seed, previous, interface, site, tool, targets.copy(), twists, lower, upper, **kwargs
    )
    result["first_stage_search_objective"] = dict(
        normalized_guard=guarded.SEARCH_GUARD,
        scale=guarded.OBJECTIVE_SCALE,
        acceptance_caps_changed=False,
        future_stage_changed=False,
    )
    if rollout_calls != result["response_evaluations"] or len(safe) != result.get("joint_safe_evaluations", 0):
        raise ValueError("Recorded cached responses and safe candidates must match the primary solve")
    candidates = []
    for commands in safe.values():
        key = physical_key(commands)
        if key not in rollouts:
            raise ValueError("A safe command must have an already-evaluated rollout")
        rollout = rollouts[key]
        errors = np.asarray(
            [pose_error(target, pose @ tool) for target, pose in zip(targets, rollout["pose"], strict=True)]
        )
        pos, rot = np.linalg.norm(errors[:, :3], axis=1), np.linalg.norm(errors[:, 3:], axis=1)
        if pos[0] > 0.01 + 1e-10 or rot[0] > 0.05 + 1e-10:
            continue
        future = max(0.0, float(np.max(np.r_[pos[1:] / 0.01 - 1, rot[1:] / 0.05 - 1])))
        gap = max(0.0, float(np.max(np.r_[pos / 0.01 - 1, rot / 0.05 - 1])))
        candidates.append((future, tuple(commands.ravel()), commands, rollout, errors, gap))
    selected = None
    if candidates:
        future, _key, commands, rollout, errors, gap = min(candidates, key=lambda row: (row[0], row[1]))
        selected = dict(
            witness_found=True,
            planned_commands=commands.copy(),
            rollout=rollout,
            first_tcp_error=errors[0].copy(),
            tcp_errors=errors.copy(),
            terminal_normals=np.asarray(result["terminal_normals"]).copy(),
            quality_slack=gap,
            future_quality_slack=future,
            scope="first pose and original model safety; later poses are not hard constraints",
        )
    return dict(
        primary=result,
        recorded_rollout_calls=rollout_calls,
        unique_safe_candidates=len(safe),
        first_pose_candidates=len(candidates),
        best_first_pose=selected,
        extra_model_responses=0,
    )
