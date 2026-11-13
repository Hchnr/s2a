"""Frozen-model quality search with an extra completed TCP-innovation gate."""

import inspect
from numbers import Real
from types import FunctionType

import numpy as np
from scipy.spatial.transform import Rotation

from s2a.geometry import finite
from s2a.research.joint_control.inverse.full_coupled_pose_inverse import right_inverse_jacobian
from s2a.research.trajectory_search import quality_capped_inner_joint_plan as quality
from s2a.research.trajectory_search import quality_rate_repair_solver as original_checks
from s2a.research.trajectory_search.rate_boundary_repair import repair_nearby_rate_plan


def corrected_error(raw, translation_innovation, rotation_discrepancy):
    """Compose one previous world-frame discrepancy with the current SO(3) error."""
    raw = finite(raw, (6,))
    translation = finite(translation_innovation, (3,))
    rotation = finite(rotation_discrepancy, (3, 3))
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-9, rtol=0) or not np.isclose(
        np.linalg.det(rotation), 1, atol=1e-9, rtol=0
    ):
        raise ValueError("A proper world-frame SO(3) discrepancy is required")
    corrected = Rotation.from_rotvec(raw[3:]) * Rotation.from_matrix(rotation).inv()
    return np.r_[raw[:3] + translation, corrected.as_rotvec()]


def corrected_error_jacobian(raw, raw_jacobian, rotation_jacobian, translation_innovation, rotation_discrepancy):
    """Differentiate the corrected error under world-left pose perturbations."""
    jacobian, rotation_jacobian = finite(raw_jacobian), finite(rotation_jacobian)
    if jacobian.ndim != 2 or jacobian.shape[0] != 6 or rotation_jacobian.shape != (3, jacobian.shape[1]):
        raise ValueError("Aligned six-dimensional error and world-rotation Jacobians are required")
    error = corrected_error(raw, translation_innovation, rotation_discrepancy)
    rotation = np.asarray(rotation_discrepancy)
    return np.vstack((jacobian[:3], -right_inverse_jacobian(error[3:]) @ rotation @ rotation_jacobian))


def _assess(
    rollout,
    commands,
    previous,
    interface,
    lower,
    upper,
    past,
    normals,
    braking,
    target,
    tool,
    translation,
    rotation,
    caps,
):
    failed, raw = original_checks._checks(
        rollout, commands, previous, interface, lower, upper, past, normals, braking, target, tool
    )
    corrected = corrected_error(raw, translation, rotation)
    if np.linalg.norm(corrected[:3]) > caps[0] + 1e-10:
        failed.append("innovation_position")
    if np.linalg.norm(corrected[3:]) > caps[1] + 1e-10:
        failed.append("innovation_rotation")
    return failed, raw, corrected


def _search_with_innovation(args, options, translation, rotation, caps):
    original = quality.solve_interior_plan
    namespace = original.__globals__.copy()
    minimize = namespace["minimize"]
    captured = {}
    seed, interface = np.asarray(args[2]), args[4]
    scale = np.tile(
        (interface.output_max - interface.output_min) / (interface.input_max - interface.input_min), len(seed)
    )
    weights = np.array([100.0] * 3 + [20.0] * 3)

    def observed(objective, *positional, **keywords):
        evaluate = inspect.getclosurevars(objective).nonlocals["evaluate"]
        cache = inspect.getclosurevars(evaluate).nonlocals["cache"]

        def value_and_jacobian(x):
            item = evaluate(x)
            raw = item["score"]["residual"][:6] / weights
            raw_jac = item["score"]["jacobian"][:6] * scale[None, :] / weights[:, None]
            rotation_jac = item["rollout"]["rotation_jacobian"][0] * scale[None, :]
            error = corrected_error(raw, translation, rotation)
            jac = corrected_error_jacobian(raw, raw_jac, rotation_jac, translation, rotation)
            values, derivatives = [], []
            for start, end, cap in ((0, 3, caps[0]), (3, 6, caps[1])):
                norm = np.linalg.norm(error[start:end])
                direction = error[start:end] / norm if norm > 1e-15 else np.zeros(3)
                values.append(1 - norm / cap)
                derivatives.append(-direction @ jac[start:end] / cap)
            return np.asarray(values), np.asarray(derivatives)

        constraint = {
            "type": "ineq",
            "fun": lambda x: value_and_jacobian(x)[0],
            "jac": lambda x: value_and_jacobian(x)[1],
        }
        keywords["constraints"] = [*keywords["constraints"], constraint]
        try:
            return minimize(objective, *positional, **keywords)
        finally:
            captured["cache"] = cache

    namespace["minimize"] = observed
    clone = FunctionType(original.__code__, namespace, original.__name__, original.__defaults__, original.__closure__)
    clone.__kwdefaults__ = dict(original.__kwdefaults__ or {})
    return clone(*args, **options), captured


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
    translation_innovation,
    rotation_discrepancy,
    past_inner_error_rad,
    braking_time_s=0.1,
    corrected_position_cap_m=0.01,
    corrected_rotation_cap_rad=0.05,
):
    """Find a candidate passing both original and causal-innovation model gates.

    The candidate requires independent physical-model audit before any dispatch.
    A completed past innovation is empirical and is not a future safety bound.
    Optional stricter corrected-error caps provide a model margin without
    changing the original position and rotation acceptance limits.
    """
    caps = (corrected_position_cap_m, corrected_rotation_cap_rad)
    if any(
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Real)
        or not np.isfinite(value)
        or not 0 < value <= maximum
        for value, maximum in zip(caps, (0.01, 0.05), strict=True)
    ):
        raise ValueError("Corrected pose caps must be finite, positive, and no larger than the original caps")
    caps = tuple(float(value) for value in caps)
    translation, rotation = finite(translation_innovation, (3,)), finite(rotation_discrepancy, (3, 3))
    corrected_error(np.zeros(6), translation, rotation)
    args = (model, velocity, seed, previous, interface, site, tool, targets, twists, lower, upper)
    options = {"past_inner_error_rad": past_inner_error_rad, "braking_time_s": braking_time_s}
    original, captured = _search_with_innovation(args, options, translation, rotation, caps)
    used = int(original["response_evaluations"])
    candidate_fields = {
        "rollout",
        "planned_commands",
        "first_tcp_error",
        "first_pose_caps_enforced",
        "original_first_pose_caps_satisfied",
    }
    result = {key: value for key, value in original.items() if key not in candidate_fields}
    result.update(
        original_solver_status=original["status"],
        status="innovation_search_no_witness",
        witness_found=False,
        requires_independent_audit=True,
        future_innovation_guarantee=False,
        corrected_position_cap_m=caps[0],
        corrected_rotation_cap_rad=caps[1],
    )
    cache = captured.get("cache", {})
    if not cache:
        result.update(cached_candidates=0, strict_candidates=0, near_rate_candidates=0)
        return result
    if len(cache) != used:
        raise ValueError("Cached model evaluations must match the original solver response count")
    normals = np.asarray(original["terminal_normals"]).reshape(-1, len(lower))
    past = finite(past_inner_error_rad, (25, len(lower)))
    strict, near = [], []
    for index, (key, item) in enumerate(cache.items()):
        plan = np.asarray(key).reshape(-1, 6)
        failed, raw, corrected = _assess(
            item["rollout"],
            plan,
            previous,
            interface,
            lower,
            upper,
            past,
            normals,
            braking_time_s,
            targets[0],
            tool,
            translation,
            rotation,
            caps,
        )
        objective = float(item["score"]["objective"])
        if not failed:
            strict.append((objective, index, plan, item["rollout"], raw, corrected))
        elif all(name in ("rate_lower", "rate_upper") for name in failed):
            maximum_rate = float(np.max(np.abs(np.diff(np.vstack((previous, plan)), axis=0))))
            if maximum_rate <= 0.25 + 1e-8:
                near.append((maximum_rate - 0.25, objective, index, plan))
    result.update(cached_candidates=len(cache), strict_candidates=len(strict), near_rate_candidates=len(near))
    chosen = min(strict, key=lambda row: (row[0], row[1])) if strict else None
    if chosen is None and near and used < 160:
        _excess, _objective, _index, plan = min(near, key=lambda row: (row[0], row[1], row[2]))
        try:
            repair = repair_nearby_rate_plan(
                plan,
                previous,
                interface.input_min,
                interface.input_max,
                max_rate=0.25,
                interior=1e-10,
                max_correction=1e-8,
            )
        except ValueError as error:
            result["repair"] = {"attempted": True, "accepted": False, "reason": str(error)}
        else:
            plan = repair["planned_commands"]
            deltas = np.asarray([interface.decode(command) for command in plan])
            result["response_evaluations"] = used + 1
            rollout = quality.rollout_joint_preview(model, velocity, deltas, site, lower, upper, enforce_limits=False)
            failed, raw, corrected = _assess(
                rollout,
                plan,
                previous,
                interface,
                lower,
                upper,
                past,
                normals,
                braking_time_s,
                targets[0],
                tool,
                translation,
                rotation,
                caps,
            )
            result["repair"] = {
                "attempted": True,
                "accepted": not failed,
                "maximum_command_change": repair["maximum_command_change"],
                "failed_after_repair": failed,
                "additional_model_responses": 1,
            }
            if not failed:
                chosen = (None, None, plan, rollout, raw, corrected)
    if chosen is None:
        return result
    _objective, _index, plan, rollout, raw, corrected = chosen
    result.update(
        status="innovation_model_witness",
        witness_found=True,
        planned_commands=plan.copy(),
        first_tcp_error=raw,
        innovation_corrected_first_tcp_error=corrected,
        first_pose_caps_enforced=True,
        original_first_pose_caps_satisfied=True,
        rollout=rollout,
        scope="frozen model witness only; completed innovation does not guarantee future response",
    )
    return result
