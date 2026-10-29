"""Interior-model witness selection with explicit soft-prevention fallback."""

import numpy as np

from s2a.fast_long_joint_preview import rollout_joint_preview
from s2a.interior_joint_plan import solve_interior_plan
from s2a.preventive_joint_inverse import refine_long_plan as soft_plan


def refine_long_plan(model, velocity, seed, previous, interface, site, tool, targets, twists, lower, upper):
    args = (model, velocity, seed, previous, interface, site, tool, targets, twists, lower, upper)
    hard = solve_interior_plan(*args)
    diagnostic = {
        k: v
        for k, v in hard.items()
        if k not in ("rollout", "seed_commands", "planned_commands", "linear_witness_commands")
    }
    if not hard["witness_found"]:
        result = soft_plan(*args)
        result["soft_response_evaluations"] = result["response_evaluations"]
        result["response_evaluations"] += hard["response_evaluations"]
        result.update(selection="preventive_fallback", interior_diagnostic=diagnostic, seed_guard_model="ideal_limit")
        return result
    commands = hard["planned_commands"]
    rollout = rollout_joint_preview(
        model, velocity, np.asarray([interface.decode(c) for c in commands]), site, lower, upper, enforce_limits=True
    )
    for key in ("pose", "twist", "joint_position", "joint_velocity"):
        np.testing.assert_allclose(rollout[key], hard["rollout"][key], atol=1e-10, rtol=0)
    return dict(
        planned_commands=commands,
        rollout=rollout,
        objective=hard["objective"],
        seed_objective=hard["seed_objective"],
        tracking_objective=hard["objective"],
        seed_tracking_objective=hard["seed_objective"],
        margin_objective=0.0,
        seed_margin_objective=0.0,
        first_pose_caps=hard["first_pose_caps"],
        joint_margin_rad=0.05,
        joint_margin_weight=0.0,
        response_evaluations=hard["response_evaluations"],
        soft_response_evaluations=0,
        optimizer_success=hard["optimizer_success"],
        optimizer_message=hard["optimizer_message"],
        iterations=hard["iterations"],
        accepted=not np.array_equal(seed, commands),
        selection="interior_witness",
        interior_diagnostic=diagnostic,
        seed_guard_model="free",
        scope="frozen model witness; actual dynamics and future recursive feasibility unproven",
    )
