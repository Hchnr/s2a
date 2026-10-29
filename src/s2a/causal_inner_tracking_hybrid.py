"""Fixed interior, adaptive positive-margin witness, then explicit soft fallback."""

import numpy as np

from s2a.fast_long_joint_preview import rollout_joint_preview
from s2a.joint_limit_preview_objective import preview_objective
from s2a.past_inner_error_joint_plan import solve_interior_plan


class NoCausalWitness(RuntimeError):
    """No executable tightened plan; caller must record and end the trial."""

    def __init__(self, diagnostic):
        super().__init__(diagnostic["status"])
        self.diagnostic = diagnostic


def brief(result):
    return {
        k: v
        for k, v in result.items()
        if k not in ("rollout", "planned_commands", "seed_commands", "linear_witness_commands")
    }


def refine_long_plan(
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
):
    args = (model, velocity, seed, previous, interface, site, tool, targets, twists, lower, upper)
    hard = solve_interior_plan(*args, past_inner_error_rad=past_inner_error_rad)
    diagnostics = dict(interior_diagnostic=brief(hard), adaptive_diagnostic=None, refinement_diagnostic=None)
    counts = dict(
        hard_response_evaluations=hard["response_evaluations"],
        adaptive_response_evaluations=0,
        refinement_response_evaluations=0,
        soft_response_evaluations=0,
    )

    def pack(result, selection, margin):
        commands = result["planned_commands"]
        inner = result["rollout"]["inner_joint_position"]
        actual = float(np.min(np.minimum(inner - lower, upper - inner)))
        if actual < margin - 1e-10:
            raise ValueError("Selected retained margin failed recheck")
        endpoint = rollout_joint_preview(
            model,
            velocity,
            np.asarray([interface.decode(c) for c in commands]),
            site,
            lower,
            upper,
            enforce_limits=True,
        )
        for key in ("pose", "twist", "joint_position", "joint_velocity"):
            np.testing.assert_allclose(endpoint[key], result["rollout"][key], atol=1e-10, rtol=0)
        if selection == "interior_witness" and np.any(
            np.asarray(hard["terminal_normals"]) @ endpoint["joint_velocity"][-1] > 1e-10
        ):
            raise ValueError("Terminal velocity witness failed recheck")
        seed_endpoint = rollout_joint_preview(
            model, velocity, np.asarray([interface.decode(c) for c in seed]), site, lower, upper, enforce_limits=False
        )
        score = preview_objective(endpoint, tool, targets, twists)
        old = preview_objective(seed_endpoint, tool, targets, twists)
        caps = np.maximum(
            [np.linalg.norm(old["residual"][:3]) / 100.0, np.linalg.norm(old["residual"][3:6]) / 20.0], [0.01, 0.05]
        )
        e = score["residual"][:6] / np.array([100.0] * 3 + [20.0] * 3)
        passed = bool(np.linalg.norm(e[:3]) <= caps[0] + 1e-10 and np.linalg.norm(e[3:]) <= caps[1] + 1e-10)
        if selection != "interior_witness" and not passed:
            raise ValueError("Original first TCP caps failed selection recheck")
        return dict(
            planned_commands=commands,
            rollout=endpoint,
            objective=score["objective"],
            seed_objective=old["objective"],
            tracking_objective=score["objective"],
            seed_tracking_objective=old["objective"],
            margin_objective=0.0,
            seed_margin_objective=0.0,
            first_pose_caps=caps,
            first_pose_caps_enforced=selection != "interior_witness",
            original_first_pose_caps_satisfied=passed,
            joint_margin_rad=margin,
            past_inner_error_rad=hard["past_inner_error_rad"],
            minimum_tightened_inner_slack_rad=hard["minimum_tightened_inner_slack_rad"],
            minimum_inner_margin_rad=actual,
            requested_margin_satisfied=bool(actual >= 0.05 - 1e-10),
            joint_margin_weight=0.0,
            response_evaluations=sum(counts.values()),
            **counts,
            **diagnostics,
            optimizer_success=result["optimizer_success"],
            optimizer_message=result["optimizer_message"],
            iterations=result["iterations"],
            accepted=not np.array_equal(seed, commands),
            selection=selection,
            seed_guard_model="free",
            terminal_constraint_enforced=selection == "interior_witness",
            terminal_signed_velocities_rad_s=np.asarray(hard["terminal_normals"]) @ endpoint["joint_velocity"][-1],
            scope="positive frozen-model margin witness; original .05rad target separately classified; no physical guarantee",
        )

    if hard["witness_found"]:
        return pack(hard, "interior_witness", 0.05)
    raise NoCausalWitness(brief(hard))
