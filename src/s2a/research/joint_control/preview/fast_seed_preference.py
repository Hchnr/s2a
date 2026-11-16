"""Opt-in seed-first selection with the original constraints and witness audit."""

import copy
from types import FunctionType, SimpleNamespace

import numpy as np

from s2a.geometry import finite
from s2a.research.joint_control.preview.cached_seed_preference import choose_cached_seed
from s2a.research.joint_control.preview.shared_shift_fallback import SharedShiftFallbackAudit
from s2a.research.trajectory_search import margin_guarded_fallback as shared
from s2a.research.trajectory_search import margin_two_step_search as margin


class _AcceptedSeed(Exception):
    def __init__(self, candidate):
        super().__init__("The initial seed passed the original margin and domain checks")
        self.candidate = candidate


def _clone(function, namespace):
    result = FunctionType(function.__code__, namespace, function.__name__, function.__defaults__, function.__closure__)
    result.__kwdefaults__ = dict(function.__kwdefaults__ or {})
    return result


def _early_result(candidate):
    """Describe one actual seed rollout, with no fabricated optimizer phases."""
    trace = dict(
        primary_status="two_step_margin_candidate",
        primary_responses=1,
        primary_phases=[],
        primary_margin_evaluations=1,
        primary_feasibility_iterations=0,
        primary_total_iterations=0,
        fallback_status="not_attempted",
        fallback_responses=0,
        fallback_phases=[],
        fallback_initializer_reused=False,
        fallback_lp_attempts=0,
        selected_scope="margin_two_step",
        response_budget=shared.MAX_RESPONSES,
        feasibility_iteration_budget=shared.MAX_FEASIBILITY_ITERATIONS,
        total_iteration_budget=shared.MAX_TOTAL_ITERATIONS,
        start_lp_budget=2,
        optimizer_calls=[],
    )
    return dict(
        status="margin_guarded_candidate",
        response_evaluations=1,
        initialization=dict(
            seed_linear_safe=True,
            start_lp_attempts=0,
            latch_repair=False,
            warm_start_used=False,
            seed_rollout_reused=True,
            actual_model_rollouts=1,
        ),
        search_phases=[],
        shared_search=trace,
        best_joint_safe=dict(candidate, selection_scope="margin_two_step"),
        fast_seed=dict(
            early_selected=True,
            original_solver_continued=False,
            seed_domain_accepted=True,
            seed_margin_accepted=True,
            baseline_future_slack=None,
            selected_future_slack=candidate["future_quality_slack"],
            future_slack_increase=None,
            additional_model_responses=0,
        ),
    )


def solve_fast_preferred_seed(
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
    prefer_refresh=False,
    **kwargs,
):
    """Return a qualified initial seed early, otherwise run the original search.

    The early check runs where the frozen solver has already evaluated the seed
    and built its exact domain. The exception only unwinds private function
    namespaces; no shared module global or extra model response is changed. A
    selected candidate still requires the original independent witness audit.
    """
    if prefer_refresh is not False:
        raise ValueError("Seed and refresh preferences cannot be combined")
    seed = finite(seed, (8, 6)).copy()
    selected_info, observed = {}, {"probe_reached": False}

    def isolated_margin(capture):
        solver = shared.isolated_margin_solver(capture)
        namespace = solver.__globals__.copy()
        guarded = namespace["guarded"]

        def isolated_guarded():
            inner = guarded.isolated_solver()
            inner_namespace = inner.__globals__
            original_rollout = inner_namespace["rollout_joint_preview"]
            original_score = inner_namespace["preview_objective"]
            original_start = inner_namespace["nearest_linear_plan"]
            original_check = inner_namespace["domain_check"]
            quality = inner_namespace["pose_quality_constraints"]
            saved = {}

            def rollout(*args, **call_kwargs):
                value = original_rollout(*args, **call_kwargs)
                if "rollout" not in saved:
                    saved["rollout"] = value
                return value

            def score(*args, **call_kwargs):
                value = original_score(*args, **call_kwargs)
                if "score" not in saved:
                    saved["score"] = value
                return value

            def start(current_seed, domain, current_interface):
                if not np.array_equal(finite(current_seed, (8, 6)), seed):
                    raise ValueError("The original initializer must receive the same seed")
                if set(saved) != {"rollout", "score"}:
                    raise ValueError("The original seed rollout and score must already exist")
                observed["probe_reached"] = True
                safe = original_check(seed, domain, current_interface)
                scale = np.tile(
                    (current_interface.output_max - current_interface.output_min)
                    / (current_interface.input_max - current_interface.input_min),
                    8,
                )
                values, _, errors = quality(saved["score"], 8, scale)
                observed["seed_domain_accepted"] = bool(safe["accepted"])
                observed["seed_margin_accepted"] = margin.margin_prefix_satisfied(errors)
                if safe["accepted"] and observed["seed_margin_accepted"]:
                    raise _AcceptedSeed(
                        dict(
                            witness_found=True,
                            planned_commands=seed.copy(),
                            terminal_normals=np.asarray(domain["terminal_normals"]).copy(),
                            rollout=saved["rollout"],
                            first_tcp_error=errors[0].copy(),
                            tcp_errors=errors.copy(),
                            quality_slack=max(0.0, float(-np.min(values))),
                            future_quality_slack=max(0.0, float(-np.min(values[4:]))),
                            hard_pose_prefix=2,
                            prefix_selection_hard=True,
                            first_seen="seed_before_full_search",
                        )
                    )
                return original_start(current_seed, domain, current_interface)

            inner_namespace.update(rollout_joint_preview=rollout, preview_objective=score, nearest_linear_plan=start)
            return inner

        def select(candidates, *, prefer_refresh=False):
            if prefer_refresh is not False or selected_info:
                raise ValueError("Exactly one uncombined cached selection is allowed")
            selected, info = choose_cached_seed(candidates, seed)
            selected_info.update(info)
            return selected, None

        namespace["guarded"] = SimpleNamespace(
            isolated_solver=isolated_guarded, OBJECTIVE_SCALE=guarded.OBJECTIVE_SCALE
        )
        namespace["select_evaluated_margin_candidate"] = select
        return _clone(solver, namespace)

    solver = shared.solve_margin_guarded_fallback
    namespace = solver.__globals__.copy()
    namespace["isolated_margin_solver"] = isolated_margin
    try:
        result = _clone(solver, namespace)(
            model, velocity, seed, previous, interface, site, tool, targets, twists, lower, upper, **kwargs
        )
    except _AcceptedSeed as accepted:
        return _early_result(accepted.candidate)
    if not selected_info:
        selected_info = dict(
            seed_in_eligible_cache=False,
            selected_seed=False,
            selected_changed=False,
            eligible_candidates=0,
            default_future_slack=None,
            selected_future_slack=None,
            future_slack_increase=None,
            extra_model_responses=0,
        )
    result["seed_preference"] = selected_info
    result["fast_seed"] = dict(
        early_selected=False,
        original_solver_continued=True,
        seed_probe_reached=observed["probe_reached"],
        seed_domain_accepted=observed.get("seed_domain_accepted"),
        seed_margin_accepted=observed.get("seed_margin_accepted"),
        additional_model_responses=0,
    )
    return result


class FastSeedPreferredAudit(SharedShiftFallbackAudit):
    """Retain the full original witness audit for a seed-first candidate."""

    def __init__(self, receipt):
        if not isinstance(receipt, dict) or "fast_seed_traces" in receipt:
            raise ValueError("A fresh branch receipt is required")
        self.fast_seed_traces = receipt["fast_seed_traces"] = []

        def solver(*args, **kwargs):
            result = solve_fast_preferred_seed(*args, **kwargs)
            self.fast_seed_traces.append(copy.deepcopy(result["fast_seed"]))
            return result

        super().__init__(receipt, solver=solver)
