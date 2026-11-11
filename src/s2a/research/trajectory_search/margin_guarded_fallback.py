"""Prioritize model margins and reuse the remaining budget for guarded search."""

import copy
from types import FunctionType, SimpleNamespace

import numpy as np

from s2a.geometry import finite
from s2a.research.trajectory_search import guarded_two_step_search as guarded
from s2a.research.trajectory_search import margin_two_step_search as margin

MAX_RESPONSES = 320
MAX_FEASIBILITY_ITERATIONS = 60
MAX_TOTAL_ITERATIONS = 120


class SharedResponseBudgetExhausted(RuntimeError):
    """A fallback response would exceed the shared per-decision budget."""


class SharedIterationBudgetExhausted(RuntimeError):
    """A fallback optimization phase has no shared iterations remaining."""


def isolated_margin_solver(capture):
    """Record the primary linear domain without mutating frozen modules."""
    old = margin.solve_margin_two_step
    namespace = old.__globals__.copy()

    def isolated_guarded():
        solver = guarded.isolated_solver()
        solver_namespace = solver.__globals__
        original_domain = solver_namespace["linear_domain"]

        def record_domain(*args, **kwargs):
            domain = original_domain(*args, **kwargs)
            if "domain" in capture:
                raise ValueError("The primary solver must build exactly one linear domain")
            capture["domain"] = {key: np.asarray(value).copy() for key, value in domain.items()}
            return domain

        solver_namespace["linear_domain"] = record_domain
        return solver

    namespace["guarded"] = SimpleNamespace(isolated_solver=isolated_guarded, OBJECTIVE_SCALE=guarded.OBJECTIVE_SCALE)
    function = FunctionType(old.__code__, namespace, old.__name__, old.__defaults__, old.__closure__)
    function.__kwdefaults__ = dict(old.__kwdefaults__ or {})
    return function


def reusable_initializer(seed, info, original_domain, domain_check):
    """Bind an already-computed start to exactly the same validated domain."""
    seed = finite(seed, (8, 6)).copy()
    if not isinstance(info, dict) or not 0 <= info.get("start_lp_attempts", -1) <= 2:
        raise ValueError("A primary initializer with bounded LP accounting is required")
    if info.get("seed_linear_safe") is True and info.get("warm_start_used") is False:
        start = seed.copy()
    elif info.get("warm_start_used") is True:
        start = finite(info.get("warm_start_commands"), (8, 6)).copy()
    else:
        return None
    if not isinstance(original_domain, dict) or not original_domain:
        raise ValueError("The primary linear domain must be captured")
    frozen_domain = {key: np.asarray(value).copy() for key, value in original_domain.items()}

    def initialize(current_seed, domain, interface):
        if not np.array_equal(finite(current_seed, (8, 6)), seed):
            raise ValueError("The fallback seed must match the primary seed exactly")
        if set(domain) != set(frozen_domain) or any(
            not np.array_equal(np.asarray(domain[key]), value) for key, value in frozen_domain.items()
        ):
            raise ValueError("The fallback linear domain must exactly match the primary domain")
        check = domain_check(start, domain, interface)
        if check.get("accepted") is not True:
            raise ValueError("The reused start must pass the original linear-domain checks")
        return start.copy(), dict(
            seed_linear_safe=bool(info["seed_linear_safe"]),
            start_lp_attempts=0,
            warm_start_used=bool(info["warm_start_used"]),
            warm_start_commands=start.copy(),
            check=check,
            primary_start_reused=True,
        )

    return initialize


def primary_iterations(phases):
    """Extract only completed iteration counts; unknown counts cannot be reused."""
    if not isinstance(phases, list):
        raise ValueError("The primary optimization phases must be recorded")
    names = [phase.get("phase") for phase in phases]
    if names not in ([], ["two_point_feasibility"], ["two_point_feasibility", "future_quality"]):
        raise ValueError("The primary optimization phase ordering is invalid")
    if any(phase.get("iterations") is None for phase in phases):
        return None
    counts = [phase["iterations"] for phase in phases]
    if any(type(count) is not int or count < 0 for count in counts):
        raise ValueError("Nonnegative integer iteration counts are required")
    first = counts[0] if counts else 0
    total = sum(counts)
    if first > MAX_FEASIBILITY_ITERATIONS or total > MAX_TOTAL_ITERATIONS:
        raise ValueError("The primary search exceeded the registered iteration budget")
    return first, total


def bounded_minimize(original_minimize, primary_first, primary_total, receipt):
    """Share feasibility and total iteration limits with the primary search."""
    if not (
        0 <= primary_first <= MAX_FEASIBILITY_ITERATIONS and primary_first <= primary_total <= MAX_TOTAL_ITERATIONS
    ):
        raise ValueError("Valid consumed primary iterations are required")
    calls = []

    def run(*args, **kwargs):
        if len(calls) >= 2:
            raise ValueError("The fallback has at most two optimization phases")
        options = dict(kwargs.get("options", {}))
        requested = options.get("maxiter")
        if type(requested) is not int or requested <= 0:
            raise ValueError("The original finite optimizer iteration limit is required")
        completed = sum(item["iterations"] for item in calls if item["iterations"] is not None)
        remaining_total = MAX_TOTAL_ITERATIONS - primary_total - completed
        remaining_first = MAX_FEASIBILITY_ITERATIONS - primary_first if not calls else remaining_total
        allowed = min(requested, remaining_total, remaining_first)
        if allowed <= 0:
            raise SharedIterationBudgetExhausted("No shared iterations remain for this fallback phase")
        options["maxiter"] = allowed
        kwargs["options"] = options
        record = dict(phase="two_point_feasibility" if not calls else "future_quality", limit=allowed, iterations=None)
        calls.append(record)
        result = original_minimize(*args, **kwargs)
        iterations = int(result.nit)
        if not 0 <= iterations <= allowed:
            raise ValueError("The fallback optimizer exceeded its shared iteration limit")
        record["iterations"] = iterations
        return result

    receipt["optimizer_calls"] = calls
    return run


def solve_margin_guarded_fallback(
    model, velocity, seed, previous, interface, site, tool, targets, twists, lower, upper, **kwargs
):
    """Use margin candidates first; guarded fallback shares all decision budgets."""
    seed = finite(seed, (8, 6)).copy()
    targets = finite(targets, (8, 4, 4)).copy()
    callback = kwargs.pop("on_response", None)
    if callback is not None and not callable(callback):
        raise ValueError("The optional response callback must be callable")
    counts = {"attempted": 0, "completed": 0}

    def response(event):
        if event not in counts:
            raise ValueError("Only attempted and completed response events are allowed")
        if event == "attempted" and counts["attempted"] >= MAX_RESPONSES:
            raise SharedResponseBudgetExhausted("The shared response budget is exhausted")
        counts[event] += 1
        if counts["completed"] > counts["attempted"]:
            raise ValueError("A response cannot complete before it is attempted")
        if callback is not None:
            callback(event)

    capture = {}
    primary = isolated_margin_solver(capture)(
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
        on_response=response,
        **kwargs,
    )
    primary_count = counts["completed"]
    if counts["attempted"] != primary_count or primary["response_evaluations"] != primary_count:
        raise ValueError("The primary solver must account for every completed response")
    iterations = primary_iterations(primary["search_phases"])
    info = primary.get("initialization")
    if not isinstance(info, dict) or not 0 <= info.get("start_lp_attempts", -1) <= 2:
        raise ValueError("The primary initializer must respect the shared LP budget")
    trace = dict(
        primary_status=primary["status"],
        primary_responses=primary_count,
        primary_phases=copy.deepcopy(primary["search_phases"]),
        primary_margin_evaluations=primary.get("margin_search", {}).get("margin_prefix_evaluations", 0),
        primary_feasibility_iterations=None if iterations is None else iterations[0],
        primary_total_iterations=None if iterations is None else iterations[1],
        fallback_status="not_attempted",
        fallback_responses=0,
        fallback_phases=[],
        fallback_initializer_reused=False,
        fallback_lp_attempts=0,
        selected_scope=None,
        response_budget=MAX_RESPONSES,
        feasibility_iteration_budget=MAX_FEASIBILITY_ITERATIONS,
        total_iteration_budget=MAX_TOTAL_ITERATIONS,
        start_lp_budget=2,
        optimizer_calls=[],
    )
    selected = primary.get("best_joint_safe")
    fallback = None
    if selected is not None:
        if primary["status"] != "two_step_margin_candidate" or not margin.margin_prefix_satisfied(
            selected["tcp_errors"]
        ):
            raise ValueError("The priority candidate must satisfy the fixed model margins")
        trace["selected_scope"] = "margin_two_step"
    elif primary["status"] != "two_step_margin_no_witness":
        trace["fallback_status"] = "primary_not_eligible"
    elif iterations is None:
        trace["fallback_status"] = "primary_iteration_count_unknown"
    elif primary_count >= MAX_RESPONSES:
        trace["fallback_status"] = "response_budget_exhausted"
    elif iterations[0] >= MAX_FEASIBILITY_ITERATIONS or iterations[1] >= MAX_TOTAL_ITERATIONS:
        trace["fallback_status"] = "iteration_budget_exhausted"
    else:
        solver = guarded.isolated_solver()
        namespace = solver.__globals__
        initializer = reusable_initializer(seed, info, capture.get("domain"), namespace["domain_check"])
        if initializer is None:
            trace["fallback_status"] = "primary_start_unavailable"
        else:

            def recorded_initializer(*args, **init_kwargs):
                start, reused_info = initializer(*args, **init_kwargs)
                trace["fallback_initializer_reused"] = True
                return start, reused_info

            namespace["nearest_linear_plan"] = recorded_initializer
            namespace["minimize"] = bounded_minimize(namespace["minimize"], *iterations, trace)
            try:
                fallback = solver(
                    model,
                    velocity,
                    seed.copy(),
                    previous,
                    interface,
                    site,
                    tool,
                    targets.copy(),
                    twists,
                    lower,
                    upper,
                    on_response=response,
                    **kwargs,
                )
                trace["fallback_status"] = fallback["status"]
                trace["fallback_phases"] = copy.deepcopy(fallback["search_phases"])
                trace["fallback_lp_attempts"] = fallback["initialization"]["start_lp_attempts"]
                if trace["fallback_lp_attempts"] != 0:
                    raise ValueError("The fallback must reuse the primary initializer without LPs")
                selected = fallback.get("best_joint_safe")
                if selected is not None:
                    if fallback["status"] != "two_step_candidate":
                        raise ValueError("The fallback candidate must satisfy the original two-point contract")
                    trace["selected_scope"] = "guarded_two_step"
            except (SharedResponseBudgetExhausted, SharedIterationBudgetExhausted) as error:
                trace["fallback_status"] = type(error).__name__
                selected = None
    trace["fallback_responses"] = counts["completed"] - primary_count
    if counts["attempted"] != counts["completed"] or counts["completed"] > MAX_RESPONSES:
        raise ValueError("Shared attempted/completed responses must close within budget")
    if fallback is not None and fallback["response_evaluations"] != trace["fallback_responses"]:
        raise ValueError("The fallback response accounting must match its completed responses")
    result = dict(primary)
    result.pop("best_joint_safe", None)
    result["response_evaluations"] = counts["completed"]
    result["initialization"] = dict(info, actual_model_rollouts=counts["completed"], seed_rollout_reused=True)
    result["shared_search"] = trace
    result["search_phases"] = [dict(phase, search="margin") for phase in trace["primary_phases"]] + [
        dict(phase, search="guarded") for phase in trace["fallback_phases"]
    ]
    if selected is None:
        result["status"] = "margin_guarded_no_witness"
    else:
        result["best_joint_safe"] = dict(selected, selection_scope=trace["selected_scope"])
        result["status"] = "margin_guarded_candidate"
    return result
