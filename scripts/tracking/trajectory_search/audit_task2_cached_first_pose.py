"""Audit first-pose candidates already evaluated by guarded stop searches."""

import argparse
from pathlib import Path
from types import FunctionType

from s2a.infrastructure.publication import publish_result as publish
from s2a.research.joint_control.tracking.full_horizon_quality_audit import audit_full_horizon_witness
from s2a.research.trajectory_search.cached_first_pose import collect_cached_first_pose
from scripts.simulation.runtime import prepare_task2_cached_first_pose as binding
from scripts.simulation.runtime import registered_static_phase
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.tracking.trajectory_search import compare_task2_shifted_seed_search as comparison

REPO = prefix.REPO
QA = REPO / "outputs/qa/task2_cached_first_pose_20261111/result.json"
DRY = REPO / "outputs/task2_cached_first_pose_dry_20261111/result.json"
OUTPUT = REPO / "outputs/task2_cached_first_pose_20261111"
INDICES = (10, 11)


def source_refs():
    return binding.source_refs() + [
        prefix.reference(path)
        for path in (
            Path(__file__),
            Path(comparison.__file__),
            REPO / "tests/research/native_runtime/test_task2_cached_first_pose_runtime.py",
            REPO / "tests/research/native_runtime/test_task2_shifted_seed_search.py",
        )
    ]


def isolated_analyze(captures):
    """Replace only the comparison's solve call in an isolated namespace."""

    def observed(*args, **kwargs):
        result = collect_cached_first_pose(*args, **kwargs)
        captures.append((args, kwargs, result))
        return result["primary"]

    old = comparison.analyze
    namespace = old.__globals__.copy()
    namespace["solve_guarded_two_step"] = observed
    function = FunctionType(old.__code__, namespace, old.__name__, old.__defaults__, old.__closure__)
    function.__kwdefaults__ = dict(old.__kwdefaults__ or {})
    return function


def audit_cached(record, case, stats):
    """Audit one existing first-only model candidate without another rollout."""
    args, kwargs, result = record
    candidate = result["best_first_pose"]
    if candidate is None:
        return dict(status="not_attempted")
    model, velocity, _seed, previous, interface, site, tool, targets, _twists, lower, upper = args
    contract = case["controller"]["contract"]
    stats["independent_audits_attempted"] += 1
    stats["independent_endpoint_queries_charged"] += 200
    stats["independent_transition_steps_charged"] += 2600
    try:
        checked = audit_full_horizon_witness(
            candidate,
            model["mass"],
            model["jacobian"],
            contract["kp"],
            contract["kd"],
            model["q"],
            velocity,
            case["initial_joint"],
            interface,
            previous,
            site,
            tool,
            targets,
            lower,
            upper,
            past_inner_error_rad=kwargs["past_inner_error_rad"],
            enforce_pose_caps=False,
            enforce_braking=True,
        )
        prefix.require(
            checked["passed"] is checked["first_pose_caps_satisfied"] is checked["braking_constraint_enforced"] is True
            and checked["independent_endpoint_queries"] == 200
            and checked["independent_transition_steps"] == 2600,
            "Independent first-pose and complete model safety audit",
        )
        stats["independent_audits_passed"] += 1
        return dict(status="passed", verification=checked)
    except (AssertionError, ValueError) as error:
        return dict(status="rejected", error=dict(type=type(error).__name__, message=str(error)))


def analyze(case, inputs, stats):
    adapted = dict(case)
    adapted["saved_lp"] = dict(
        commands=case["stop_inputs"]["lp_start"],
        prior_response_evaluations=case["stop_inputs"]["prior_primary_responses"],
    )
    adapted["prior_transition"] = dict(
        next_forecast=dict(dynamics_refreshed=case["stop_inputs"]["recorded_shifted_errors"][0])
    )
    captures = []
    result = isolated_analyze(captures)(adapted, inputs, stats)
    prefix.require(len(captures) == len(result["outcomes"]) == 2, "Both bounded guarded searches were inspected")
    for record, row in zip(captures, result["outcomes"], strict=True):
        inspected = record[2]
        prefix.require(
            row["solver"] is inspected["primary"]
            and inspected["recorded_rollout_calls"] == row["solver"]["response_evaluations"]
            and inspected["extra_model_responses"] == 0,
            "Inspection retains the original primary result without extra responses",
        )
        if row["solver"].get("best_joint_safe") is not None:
            cached_audit = dict(status="primary_two_point_candidate_takes_priority")
        else:
            cached_audit = audit_cached(record, case, stats)
        row["cache_inspection"] = dict(
            recorded_rollout_calls=inspected["recorded_rollout_calls"],
            unique_safe_candidates=inspected["unique_safe_candidates"],
            first_pose_candidates=inspected["first_pose_candidates"],
            candidate=inspected["best_first_pose"],
            independent_audit=cached_audit,
            independently_audited_first_pose_candidate=bool(cached_audit["status"] == "passed"),
            extra_model_responses=0,
        )
    result["scope"] = "static cached-plan inspection at replayed stops; no physical dispatch or full task claim"
    return result


def worker(index):
    prefix.environment()
    registration = prefix.read(prefix.reference(OUTPUT / "registration.json"))
    inputs = binding.load_inputs()
    prefix.require(
        binding.input_receipt(inputs) == registration["inputs"] and source_refs() == registration["source_refs"],
        "Frozen fallback stop inputs and source identity",
    )
    case = next(item for item in inputs["cases"] if item["stop_spec"]["case_index"] == index)
    stats = comparison.new_stats()
    result = dict(complete=False, passed=False, control_qualified=False, case=case["stop_spec"], stats=stats)
    try:
        data = analyze(case, inputs, stats)
        comparison.require_case_counts(stats)
        result.update(complete=True, passed=True, data=data)
    except BaseException as error:
        result["error"] = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        print(publish(OUTPUT / f"case_{index:02d}" / "result.json", result), flush=True)


def summarize(cases, config):
    prefix.require([item["case"] for item in cases] == binding.CASES, "Exact replayed stop identities")
    for item in cases:
        comparison.require_case_counts(item["stats"])
        prefix.require(
            [row["seed"] for row in item["data"]["outcomes"]] == list(binding.SEEDS), "Both saved causal seeds"
        )
    keys = (
        "model_initializations_completed",
        "forward_queries_completed",
        "baseline_rollouts",
        "additional_preparation_rollouts",
        "solver_calls_completed",
        "solver_responses_completed",
        "solver_lp_actual",
        "independent_audits_attempted",
        "independent_audits_passed",
        "independent_endpoint_queries_charged",
        "independent_transition_steps_charged",
    )
    counts = {key: sum(item["stats"][key] for item in cases) for key in keys}
    prefix.require(
        counts["model_initializations_completed"] == config["max_models"] == 2
        and counts["forward_queries_completed"] == config["max_static_forward_queries"] == 2
        and counts["baseline_rollouts"] + counts["additional_preparation_rollouts"]
        == config["max_preparation_rollouts"]
        == 4
        and counts["solver_calls_completed"] == config["max_solver_calls"] == 4
        and counts["solver_responses_completed"] <= config["max_solver_responses"]
        and counts["independent_audits_attempted"] <= config["max_independent_audits"]
        and counts["independent_endpoint_queries_charged"] <= config["max_independent_endpoint_queries"]
        and counts["independent_transition_steps_charged"] <= config["max_independent_transition_steps"],
        "Registered cached-plan search and independent audit budgets",
    )
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true")
    group.add_argument("--run", action="store_true")
    group.add_argument("--worker", type=int, choices=INDICES)
    args = parser.parse_args()
    prefix.environment()
    if args.dry_run:
        inputs = binding.load_inputs()
        print(
            publish(
                DRY,
                dict(
                    complete=True,
                    passed=True,
                    inputs=binding.input_receipt(inputs),
                    source_refs=source_refs(),
                    model_initializations=0,
                    physics_steps=0,
                ),
            )
        )
    elif args.run:
        registered_static_phase.execute_static_phase(
            root=OUTPUT,
            qa_path=QA,
            dry_path=DRY,
            indices=INDICES,
            worker_module="scripts.tracking.trajectory_search.audit_task2_cached_first_pose",
            load_inputs=binding.load_inputs,
            input_receipt=binding.input_receipt,
            source_refs=source_refs,
            summarize=summarize,
        )
    else:
        worker(args.worker)


if __name__ == "__main__":
    main()
