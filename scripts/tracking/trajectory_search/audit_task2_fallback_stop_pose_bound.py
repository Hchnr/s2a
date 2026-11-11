"""Apply frozen first-pose outer bounds to replayed fallback stops."""

import argparse
import copy
from pathlib import Path
from types import FunctionType, SimpleNamespace

from s2a.infrastructure.publication import publish_result as publish
from scripts.simulation.runtime import prepare_task2_cached_first_pose as previous_binding
from scripts.simulation.runtime import registered_static_phase
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.tracking.trajectory_search import audit_stopped_pose_safety_bound as original
from scripts.tracking.trajectory_search import audit_task2_cached_first_pose as prior_runner

REPO = prefix.REPO
CONFIG = REPO / "configs/tracking/joint_control/task2_fallback_stop_pose_bound_20261111.json"
PLAN = REPO / "plans/v0.0.1/tracking/joint_control/task2_fallback_stop_pose_bound_20261111.md"
QA = REPO / "outputs/qa/task2_fallback_stop_pose_bound_20261111/result.json"
DRY = REPO / "outputs/task2_fallback_stop_pose_bound_dry_20261111/result.json"
OUTPUT = REPO / "outputs/task2_fallback_stop_pose_bound_20261111"
PRIOR = {
    "path": str(REPO / "outputs/task2_cached_first_pose_20261111/result.json"),
    "sha256": "f1752184a8f20957303e49b655020b1d95387aedc089df328fba58ac858ff898",
}
CASES = [dict(case_index=10, step=33), dict(case_index=11, step=85)]
INDICES = (10, 11)


def source_refs():
    return prior_runner.source_refs() + [
        prefix.reference(path)
        for path in (
            Path(__file__),
            CONFIG,
            PLAN,
            Path(original.__file__),
            REPO / "src/s2a/research/joint_control/tracking/stopped_pose_safety_bound.py",
            REPO / "scripts/verification/reference_ports/check_stopped_pose_safety_bound.py",
            REPO / "tests/research/joint_control/tracking/test_stopped_pose_safety_bound.py",
            REPO / "tests/research/native_runtime/test_stopped_pose_safety_bound_check.py",
            REPO / "tests/research/native_runtime/test_task2_fallback_stop_pose_bound.py",
        )
    ]


def read_config():
    ref = prefix.reference(CONFIG)
    config = prefix.read(ref)
    prefix.require(
        config
        == dict(
            schema="s2a.task2-fallback-stop-pose-bound.v1",
            cases=CASES,
            seed=43,
            control_dt_s=0.05,
            inner_dt_s=0.002,
            joint_buffer_rad=0.05,
            braking_time_s=0.1,
            position_cap_m=0.01,
            rotation_cap_rad=0.05,
            arm_rate_limit=0.25,
            scopes=["box", "first_interval_safety", "eight_interval_safety"],
            position_weights=[0.0, 0.25, 0.5, 0.75, 1.0],
            normalized_numerical_allowance=1e-7,
            terminal_direction_included=False,
            max_models=2,
            max_static_forward_queries=2,
            max_seed_rollouts=2,
            max_support_lp_calls=20,
            max_independent_first_responses=2,
            max_independent_transition_steps=50,
            max_workers=2,
            worker_timeout_seconds=1200,
            physics_steps=0,
            world_model_calls=0,
            fresh_cases=0,
            automatic_retry=False,
        ),
        "Exact two fallback stops, relaxed domains and support budgets",
    )
    return ref, config


def load_inputs():
    config_ref, config = read_config()
    old = previous_binding.load_inputs()
    previous_receipt = copy.deepcopy(previous_binding.input_receipt(old))
    refs = dict(old["source_sha256"])
    refs[config_ref["path"]] = config_ref["sha256"]
    prior = prefix.read(PRIOR, refs)
    prefix.require(
        prior["complete"] is prior["passed"] is prior["sources_unchanged"] is True
        and prior["control_qualified"] is False
        and prior["physics_steps"] == prior["world_model_calls"] == prior["fresh_cases"] == 0,
        "Closed previous cached-plan assessment",
    )
    registration = prefix.read(prior["registration_ref"], refs)
    prefix.require(
        registration["inputs"] == previous_receipt and registration["source_refs"] == prior_runner.source_refs(),
        "Same replayed feedback inputs and frozen search sources",
    )
    prior_cases = {}
    for ref in prior["case_refs"]:
        item = prefix.read(ref, refs)
        index = item["case"]["case_index"]
        prefix.require(item["complete"] is item["passed"] is True and index not in prior_cases, "Unique closed stop")
        prior_cases[index] = item
    prefix.require(sorted(prior_cases) == list(INDICES), "Both replayed fallback stops")
    cases = []
    for spec in CASES:
        index = spec["case_index"]
        case = dict(next(item for item in old["cases"] if item["stop_spec"]["case_index"] == index))
        prior_case = prior_cases[index]
        prefix.require(
            prior_case["case"] == dict(case_index=index, stop_step=spec["step"])
            and all(row["cache_inspection"]["first_pose_candidates"] == 0 for row in prior_case["data"]["outcomes"]),
            "Exact stop and absent first-pose candidates in finite cached searches",
        )
        completed = case["stop_inputs"]
        case.update(
            bound_spec=spec,
            bound_seed=completed["shifted"],
            bound_profile=completed["profile"],
            closest_commands=completed["shifted"],
            closest_error=completed["recorded_shifted_errors"][0],
            seed_scope="last_completed_plan_shifted_not_original_stop_seed",
            previous_fingerprint=None,
        )
        cases.append(case)
    return dict(
        config_ref=config_ref,
        config=config,
        cases=cases,
        asset_refs=old["asset_refs"],
        previous_receipt=previous_receipt,
        source_sha256=refs,
    )


def input_receipt(inputs):
    return dict(
        previous=inputs["previous_receipt"],
        config_ref=inputs["config_ref"],
        prior_ref=PRIOR,
        cases=CASES,
        asset_refs=inputs["asset_refs"],
        source_sha256=inputs["source_sha256"],
    )


def isolated_worker():
    old = original.worker
    namespace = old.__globals__.copy()
    namespace.update(
        OUTPUT=OUTPUT,
        binding=SimpleNamespace(load_inputs=load_inputs, input_receipt=input_receipt, source_refs=source_refs),
    )
    function = FunctionType(old.__code__, namespace, old.__name__, old.__defaults__, old.__closure__)
    function.__kwdefaults__ = dict(old.__kwdefaults__ or {})
    return function


def summarize(cases, config):
    prefix.require([item["case"] for item in cases] == CASES, "Exact replayed fallback stop identities")
    keys = (
        "model_initializations_completed",
        "forward_queries_completed",
        "baseline_rollouts",
        "support_lp_attempted",
        "support_lp_completed",
        "independent_first_responses_charged",
        "independent_first_responses_completed",
        "independent_transition_steps_charged",
        "independent_transition_steps_completed",
    )
    counts = {key: sum(item["stats"][key] for item in cases) for key in keys}
    prefix.require(
        counts["model_initializations_completed"] == config["max_models"] == 2
        and counts["forward_queries_completed"] == config["max_static_forward_queries"] == 2
        and counts["baseline_rollouts"] == config["max_seed_rollouts"] == 2
        and counts["support_lp_attempted"] == counts["support_lp_completed"] == config["max_support_lp_calls"] == 20
        and counts["independent_first_responses_charged"]
        == counts["independent_first_responses_completed"]
        == config["max_independent_first_responses"]
        == 2
        and counts["independent_transition_steps_charged"]
        == counts["independent_transition_steps_completed"]
        == config["max_independent_transition_steps"]
        == 50,
        "Exact aggregate support and independent scalar-response costs",
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
        inputs = load_inputs()
        print(
            publish(
                DRY,
                dict(
                    complete=True,
                    passed=True,
                    inputs=input_receipt(inputs),
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
            worker_module="scripts.tracking.trajectory_search.audit_task2_fallback_stop_pose_bound",
            load_inputs=load_inputs,
            input_receipt=input_receipt,
            source_refs=source_refs,
            summarize=summarize,
        )
    else:
        isolated_worker()(args.worker)


if __name__ == "__main__":
    main()
