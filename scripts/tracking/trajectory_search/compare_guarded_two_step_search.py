"""Compare guarded two-step search on exactly the prior four model inputs."""

import argparse
import copy
from pathlib import Path
from types import FunctionType, SimpleNamespace

import numpy as np

from s2a.infrastructure.publication import publish_result as publish
from s2a.research.trajectory_search.guarded_two_step_search import (
    OBJECTIVE_SCALE,
    SEARCH_GUARD,
    solve_guarded_two_step,
)
from scripts.simulation.runtime import prepare_two_step_viability as previous_binding
from scripts.simulation.runtime import registered_static_phase
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.tracking.trajectory_search import audit_two_step_viability as previous

REPO = prefix.REPO
CONFIG = REPO / "configs/tracking/joint_control/guarded_two_step_search_20261111.json"
PLAN = REPO / "plans/v0.0.1/tracking/joint_control/guarded_two_step_search_20261111.md"
QA = REPO / "outputs/qa/guarded_two_step_search_20261111/result.json"
DRY = REPO / "outputs/guarded_two_step_search_dry_20261111/result.json"
OUTPUT = REPO / "outputs/guarded_two_step_search_20261111"
PRIOR = {
    "path": str(REPO / "outputs/two_step_viability_20261111/result.json"),
    "sha256": "d9e4a6aabe8e5fb29a0c5e9feeaaf5ac020be5461324dd7e36a22f0db600825f",
}
INDICES = (2, 6, 10, 11)


def source_refs():
    return previous_binding.source_refs() + [
        prefix.reference(path)
        for path in (
            Path(__file__),
            CONFIG,
            PLAN,
            REPO / "src/s2a/research/trajectory_search/guarded_two_step_search.py",
            REPO / "tests/research/trajectory_search/test_guarded_two_step_search.py",
            REPO / "tests/research/native_runtime/test_guarded_two_step_search.py",
        )
    ]


def read_config():
    ref = prefix.reference(CONFIG)
    value = prefix.read(ref)
    prefix.require(
        value
        == dict(
            schema="s2a.guarded-two-step-search.v1",
            cases=previous_binding.CASES,
            seed=43,
            control_dt_s=0.05,
            inner_dt_s=0.002,
            joint_buffer_rad=0.05,
            position_cap_m=0.01,
            rotation_cap_rad=0.05,
            arm_rate_limit=0.25,
            braking_time_s=0.1,
            plan_horizon=8,
            hard_pose_prefix=2,
            normalized_search_guard=SEARCH_GUARD,
            feasibility_objective_scale=OBJECTIVE_SCALE,
            max_responses_per_solve=320,
            max_feasibility_iterations=60,
            max_total_iterations=120,
            max_start_lp_per_solve=2,
            max_start_lp_calls=8,
            max_models=4,
            max_static_forward_queries=4,
            max_preparation_seed_rollouts=4,
            max_solver_calls=4,
            max_solver_responses=1280,
            max_audit_endpoint_queries=800,
            max_audit_transition_steps=10400,
            max_workers=4,
            worker_timeout_seconds=1800,
            physics_steps=0,
            world_model_calls=0,
            fresh_cases=0,
            automatic_retry=False,
        ),
        "Exact matched four-state objective and original budgets",
    )
    return ref, value


def load_inputs():
    config_ref, config = read_config()
    inputs = previous_binding.load_inputs()
    old_receipt = copy.deepcopy(previous_binding.input_receipt(inputs))
    refs = inputs["source_sha256"]
    refs[config_ref["path"]] = config_ref["sha256"]
    prior = prefix.read(PRIOR, refs)
    prefix.require(
        prior["complete"] is prior["passed"] is prior["sources_unchanged"] is True
        and prior["control_qualified"] is False
        and prior["physics_steps"] == prior["world_model_calls"] == prior["fresh_cases"] == 0,
        "Closed prior two-step static comparison",
    )
    registration = prefix.read(prior["registration_ref"], refs)
    prefix.require(
        registration["inputs"] == old_receipt and registration["source_refs"] == previous_binding.source_refs(),
        "Same prior inputs and frozen source identity",
    )
    prior_cases = {}
    for ref in prior["case_refs"]:
        item = prefix.read(ref, refs)
        prefix.require(item["complete"] is item["passed"] is True, "Closed prior case")
        index = item["case"]["case_index"]
        prefix.require(index not in prior_cases, "Unique prior case")
        prior_cases[index] = item["data"]
    prefix.require(sorted(prior_cases) == list(INDICES), "All four prior development cases")
    for case in inputs["cases"]:
        index = case["decision_spec"]["case_index"]
        old = prior_cases[index]
        prefix.require(old["case"] == case["decision_spec"], "Same pre-stop decision")
        case["prior_static"] = dict(
            input_sha256=old["input_sha256"],
            status=old["solver"]["status"],
            response_evaluations=old["solver"]["response_evaluations"],
            model_prefix_witness=old["model_prefix_witness"],
            seed_model_first_error=old["seed_model_first_error"],
            seed_observed_first_error=old["seed_observed_first_error"],
        )
    inputs.update(guard_config_ref=config_ref, config=config)
    return inputs


def input_receipt(inputs):
    return dict(
        previous=previous_binding.input_receipt(inputs),
        guard_config_ref=inputs["guard_config_ref"],
        prior_ref=PRIOR,
        source_sha256=inputs["source_sha256"],
    )


def isolated_analyze():
    old = previous.analyze
    namespace = old.__globals__.copy()
    namespace["solve_two_step_viability"] = solve_guarded_two_step
    function = FunctionType(old.__code__, namespace, old.__name__, old.__defaults__, old.__closure__)
    function.__kwdefaults__ = dict(old.__kwdefaults__ or {})
    return function


def analyze(case, inputs, stats):
    result = isolated_analyze()(case, inputs, stats)
    prior = case["prior_static"]
    prefix.require(result["input_sha256"] == prior["input_sha256"], "New and old solver inputs are byte-identical")
    np.testing.assert_allclose(result["seed_model_first_error"], prior["seed_model_first_error"], atol=1e-12, rtol=0)
    np.testing.assert_array_equal(result["seed_observed_first_error"], prior["seed_observed_first_error"])
    result["prior_static"] = prior
    return result


def isolated_worker():
    old = previous.worker
    namespace = old.__globals__.copy()
    namespace.update(
        OUTPUT=OUTPUT,
        binding=SimpleNamespace(load_inputs=load_inputs, input_receipt=input_receipt, source_refs=source_refs),
        analyze=analyze,
    )
    function = FunctionType(old.__code__, namespace, old.__name__, old.__defaults__, old.__closure__)
    function.__kwdefaults__ = dict(old.__kwdefaults__ or {})
    return function


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
            worker_module="scripts.tracking.trajectory_search.compare_guarded_two_step_search",
            load_inputs=load_inputs,
            input_receipt=input_receipt,
            source_refs=source_refs,
            summarize=previous.summarize,
        )
    else:
        isolated_worker()(args.worker)


if __name__ == "__main__":
    main()
