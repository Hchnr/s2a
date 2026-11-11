"""Compare shared-budget margin-first fallback on four prior causal inputs."""

import argparse
import copy
from pathlib import Path
from types import FunctionType, SimpleNamespace

import numpy as np

from s2a.infrastructure.publication import publish_result as publish
from s2a.research.trajectory_search import margin_guarded_fallback as policy
from s2a.research.trajectory_search import margin_two_step_search as margin
from scripts.simulation.runtime import prepare_two_step_viability as original_binding
from scripts.simulation.runtime import registered_static_phase
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.tracking.trajectory_search import audit_two_step_viability as original
from scripts.tracking.trajectory_search import compare_margin_two_step_search as previous

REPO = prefix.REPO
CONFIG = REPO / "configs/tracking/joint_control/margin_guarded_fallback_20261111.json"
PLAN = REPO / "plans/v0.0.1/tracking/joint_control/margin_guarded_fallback_20261111.md"
QA = REPO / "outputs/qa/margin_guarded_fallback_20261111/result.json"
DRY = REPO / "outputs/margin_guarded_fallback_dry_20261111/result.json"
OUTPUT = REPO / "outputs/margin_guarded_fallback_20261111"
PRIOR = {
    "path": str(REPO / "outputs/margin_two_step_search_20261111/result.json"),
    "sha256": "117800ab09af1c44bc44670a30e730f8b45e53331a6c3652d483d93dc7bf6bba",
}
INDICES = (2, 6, 10, 11)


def source_refs():
    return previous.source_refs() + [
        prefix.reference(path)
        for path in (
            Path(__file__),
            CONFIG,
            PLAN,
            REPO / "src/s2a/research/trajectory_search/margin_guarded_fallback.py",
            REPO / "tests/research/trajectory_search/test_margin_guarded_fallback.py",
            REPO / "tests/research/native_runtime/test_margin_guarded_fallback.py",
        )
    ]


def read_config():
    ref = prefix.reference(CONFIG)
    value = prefix.read(ref)
    prefix.require(
        value
        == dict(
            schema="s2a.margin-guarded-fallback.v1",
            cases=original_binding.CASES,
            seed=43,
            control_dt_s=0.05,
            inner_dt_s=0.002,
            joint_buffer_rad=0.05,
            position_cap_m=margin.POSITION_CAP_M,
            rotation_cap_rad=margin.ROTATION_CAP_RAD,
            model_position_margin_m=margin.POSITION_MARGIN_M,
            model_rotation_margin_rad=margin.ROTATION_MARGIN_RAD,
            selection_tolerance=margin.SELECTION_TOLERANCE,
            arm_rate_limit=0.25,
            braking_time_s=0.1,
            plan_horizon=8,
            hard_pose_prefix=2,
            max_responses_per_decision=policy.MAX_RESPONSES,
            max_feasibility_iterations_per_decision=policy.MAX_FEASIBILITY_ITERATIONS,
            max_total_iterations_per_decision=policy.MAX_TOTAL_ITERATIONS,
            max_start_lp_per_decision=2,
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
        "Exact causal decisions and shared response, iteration and LP budgets",
    )
    return ref, value


def load_inputs():
    config_ref, config = read_config()
    inputs = previous.load_inputs()
    prior_receipt = copy.deepcopy(previous.input_receipt(inputs))
    refs = inputs["source_sha256"]
    refs[config_ref["path"]] = config_ref["sha256"]
    prior = prefix.read(PRIOR, refs)
    prefix.require(
        prior["complete"] is prior["passed"] is prior["sources_unchanged"] is True
        and prior["control_qualified"] is False
        and prior["physics_steps"] == prior["world_model_calls"] == prior["fresh_cases"] == 0,
        "Closed prior margin static comparison",
    )
    registration = prefix.read(prior["registration_ref"], refs)
    prefix.require(
        registration["inputs"] == prior_receipt and registration["source_refs"] == previous.source_refs(),
        "Identical margin comparison inputs and frozen sources",
    )
    old_cases = {}
    for ref in prior["case_refs"]:
        item = prefix.read(ref, refs)
        index = item["case"]["case_index"]
        prefix.require(item["complete"] is item["passed"] is True and index not in old_cases, "Unique closed case")
        old_cases[index] = item["data"]
    prefix.require(sorted(old_cases) == list(INDICES), "All four matched development decisions")
    for case in inputs["cases"]:
        old = old_cases[case["decision_spec"]["case_index"]]
        prefix.require(old["case"] == case["decision_spec"], "Same pre-stop decision")
        witness = old["solver"].get("best_joint_safe")
        case["margin_static"] = dict(
            input_sha256=old["input_sha256"],
            status=old["solver"]["status"],
            response_evaluations=old["solver"]["response_evaluations"],
            model_prefix_witness=old["model_prefix_witness"],
            seed_model_first_error=old["seed_model_first_error"],
            seed_observed_first_error=old["seed_observed_first_error"],
            prefix_tcp_errors=None if witness is None else witness["tcp_errors"][:2],
        )
    inputs.update(prior_margin_receipt=prior_receipt, shared_config_ref=config_ref, config=config)
    return inputs


def input_receipt(inputs):
    return dict(
        previous=inputs["prior_margin_receipt"],
        shared_config_ref=inputs["shared_config_ref"],
        prior_ref=PRIOR,
        source_sha256=inputs["source_sha256"],
    )


def isolated_analyze():
    old = previous.isolated_analyze()
    namespace = old.__globals__.copy()
    namespace["solve_two_step_viability"] = policy.solve_margin_guarded_fallback
    function = FunctionType(old.__code__, namespace, old.__name__, old.__defaults__, old.__closure__)
    function.__kwdefaults__ = dict(old.__kwdefaults__ or {})
    return function


def analyze(case, inputs, stats):
    result = isolated_analyze()(case, inputs, stats)
    prior = case["margin_static"]
    prefix.require(result["input_sha256"] == prior["input_sha256"], "The compared solver inputs are byte-identical")
    np.testing.assert_allclose(result["seed_model_first_error"], prior["seed_model_first_error"], atol=1e-12, rtol=0)
    np.testing.assert_array_equal(result["seed_observed_first_error"], prior["seed_observed_first_error"])
    candidate = result["solver"].get("best_joint_safe")
    if candidate is None:
        result["independent_scope_audit"] = dict(status="not_attempted")
    elif result["independent_audit"]["status"] != "passed":
        result["independent_scope_audit"] = dict(status="rejected", scope=candidate["selection_scope"])
        result["model_prefix_witness"] = False
    else:
        scope = candidate["selection_scope"]
        prefix.require(scope in ("margin_two_step", "guarded_two_step"), "Known two-point selection scope")
        errors = result["independent_audit"]["verification"]["tcp_errors"]
        accepted = scope == "guarded_two_step" or margin.margin_prefix_satisfied(errors)
        result["independent_scope_audit"] = dict(status="passed" if accepted else "rejected", scope=scope)
        result["model_prefix_witness"] = bool(result["model_prefix_witness"] and accepted)
    result["margin_static"] = prior
    result["guarded_static"] = case["guarded_static"]
    result["scope"] = "matched static model comparison; no physical or full-horizon qualification"
    return result


def isolated_worker():
    old = original.worker
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
            worker_module="scripts.tracking.trajectory_search.compare_margin_guarded_fallback",
            load_inputs=load_inputs,
            input_receipt=input_receipt,
            source_refs=source_refs,
            summarize=original.summarize,
        )
    else:
        isolated_worker()(args.worker)


if __name__ == "__main__":
    main()
