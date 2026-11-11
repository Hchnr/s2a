"""Compare a fixed two-pose model margin on the four prior causal inputs."""

import argparse
import copy
from pathlib import Path
from types import FunctionType, SimpleNamespace

import numpy as np

from s2a.infrastructure.publication import publish_result as publish
from s2a.research.trajectory_search import margin_two_step_search as margin
from scripts.simulation.runtime import prepare_two_step_viability as original_binding
from scripts.simulation.runtime import registered_static_phase
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.tracking.trajectory_search import audit_two_step_viability as original
from scripts.tracking.trajectory_search import compare_guarded_two_step_search as guarded

REPO = prefix.REPO
CONFIG = REPO / "configs/tracking/joint_control/margin_two_step_search_20261111.json"
PLAN = REPO / "plans/v0.0.1/tracking/joint_control/margin_two_step_search_20261111.md"
QA = REPO / "outputs/qa/margin_two_step_comparison_20261111/result.json"
DRY = REPO / "outputs/margin_two_step_search_dry_20261111/result.json"
OUTPUT = REPO / "outputs/margin_two_step_search_20261111"
PRIOR = {
    "path": str(REPO / "outputs/guarded_two_step_search_20261111/result.json"),
    "sha256": "0f6ed1926a5091eda36e6bf3794e588aa62d7b3ffadfba82afa1b289750a404b",
}
INDICES = (2, 6, 10, 11)


def source_refs():
    return guarded.source_refs() + [
        prefix.reference(path)
        for path in (
            Path(__file__),
            CONFIG,
            PLAN,
            REPO / "src/s2a/research/trajectory_search/margin_two_step_search.py",
            REPO / "tests/research/trajectory_search/test_margin_two_step_search.py",
            REPO / "tests/research/native_runtime/test_margin_two_step_search.py",
        )
    ]


def read_config():
    ref = prefix.reference(CONFIG)
    value = prefix.read(ref)
    prefix.require(
        value
        == dict(
            schema="s2a.margin-two-step-search.v1",
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
            feasibility_objective_scale=10000.0,
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
        "Exact matched decisions, fixed model margins and original budgets",
    )
    return ref, value


def load_inputs():
    config_ref, config = read_config()
    inputs = guarded.load_inputs()
    guarded_receipt = copy.deepcopy(guarded.input_receipt(inputs))
    refs = inputs["source_sha256"]
    refs[config_ref["path"]] = config_ref["sha256"]
    prior = prefix.read(PRIOR, refs)
    prefix.require(
        prior["complete"] is prior["passed"] is prior["sources_unchanged"] is True
        and prior["control_qualified"] is False
        and prior["physics_steps"] == prior["world_model_calls"] == prior["fresh_cases"] == 0,
        "Closed guarded static comparison",
    )
    registration = prefix.read(prior["registration_ref"], refs)
    prefix.require(
        registration["inputs"] == guarded_receipt and registration["source_refs"] == guarded.source_refs(),
        "Exactly the guarded comparison's causal inputs and frozen sources",
    )
    prior_cases = {}
    for ref in prior["case_refs"]:
        item = prefix.read(ref, refs)
        prefix.require(item["complete"] is item["passed"] is True, "Closed guarded comparison case")
        index = item["case"]["case_index"]
        prefix.require(index not in prior_cases, "Unique guarded comparison case")
        prior_cases[index] = item["data"]
    prefix.require(sorted(prior_cases) == list(INDICES), "All four guarded development decisions")
    for case in inputs["cases"]:
        old = prior_cases[case["decision_spec"]["case_index"]]
        prefix.require(old["case"] == case["decision_spec"], "Same pre-stop decision")
        witness = old["solver"].get("best_joint_safe")
        case["guarded_static"] = dict(
            input_sha256=old["input_sha256"],
            status=old["solver"]["status"],
            response_evaluations=old["solver"]["response_evaluations"],
            model_prefix_witness=old["model_prefix_witness"],
            seed_model_first_error=old["seed_model_first_error"],
            seed_observed_first_error=old["seed_observed_first_error"],
            prefix_tcp_errors=None if witness is None else witness["tcp_errors"][:2],
        )
    inputs.update(guarded_receipt=guarded_receipt, margin_config_ref=config_ref, config=config)
    return inputs


def input_receipt(inputs):
    return dict(
        guarded=inputs["guarded_receipt"],
        margin_config_ref=inputs["margin_config_ref"],
        prior_ref=PRIOR,
        source_sha256=inputs["source_sha256"],
    )


def isolated_analyze():
    old = guarded.isolated_analyze()
    namespace = old.__globals__.copy()
    namespace["solve_two_step_viability"] = margin.solve_margin_two_step
    function = FunctionType(old.__code__, namespace, old.__name__, old.__defaults__, old.__closure__)
    function.__kwdefaults__ = dict(old.__kwdefaults__ or {})
    return function


def analyze(case, inputs, stats):
    result = isolated_analyze()(case, inputs, stats)
    prior = case["guarded_static"]
    prefix.require(result["input_sha256"] == prior["input_sha256"], "Guarded and margin inputs are byte-identical")
    np.testing.assert_allclose(result["seed_model_first_error"], prior["seed_model_first_error"], atol=1e-12, rtol=0)
    np.testing.assert_array_equal(result["seed_observed_first_error"], prior["seed_observed_first_error"])
    original_audit = result["independent_audit"]
    result["external_two_step_witness"] = result["model_prefix_witness"]
    if original_audit["status"] == "passed":
        errors = original_audit["verification"]["tcp_errors"]
        accepted = margin.margin_prefix_satisfied(errors)
        result["independent_margin_audit"] = dict(
            status="passed" if accepted else "rejected",
            position_m=np.linalg.norm(np.asarray(errors)[:2, :3], axis=1),
            rotation_rad=np.linalg.norm(np.asarray(errors)[:2, 3:], axis=1),
        )
        result["model_prefix_witness"] = bool(result["model_prefix_witness"] and accepted)
    else:
        result["independent_margin_audit"] = dict(status="not_attempted")
        result["model_prefix_witness"] = False
    result["guarded_static"] = prior
    result["scope"] = "matched static model comparison; independent margins do not establish physical tracking"
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
            worker_module="scripts.tracking.trajectory_search.compare_margin_two_step_search",
            load_inputs=load_inputs,
            input_receipt=input_receipt,
            source_refs=source_refs,
            summarize=original.summarize,
        )
    else:
        isolated_worker()(args.worker)


if __name__ == "__main__":
    main()
