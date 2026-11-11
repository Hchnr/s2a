"""Compare first-pose search seeds at the replayed terminal058 stop."""

import argparse
import copy
from pathlib import Path
from types import FunctionType

import numpy as np

from s2a.geometry import inverse, pose_error
from s2a.infrastructure.publication import publish_result as publish
from s2a.references.generated_twist import reference_world_twist
from s2a.research.joint_control.preview.interior_joint_preview import rollout_joint_preview
from s2a.research.joint_control.tracking.full_horizon_quality_audit import audit_full_horizon_witness
from s2a.research.trajectory_search import staged_first_feasibility as staged
from s2a.research.trajectory_search.feasible_start_restoration import domain_check, linear_domain
from scripts.simulation.runtime import registered_static_phase
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.tracking.trajectory_search import audit_early_quality_feasibility as context
from scripts.tracking.trajectory_search import audit_task2_fallback_stop_pose_bound as previous
from scripts.tracking.trajectory_search import compare_task2_shifted_seed_search as comparison

REPO = prefix.REPO
CONFIG = REPO / "configs/tracking/joint_control/task2_terminal85_first_search_20261111.json"
PLAN = REPO / "plans/v0.0.1/tracking/joint_control/task2_terminal85_first_search_20261111.md"
QA = REPO / "outputs/qa/task2_terminal85_first_search_20261111/result.json"
DRY = REPO / "outputs/task2_terminal85_first_search_dry_20261111/result.json"
OUTPUT = REPO / "outputs/task2_terminal85_first_search_20261111"
PRIOR = {
    "path": str(REPO / "outputs/task2_fallback_stop_pose_bound_20261111/result.json"),
    "sha256": "194226209fbc97c72b6c65e4eb9929f21bd955bdd2bcd5a9913b926b051da55b",
}
SEEDS = ("recorded_lp_start", "shifted_completed_plan")
INDICES = (11,)


def source_refs():
    return previous.source_refs() + [
        prefix.reference(path)
        for path in (
            Path(__file__),
            CONFIG,
            PLAN,
            REPO / "src/s2a/research/trajectory_search/staged_first_feasibility.py",
            REPO / "tests/research/trajectory_search/test_staged_first_feasibility.py",
            REPO / "tests/research/native_runtime/test_task2_terminal85_first_search.py",
        )
    ]


def read_config():
    ref = prefix.reference(CONFIG)
    config = prefix.read(ref)
    prefix.require(
        config
        == dict(
            schema="s2a.task2-terminal85-first-search.v1",
            case_index=11,
            stop_step=85,
            seeds=list(SEEDS),
            seed=43,
            control_dt_s=0.05,
            inner_dt_s=0.002,
            joint_buffer_rad=0.05,
            braking_time_s=0.1,
            position_cap_m=0.01,
            rotation_cap_rad=0.05,
            arm_rate_limit=0.25,
            max_responses_per_solve=320,
            max_first_iterations=40,
            max_total_iterations=120,
            max_start_lp_per_solve=2,
            max_models=1,
            max_static_forward_queries=1,
            max_preparation_rollouts=2,
            max_solver_calls=2,
            max_solver_responses=640,
            max_independent_audits=2,
            max_independent_endpoint_queries=400,
            max_independent_transition_steps=5200,
            max_workers=1,
            worker_timeout_seconds=1800,
            physics_steps=0,
            world_model_calls=0,
            fresh_cases=0,
            automatic_retry=False,
        ),
        "Exact terminal stop and first-pose search budget",
    )
    return ref, config


def load_inputs():
    config_ref, config = read_config()
    old = previous.load_inputs()
    previous_receipt = copy.deepcopy(previous.input_receipt(old))
    refs = dict(old["source_sha256"])
    refs[config_ref["path"]] = config_ref["sha256"]
    prior = prefix.read(PRIOR, refs)
    prefix.require(
        prior["complete"] is prior["passed"] is prior["sources_unchanged"] is True
        and prior["control_qualified"] is False
        and prior["physics_steps"] == prior["world_model_calls"] == prior["fresh_cases"] == 0,
        "Closed prior stop-bound analysis",
    )
    registration = prefix.read(prior["registration_ref"], refs)
    prefix.require(
        registration["inputs"] == previous_receipt and registration["source_refs"] == previous.source_refs(),
        "Exact replayed stop-bound sources and inputs",
    )
    results = [prefix.read(ref, refs) for ref in prior["case_refs"]]
    match = [item for item in results if item["case"] == dict(case_index=11, step=85)]
    prefix.require(
        len(match) == 1
        and match[0]["complete"] is match[0]["passed"] is True
        and match[0]["data"]["excludes_first_pose_caps"]["eight_interval_safety"] is False,
        "The terminal stop's existing outer bound is inconclusive",
    )
    case = dict(next(item for item in old["cases"] if item["bound_spec"]["case_index"] == 11))
    prefix.require(case["bound_spec"]["step"] == 85 and len(case["feedback_periods"]) == 5, "Exact terminal stop")
    return dict(
        config_ref=config_ref,
        config=config,
        cases=[case],
        asset_refs=old["asset_refs"],
        previous_receipt=previous_receipt,
        source_sha256=refs,
    )


def input_receipt(inputs):
    return dict(
        previous=inputs["previous_receipt"],
        config_ref=inputs["config_ref"],
        prior_ref=PRIOR,
        case_index=11,
        stop_step=85,
        seeds=list(SEEDS),
        asset_refs=inputs["asset_refs"],
        source_sha256=inputs["source_sha256"],
    )


def counted_solver(stats):
    """Isolate the old solver and count its actual model responses."""
    old = staged.solve_staged_first_feasibility
    namespace = old.__globals__.copy()
    rollout = namespace["rollout_joint_preview"]

    def counted(*args, **kwargs):
        stats["solver_responses_attempted"] += 1
        prefix.require(stats["solver_responses_attempted"] <= 640, "Aggregate model response budget")
        result = rollout(*args, **kwargs)
        stats["solver_responses_completed"] += 1
        return result

    namespace["rollout_joint_preview"] = counted
    function = FunctionType(old.__code__, namespace, old.__name__, old.__defaults__, old.__closure__)
    function.__kwdefaults__ = dict(old.__kwdefaults__ or {})
    return function


def new_stats():
    return dict(
        model_initializations_attempted=0,
        model_initializations_completed=0,
        forward_queries_attempted=0,
        forward_queries_completed=0,
        baseline_rollouts=0,
        max_qpos_restore_difference=0.0,
        additional_preparation_rollouts=0,
        solver_calls_attempted=0,
        solver_calls_completed=0,
        solver_responses_attempted=0,
        solver_responses_completed=0,
        solver_lp_actual=0,
        independent_audits_attempted=0,
        independent_audits_passed=0,
        independent_endpoint_queries_charged=0,
        independent_transition_steps_charged=0,
    )


def analyze(case, inputs, stats):
    context.preload_software_gl(case, inputs["source_sha256"])
    import mujoco

    def prohibited(*_args, **_kwargs):
        raise RuntimeError("Static first-pose search forbids physical integration")

    for name in ("mj_step", "mj_step1", "mj_step2"):
        setattr(mujoco, name, prohibited)
    stats["model_initializations_attempted"] += 1
    model = mujoco.MjModel.from_xml_string(case["record_xml"])
    stats["model_initializations_completed"] += 1
    data = mujoco.MjData(model)
    periods, last = case["feedback_periods"], case["feedback_periods"][-1]
    shifted, warm = case["stop_inputs"]["shifted"], case["stop_inputs"]["lp_start"]
    decision = dict(step=85, decision=dict(refined_plan=shifted))
    current = context.restored_context(model, data, case, periods + [decision], len(periods), stats)
    np.testing.assert_array_equal(current["state"], last["actual_state"])
    np.testing.assert_array_equal(current["previous"], last["command"])
    np.testing.assert_allclose(current["q"], last["actual_inner"][-1]["q"], atol=1e-10, rtol=0)
    np.testing.assert_allclose(current["velocity"], last["actual_inner"][-1]["qvel"], atol=1e-10, rtol=0)
    cursor = case["intervention_spec"]["cursor"] + len(periods)
    targets = np.asarray(case["targets"][cursor : cursor + 8])
    twists = np.asarray(
        [reference_world_twist(case["targets"], i, case["anchor"], 0.05)["twist"] for i in range(cursor, cursor + 8)]
    )
    prefix.require(targets.shape == (8, 4, 4) and twists.shape == (8, 6), "Eight already-generated targets")
    tool = inverse(np.asarray(case["controller"]["source_from_site"])) @ case["tool"]
    first_error = pose_error(targets[0], current["rollout"]["pose"][0] @ tool)
    np.testing.assert_allclose(first_error, case["stop_inputs"]["recorded_shifted_errors"][0], atol=1e-10, rtol=0)
    profile, limits = case["stop_inputs"]["profile"], current["limits"]
    shift_domain = linear_domain(
        current["model"],
        shifted,
        current["previous"][:6],
        current["interface"],
        limits[:, 0],
        limits[:, 1],
        profile,
        current["rollout"],
    )
    warm_rollout = rollout_joint_preview(
        current["model"],
        current["velocity"],
        np.asarray([current["interface"].decode(row) for row in warm]),
        current["site"],
        limits[:, 0],
        limits[:, 1],
        enforce_limits=False,
    )
    stats["additional_preparation_rollouts"] += 1
    warm_domain = linear_domain(
        current["model"],
        warm,
        current["previous"][:6],
        current["interface"],
        limits[:, 0],
        limits[:, 1],
        profile,
        warm_rollout,
        terminal_normals=shift_domain["terminal_normals"],
    )
    differences = comparison.compare_domains(shift_domain, warm_domain)
    checks = {
        SEEDS[0]: domain_check(warm, warm_domain, current["interface"]),
        SEEDS[1]: domain_check(shifted, shift_domain, current["interface"]),
    }
    prefix.require(
        all(value["accepted"] for value in checks.values()), "Both saved seeds satisfy the common safety domain"
    )
    outcomes = []
    for label, seed in zip(SEEDS, (warm, shifted), strict=True):
        args = (
            current["model"],
            current["velocity"],
            seed,
            current["previous"][:6],
            current["interface"],
            current["site"],
            tool,
            targets,
            twists,
            limits[:, 0],
            limits[:, 1],
        )
        before = stats["solver_responses_attempted"]
        stats["solver_calls_attempted"] += 1
        solved = counted_solver(stats)(
            *args, past_inner_error_rad=profile, terminal_normals=shift_domain["terminal_normals"]
        )
        stats["solver_calls_completed"] += 1
        prefix.require(
            stats["solver_responses_attempted"] - before == solved["response_evaluations"] <= 320
            and solved["initialization"]["actual_model_rollouts"] == solved["response_evaluations"]
            and 0 <= solved["initialization"]["start_lp_attempts"] <= 2,
            "Exact per-solve staged search responses and LP budget",
        )
        stats["solver_lp_actual"] += solved["initialization"]["start_lp_attempts"]
        candidate = solved.get("best_joint_safe")
        audit = dict(status="not_attempted")
        if candidate is not None:
            stats["independent_audits_attempted"] += 1
            stats["independent_endpoint_queries_charged"] += 200
            stats["independent_transition_steps_charged"] += 2600
            try:
                verified = audit_full_horizon_witness(
                    candidate,
                    current["mass"],
                    current["jacobian"],
                    current["contract"]["kp"],
                    current["contract"]["kd"],
                    current["q"],
                    current["velocity"],
                    case["initial_joint"],
                    current["interface"],
                    current["previous"][:6],
                    current["site"],
                    tool,
                    targets,
                    limits[:, 0],
                    limits[:, 1],
                    past_inner_error_rad=profile,
                    enforce_pose_caps=False,
                    enforce_braking=True,
                )
                prefix.require(
                    verified["passed"]
                    is verified["first_pose_caps_satisfied"]
                    is verified["braking_constraint_enforced"]
                    is True
                    and verified["independent_endpoint_queries"] == 200
                    and verified["independent_transition_steps"] == 2600,
                    "Independent first-pose and complete safety audit",
                )
                stats["independent_audits_passed"] += 1
                audit = dict(status="passed", verification=verified)
            except (AssertionError, ValueError) as error:
                audit = dict(status="rejected", error=dict(type=type(error).__name__, message=str(error)))
        outcomes.append(
            dict(
                seed=label,
                solver=solved,
                independent_audit=audit,
                model_first_pose_candidate=bool(candidate is not None and audit["status"] == "passed"),
            )
        )
    return dict(
        first_target_cursor=cursor,
        domain_max_differences=differences,
        seed_safety=checks,
        common_terminal_normals=shift_domain["terminal_normals"],
        outcomes=outcomes,
        scope="static first-pose candidate search; no dual-pose or physical feasibility claim",
    )


def require_counts(stats, config):
    prefix.require(
        stats["model_initializations_attempted"]
        == stats["model_initializations_completed"]
        == config["max_models"]
        == 1
        and stats["forward_queries_attempted"] == stats["forward_queries_completed"] == stats["baseline_rollouts"] == 1
        and stats["baseline_rollouts"] + stats["additional_preparation_rollouts"]
        == config["max_preparation_rollouts"]
        == 2
        and stats["solver_calls_attempted"] == stats["solver_calls_completed"] == config["max_solver_calls"] == 2
        and stats["solver_responses_attempted"] == stats["solver_responses_completed"] <= config["max_solver_responses"]
        and 0 <= stats["solver_lp_actual"] <= 4
        and 0
        <= stats["independent_audits_passed"]
        <= stats["independent_audits_attempted"]
        <= config["max_independent_audits"]
        and stats["independent_endpoint_queries_charged"]
        == 200 * stats["independent_audits_attempted"]
        <= config["max_independent_endpoint_queries"]
        and stats["independent_transition_steps_charged"]
        == 2600 * stats["independent_audits_attempted"]
        <= config["max_independent_transition_steps"],
        "Registered terminal first-pose search and audit costs",
    )


def worker(index):
    prefix.environment()
    registration = prefix.read(prefix.reference(OUTPUT / "registration.json"))
    inputs = load_inputs()
    prefix.require(
        input_receipt(inputs) == registration["inputs"] and source_refs() == registration["source_refs"],
        "Frozen terminal stop inputs and source identity",
    )
    case = next(item for item in inputs["cases"] if item["bound_spec"]["case_index"] == index)
    stats = new_stats()
    result = dict(
        complete=False, passed=False, control_qualified=False, case=dict(case_index=index, step=85), stats=stats
    )
    try:
        data = analyze(case, inputs, stats)
        require_counts(stats, inputs["config"])
        result.update(complete=True, passed=True, data=data)
    except BaseException as error:
        result["error"] = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        print(publish(OUTPUT / f"case_{index:02d}" / "result.json", result), flush=True)


def summarize(cases, config):
    prefix.require(len(cases) == 1 and cases[0]["case"] == dict(case_index=11, step=85), "Exact terminal stop")
    require_counts(cases[0]["stats"], config)
    prefix.require([row["seed"] for row in cases[0]["data"]["outcomes"]] == list(SEEDS), "Both registered causal seeds")
    return cases[0]["stats"]


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
            worker_module="scripts.tracking.trajectory_search.audit_task2_terminal85_first_search",
            load_inputs=load_inputs,
            input_receipt=input_receipt,
            source_refs=source_refs,
            summarize=summarize,
        )
    else:
        worker(args.worker)


if __name__ == "__main__":
    main()
