"""Compare guarded two-point search from two causal seeds at fixed stops."""

import argparse
import copy
from pathlib import Path

import numpy as np

from s2a.geometry import finite, inverse, pose_error
from s2a.infrastructure.publication import publish_result as publish
from s2a.references.generated_twist import reference_world_twist
from s2a.research.joint_control.preview.interior_joint_preview import rollout_joint_preview
from s2a.research.joint_control.tracking.plan_model_drift import shifted_completed_plan
from s2a.research.joint_control.tracking.two_step_quality_audit import audit_two_step_witness
from s2a.research.trajectory_search.feasible_start_restoration import domain_check, linear_domain
from s2a.research.trajectory_search.guarded_two_step_search import solve_guarded_two_step
from scripts.simulation.runtime import prepare_task2_guarded_model_drift as drift_binding
from scripts.simulation.runtime import registered_static_phase
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.tracking.trajectory_search import audit_early_quality_feasibility as context
from scripts.tracking.trajectory_search import audit_task2_shifted_seed as qualification

REPO = prefix.REPO
CONFIG = REPO / "configs/tracking/joint_control/task2_shifted_seed_search_20261111.json"
PLAN = REPO / "plans/v0.0.1/tracking/joint_control/task2_shifted_seed_search_20261111.md"
QA = REPO / "outputs/qa/task2_shifted_seed_search_20261111/result.json"
DRY = REPO / "outputs/task2_shifted_seed_search_dry_20261111/result.json"
OUTPUT = REPO / "outputs/task2_shifted_seed_search_20261111"
PRIOR = {
    "path": str(REPO / "outputs/task2_shifted_seed_qualification_20261111/result.json"),
    "sha256": "f5af0996b2efced760615c19dcbe143e26b1a2b18a020ff4fea7731e3fcd1279",
}
CASES = qualification.CASES
SEEDS = ("recorded_lp_start", "shifted_completed_plan")
INDICES = (10, 11)


def source_refs():
    return qualification.source_refs() + [
        prefix.reference(path)
        for path in (
            Path(__file__),
            CONFIG,
            PLAN,
            REPO / "tests/research/native_runtime/test_task2_shifted_seed_search.py",
        )
    ]


def read_config():
    ref = prefix.reference(CONFIG)
    value = prefix.read(ref)
    prefix.require(
        value
        == dict(
            schema="s2a.task2-shifted-seed-search.v1",
            cases=CASES,
            seeds=list(SEEDS),
            seed=43,
            control_dt_s=0.05,
            inner_dt_s=0.002,
            joint_buffer_rad=0.05,
            braking_time_s=0.1,
            position_cap_m=0.01,
            rotation_cap_rad=0.05,
            arm_rate_limit=0.25,
            hard_pose_prefix=2,
            max_responses_per_solve=320,
            max_feasibility_iterations=60,
            max_total_iterations=120,
            max_start_lp_per_solve=2,
            max_models=2,
            max_static_forward_queries=2,
            max_preparation_rollouts=4,
            max_solver_calls=4,
            max_solver_responses=1280,
            max_independent_audits=4,
            max_independent_endpoint_queries=800,
            max_independent_transition_steps=10400,
            max_workers=2,
            worker_timeout_seconds=1800,
            physics_steps=0,
            world_model_calls=0,
            fresh_cases=0,
            automatic_retry=False,
        ),
        "Exact stopped-state two-seed solver comparison and budgets",
    )
    return ref, value


def stopped_lp_start(branch, spec):
    """Keep the logged post-LP initial plan, not an invented baseline seed."""
    runtime = branch["runtime"]
    stop = runtime["continuation"]["stop"]
    trace, quality = runtime["feasible_start_traces"][-1], runtime["two_step_feedback_traces"][-1]
    prefix.require(
        branch["complete"] is branch["passed"] is True
        and branch["phase"] == "producer"
        and branch["branch"] == "guarded_two_step"
        and branch["case"]["case_index"] == spec["case_index"]
        and stop["at_step"] == spec["stop_step"]
        and stop["reason"] == "quality_no_witness"
        and stop["next_action_dispatched"] is False
        and trace["seed_linear_safe"] is False
        and trace["warm_start_used"] is True
        and trace["check"]["accepted"] is True
        and quality["status"] == "two_step_no_witness"
        and quality["closest_prefix_rejected"] is not None,
        "Saved LP start and prior stopped finite-search evidence",
    )
    return dict(
        commands=finite(trace["warm_start_commands"], (8, 6)).copy(),
        prior_response_evaluations=quality["response_evaluations"],
        prior_closest_prefix_rejected=copy.deepcopy(quality["closest_prefix_rejected"]),
        prior_phases=copy.deepcopy(quality["phases"]),
    )


def load_inputs():
    config_ref, config = read_config()
    old = qualification.load_inputs()
    previous_receipt = copy.deepcopy(qualification.input_receipt(old))
    refs = dict(old["source_sha256"])
    refs[config_ref["path"]] = config_ref["sha256"]
    prior = prefix.read(PRIOR, refs)
    prefix.require(
        prior["complete"] is prior["passed"] is prior["sources_unchanged"] is True
        and prior["control_qualified"] is False
        and prior["physics_steps"] == prior["world_model_calls"] == prior["fresh_cases"] == 0,
        "Closed prior shifted-plan qualification",
    )
    registration = prefix.read(prior["registration_ref"], refs)
    prefix.require(
        registration["inputs"] == previous_receipt and registration["source_refs"] == qualification.source_refs(),
        "Prior qualification input and source identity",
    )
    qualified = {}
    for ref in prior["case_refs"]:
        result = prefix.read(ref, refs)
        index = result["case"]["case_index"]
        prefix.require(
            result["complete"] is result["passed"] is True and index not in qualified, "Unique closed seed case"
        )
        qualified[index] = result
    producer = prefix.read(drift_binding.PRODUCER, refs)
    branches = {}
    for ref in producer["branch_refs"]:
        branch = prefix.read(ref, refs)
        index = branch["case"]["case_index"]
        prefix.require(index not in branches, "Unique prior producer branch")
        branches[index] = branch
    prefix.require(sorted(qualified) == sorted(branches) == list(INDICES), "Both guarded stops")
    cases = []
    for spec in CASES:
        index = spec["case_index"]
        case = dict(next(item for item in old["cases"] if item["stop_spec"]["case_index"] == index))
        item = qualified[index]
        prefix.require(
            item["case"] == spec
            and item["data"]["independent_audit"]["status"] == "passed"
            and item["data"]["linear_safety"]["accepted"] is True
            and item["data"]["quality"]["model_prefix_candidate"] is False,
            "Prior shifted seed has independent safety and no two-point candidate",
        )
        shifted = shifted_completed_plan(case["feedback_periods"][-1])
        np.testing.assert_array_equal(shifted, item["data"]["shifted_commands"])
        case.update(saved_lp=stopped_lp_start(branches[index], spec), shifted_qualification=item["data"]["quality"])
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
        seeds=list(SEEDS),
        asset_refs=inputs["asset_refs"],
        source_sha256=inputs["source_sha256"],
    )


def compare_domains(first, second):
    """Check both seed expansions describe the same affine safety rows."""
    keys = (
        "lo",
        "hi",
        "rate",
        "center",
        "joint_a",
        "joint_lo",
        "joint_hi",
        "terminal_a",
        "terminal_offset",
        "terminal_normals",
    )
    differences = {}
    for key in keys:
        a, b = np.asarray(first[key]), np.asarray(second[key])
        if a.shape != b.shape:
            raise ValueError("Both seeds must share the same affine safety domain")
        differences[key] = float(np.max(abs(a - b))) if a.size else 0.0
        if not np.allclose(a, b, atol=1e-10, rtol=0):
            raise ValueError("Both seeds must share the same affine safety domain")
    return differences


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
        raise RuntimeError("Static seed search forbids physical integration")

    for name in ("mj_step", "mj_step1", "mj_step2"):
        setattr(mujoco, name, prohibited)
    stats["model_initializations_attempted"] += 1
    model = mujoco.MjModel.from_xml_string(case["record_xml"])
    stats["model_initializations_completed"] += 1
    data = mujoco.MjData(model)
    periods, last = case["feedback_periods"], case["feedback_periods"][-1]
    shifted = shifted_completed_plan(last)
    decision = dict(step=case["stop_spec"]["stop_step"], decision=dict(refined_plan=shifted))
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
    prefix.require(targets.shape == (8, 4, 4) and twists.shape == (8, 6), "Eight generated targets and twists")
    tool = inverse(np.asarray(case["controller"]["source_from_site"])) @ case["tool"]
    first_error = pose_error(targets[0], current["rollout"]["pose"][0] @ tool)
    np.testing.assert_allclose(
        first_error, case["prior_transition"]["next_forecast"]["dynamics_refreshed"], atol=1e-10, rtol=0
    )
    profile = np.asarray(last["completed_feedback"]["past_error_after_completion"]["inner_position_error_rad"])
    bounds = current["limits"]
    shift_domain = linear_domain(
        current["model"],
        shifted,
        current["previous"][:6],
        current["interface"],
        bounds[:, 0],
        bounds[:, 1],
        profile,
        current["rollout"],
    )
    warm = case["saved_lp"]["commands"]
    warm_rollout = rollout_joint_preview(
        current["model"],
        current["velocity"],
        np.asarray([current["interface"].decode(row) for row in warm]),
        current["site"],
        bounds[:, 0],
        bounds[:, 1],
        enforce_limits=False,
    )
    stats["additional_preparation_rollouts"] += 1
    warm_domain = linear_domain(
        current["model"],
        warm,
        current["previous"][:6],
        current["interface"],
        bounds[:, 0],
        bounds[:, 1],
        profile,
        warm_rollout,
        terminal_normals=shift_domain["terminal_normals"],
    )
    differences = compare_domains(shift_domain, warm_domain)
    checks = {
        SEEDS[0]: domain_check(warm, warm_domain, current["interface"]),
        SEEDS[1]: domain_check(shifted, shift_domain, current["interface"]),
    }
    prefix.require(
        all(value["accepted"] for value in checks.values()),
        "Both seeds must satisfy the common reconstructed safety domain",
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
            bounds[:, 0],
            bounds[:, 1],
        )
        before = stats["solver_responses_attempted"]

        def response(event):
            stats[f"solver_responses_{event}"] += 1
            prefix.require(stats["solver_responses_attempted"] - before <= 320, "Per-solve response budget")

        stats["solver_calls_attempted"] += 1
        solved = solve_guarded_two_step(
            *args, past_inner_error_rad=profile, terminal_normals=shift_domain["terminal_normals"], on_response=response
        )
        stats["solver_calls_completed"] += 1
        prefix.require(
            stats["solver_responses_attempted"] - before == solved["response_evaluations"] <= 320,
            "Actual bounded solver responses",
        )
        stats["solver_lp_actual"] += solved["initialization"]["start_lp_attempts"]
        candidate = solved.get("best_joint_safe")
        audit = dict(status="not_attempted")
        if candidate is not None:
            stats["independent_audits_attempted"] += 1
            stats["independent_endpoint_queries_charged"] += 200
            stats["independent_transition_steps_charged"] += 2600
            try:
                verified = audit_two_step_witness(
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
                    bounds[:, 0],
                    bounds[:, 1],
                    past_inner_error_rad=profile,
                )
                prefix.require(
                    verified["passed"] is verified["prefix_pose_caps_satisfied"] is True
                    and verified["independent_endpoint_queries"] == 200
                    and verified["independent_transition_steps"] == 2600,
                    "Independent two-point audit",
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
                model_prefix_candidate=bool(candidate is not None and audit["status"] == "passed"),
            )
        )
    return dict(
        first_target_cursor=cursor,
        domain_max_differences=differences,
        seed_safety=checks,
        common_terminal_normals=shift_domain["terminal_normals"],
        prior_stop_search=case["saved_lp"],
        outcomes=outcomes,
        scope="static reconstructed seed comparison; not an exact replay of the original stop solver or physical feedback",
    )


def require_case_counts(stats):
    prefix.require(
        stats["model_initializations_attempted"] == stats["model_initializations_completed"] == 1
        and stats["forward_queries_attempted"] == stats["forward_queries_completed"] == stats["baseline_rollouts"] == 1
        and stats["additional_preparation_rollouts"] == 1
        and stats["solver_calls_attempted"] == stats["solver_calls_completed"] == 2
        and stats["solver_responses_attempted"] == stats["solver_responses_completed"] <= 640
        and 0 <= stats["solver_lp_actual"] <= 4
        and 0 <= stats["independent_audits_passed"] <= stats["independent_audits_attempted"] <= 2
        and stats["independent_endpoint_queries_charged"] == 200 * stats["independent_audits_attempted"]
        and stats["independent_transition_steps_charged"] == 2600 * stats["independent_audits_attempted"],
        "Bounded per-case seed search and audit costs",
    )


def worker(index):
    prefix.environment()
    registration = prefix.read(prefix.reference(OUTPUT / "registration.json"))
    inputs = load_inputs()
    prefix.require(
        input_receipt(inputs) == registration["inputs"] and source_refs() == registration["source_refs"],
        "Frozen stop inputs and source identity",
    )
    case = next(item for item in inputs["cases"] if item["stop_spec"]["case_index"] == index)
    stats = new_stats()
    result = dict(complete=False, passed=False, control_qualified=False, case=case["stop_spec"], stats=stats)
    try:
        data = analyze(case, inputs, stats)
        require_case_counts(stats)
        result.update(complete=True, passed=True, data=data)
    except BaseException as error:
        result["error"] = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        print(publish(OUTPUT / f"case_{index:02d}" / "result.json", result), flush=True)


def summarize(cases, config):
    prefix.require([item["case"] for item in cases] == CASES, "Exact registered stops")
    for item in cases:
        require_case_counts(item["stats"])
        prefix.require(
            [row["seed"] for row in item["data"]["outcomes"]] == list(SEEDS), "Both registered seed searches"
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
        "Registered aggregate seed-search and audit budgets",
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
            worker_module="scripts.tracking.trajectory_search.compare_task2_shifted_seed_search",
            load_inputs=load_inputs,
            input_receipt=input_receipt,
            source_refs=source_refs,
            summarize=summarize,
        )
    else:
        worker(args.worker)


if __name__ == "__main__":
    main()
