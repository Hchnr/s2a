"""Compare a first-pose solver at a replayed, causally bound stopping state."""

import argparse
import sys
import time
from types import FunctionType

import numpy as np

from s2a.geometry import inverse, pose_error
from s2a.infrastructure.owned_processes import run_owned_workers
from s2a.infrastructure.publication import publish_result as publish
from s2a.references.generated_twist import reference_world_twist
from s2a.research.joint_control.preview.staged_first_feasibility_feedback import StagedFirstFeasibilitySolverAudit
from s2a.research.trajectory_search import staged_first_feasibility
from scripts.maintenance.run_cpu_tests import snapshot
from scripts.simulation.runtime import registered_feedback_phase as lifecycle_check
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.tracking.trajectory_search import audit_early_quality_feasibility as static
from scripts.tracking.trajectory_search import audit_replayed_reference_pose_bounds as bound
from scripts.tracking.trajectory_search import audit_replayed_reference_pose_bounds_v2 as source

ROOT = prefix.REPO
CONFIG = ROOT / "configs/tracking/joint_control/replayed_first_pose_search_20261112.json"
PLAN = ROOT / "plans/v0.0.1/tracking/joint_control/replayed_first_pose_search_20261112.md"
QA = ROOT / "outputs/qa/replayed_first_pose_search_20261112/result.json"
DRY = ROOT / "outputs/replayed_first_pose_search_dry_20261112/result.json"
OUTPUT = ROOT / "outputs/replayed_first_pose_search_20261112"
MODULE = "scripts.tracking.trajectory_search.audit_replayed_first_pose_search"
BOUND = {
    "path": str(ROOT / "outputs/replayed_reference_pose_bounds_v2_20261112/result.json"),
    "sha256": "4f6deb71a81735a3736b5467e5ade7ea08ecb997b751b800447ef6dfc7e2ab7c",
}
FIXED = dict(
    schema="s2a.replayed-first-pose-search.v1",
    case_id="task0_init0_s42_raw",
    seed=42,
    reference_start_step=88,
    decision_step=108,
    solver="staged_first_feasibility",
    max_static_models=1,
    max_forward_queries=1,
    max_reconstruction_rollouts=1,
    max_solver_responses=320,
    max_initializer_lps=2,
    max_first_iterations=40,
    max_total_iterations=120,
    max_audit_endpoint_queries=200,
    max_audit_transition_steps=2600,
    physics_steps=0,
    world_model_calls=0,
    fresh_cases=0,
    automatic_retry=False,
)


def generated_horizon(poses, anchor, reference_start, decision_step):
    """Preserve the generated trajectory clock and eight-point twist preview."""
    targets = bound.reference_window(poses, reference_start, decision_step)
    cursor = decision_step - reference_start
    twists = np.asarray(
        [reference_world_twist(poses, index, anchor, 0.05)["twist"] for index in range(cursor, cursor + 8)]
    )
    return targets, twists


def load_inputs():
    prefix.environment()
    config_ref = prefix.reference(CONFIG)
    config = prefix.read(config_ref)
    prefix.require(
        type(config) is dict
        and set(config) == set(FIXED)
        and all(type(config[key]) is type(value) and config[key] == value for key, value in FIXED.items()),
        "The fixed static search and original budgets are required",
    )
    inputs = source.load_inputs()
    owner = prefix.read(BOUND)
    result = prefix.read(owner["worker_result"])
    prefix.require(
        owner["complete"] is owner["passed"] is owner["sources_unchanged"] is True
        and result["complete"] is result["passed"] is True
        and inputs["producer"]["case"]["case_id"] == config["case_id"]
        and len(result["outcomes"]) == 4
        and result["outcomes"][-1]["decision_step"] == 108,
        "The completed bound audit and matching development state are required",
    )
    inputs["refs"].update(
        search_config=config_ref, search_plan=prefix.reference(PLAN), bound=BOUND, bound_worker=owner["worker_result"]
    )
    inputs["search_config"] = config
    inputs["bound_outcome"] = result["outcomes"][-1]
    return inputs


def receipt(inputs):
    return dict(inputs=bound.input_receipt(inputs), decision_step=inputs["search_config"]["decision_step"])


def audit(inputs, stats):
    config, case = inputs["search_config"], inputs["case"]
    static.preload_software_gl(case, inputs["source"]["source_sha256"])
    import mujoco

    def prohibit_step(*_args, **_kwargs):
        raise RuntimeError("This static search cannot advance physics")

    for name in ("mj_step", "mj_step1", "mj_step2"):
        setattr(mujoco, name, prohibit_step)
    stats["static_models_attempted"] += 1
    model = mujoco.MjModel.from_xml_string(case["record_xml"])
    stats["static_models_completed"] += 1
    data = mujoco.MjData(model)
    step = config["decision_step"]
    prior = inputs["periods"][step - 1]
    seed, profile = bound.completed_plan_input(prior, step)
    static_case = dict(
        case,
        spec={"prefix_steps": step - 1},
        saved={"sim_states": np.asarray([prior["actual_state"]])},
        commands=np.asarray([prior["command"]]),
        initial_joint=np.asarray(inputs["producer"]["runtime"]["runtime_initial_joint"]),
    )
    periods = [prior, {"step": step, "decision": {"refined_plan": seed}}]
    current = static.restored_context(model, data, static_case, periods, 1, stats)
    np.testing.assert_array_equal(current["state"], prior["actual_state"])
    np.testing.assert_array_equal(current["previous"], prior["command"])
    tool = inverse(np.asarray(case["controller"]["source_from_site"])) @ case["tool"]
    anchor = np.asarray(inputs["source"]["capture88"]["current_source_pose"]) @ case["tool"]
    targets, twists = generated_horizon(inputs["poses"], anchor, config["reference_start_step"], step)
    seed_error = pose_error(targets[0], current["rollout"]["pose"][0] @ tool)
    np.testing.assert_allclose(seed_error, inputs["bound_outcome"]["seed_first_tcp_error"], atol=1e-10, rtol=0)

    # Isolate the imported solver's namespace so actual rollout attempts are
    # counted without modifying the historical solver or its global module.
    old = staged_first_feasibility.solve_staged_first_feasibility
    namespace = old.__globals__.copy()
    original_rollout = namespace["rollout_joint_preview"]

    def counted_rollout(*args, **kwargs):
        stats["solver_responses_attempted"] += 1
        prefix.require(stats["solver_responses_attempted"] <= config["max_solver_responses"], "Solver response budget")
        value = original_rollout(*args, **kwargs)
        stats["solver_responses_completed"] += 1
        return value

    namespace["rollout_joint_preview"] = counted_rollout
    solve = FunctionType(old.__code__, namespace, old.__name__, old.__defaults__, old.__closure__)
    solve.__kwdefaults__ = dict(old.__kwdefaults__ or {})
    trace = {}
    adapter = StagedFirstFeasibilitySolverAudit(trace, solver=solve)
    interface, limits = current["interface"], current["limits"]
    stats["solver_calls_attempted"] += 1
    start = time.monotonic()
    witness = adapter.solve(
        current["model"],
        current["velocity"],
        seed,
        current["previous"][:6],
        interface,
        current["site"],
        tool,
        targets,
        twists,
        limits[:, 0],
        limits[:, 1],
        past_inner_error_rad=profile,
    )
    elapsed = time.monotonic() - start
    stats["solver_calls_completed"] += 1
    prefix.require(
        witness["response_evaluations"] == stats["solver_responses_attempted"] == stats["solver_responses_completed"]
        and trace["feasible_start_traces"][0]["start_lp_attempts"] <= config["max_initializer_lps"],
        "Every solver response and initializer LP must be counted",
    )
    verification = None
    if witness["witness_found"] is True:
        stats["audit_endpoint_queries_charged"] += 200
        stats["audit_transition_steps_charged"] += 2600
        contract = current["contract"]
        verification = adapter.audit(
            witness,
            current["mass"],
            current["jacobian"],
            contract["kp"],
            contract["kd"],
            current["q"],
            current["velocity"],
            static_case["initial_joint"],
            interface,
            current["previous"][:6],
            current["site"],
            tool,
            targets[0],
            limits[:, 0],
            limits[:, 1],
            past_inner_error_rad=profile,
        )
        prefix.require(
            verification["passed"] is verification["first_pose_caps_satisfied"] is True
            and verification["independent_endpoint_queries"] == 200
            and verification["independent_transition_steps"] == 2600,
            "The same first-pose and joint-safety audit must independently pass",
        )
        stats["audit_endpoint_queries_completed"] += verification["independent_endpoint_queries"]
        stats["audit_transition_steps_completed"] += verification["independent_transition_steps"]
    return dict(
        decision_step=step,
        seed_first_tcp_error=seed_error,
        witness=witness,
        verification=verification,
        solver_trace=trace,
        solver_elapsed_seconds=elapsed,
    )


def dry_run():
    inputs = load_inputs()
    print(
        publish(
            DRY,
            dict(
                complete=True,
                passed=True,
                inputs=receipt(inputs),
                sources_sha256=bound.sources_digest(snapshot(ROOT)["sources"]),
                static_models=0,
                physics_steps=0,
                world_model_calls=0,
                automatic_retry=False,
            ),
        ),
        flush=True,
    )


def worker():
    registration = prefix.read(prefix.reference(OUTPUT / "registration.json"))
    inputs = load_inputs()
    prefix.require(receipt(inputs) == registration["inputs"], "Registered source identity must match")
    stats = dict(
        static_models_attempted=0,
        static_models_completed=0,
        forward_queries_attempted=0,
        forward_queries_completed=0,
        baseline_rollouts=0,
        max_qpos_restore_difference=0.0,
        solver_calls_attempted=0,
        solver_calls_completed=0,
        solver_responses_attempted=0,
        solver_responses_completed=0,
        audit_endpoint_queries_charged=0,
        audit_endpoint_queries_completed=0,
        audit_transition_steps_charged=0,
        audit_transition_steps_completed=0,
    )
    result = dict(complete=False, passed=False, control_qualified=False, stats=stats)
    try:
        outcome = audit(inputs, stats)
        prefix.require(
            stats["static_models_attempted"] == stats["static_models_completed"] == 1
            and stats["forward_queries_attempted"] == stats["forward_queries_completed"] == 1
            and stats["baseline_rollouts"] == stats["solver_calls_attempted"] == stats["solver_calls_completed"] == 1
            and stats["solver_responses_attempted"] == stats["solver_responses_completed"] <= 320
            and stats["audit_endpoint_queries_charged"] == stats["audit_endpoint_queries_completed"] <= 200
            and stats["audit_transition_steps_charged"] == stats["audit_transition_steps_completed"] <= 2600,
            "Actual static costs must match the registered limits",
        )
        result.update(complete=True, passed=True, outcome=outcome)
    except BaseException as error:
        result["error"] = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        print(publish(OUTPUT / "worker/result.json", result), flush=True)


def execute():
    prefix.environment()
    qa, dry = prefix.read(prefix.reference(QA)), prefix.read(prefix.reference(DRY))
    prefix.require(
        qa["complete"] is qa["passed"] is qa["all_tests_passed"] is qa["source_unchanged"] is True
        and qa["actual_wait"] is qa["reaped"] is True
        and qa["signal_sent"] is False
        and qa["configured_parallelism"] == 32
        and qa["skipped_count"] == qa["xfail_count"] == qa["xpass_count"] == 0
        and dry["complete"] is dry["passed"] is True,
        "A closed parallel regression and zero-model dry are required",
    )
    before = snapshot(ROOT)["sources"]
    prefix.require(
        before == prefix.read(qa["source_before_ref"])["sources"]
        and bound.sources_digest(before) == dry["sources_sha256"],
        "Frozen source identity",
    )
    inputs = load_inputs()
    prefix.require(receipt(inputs) == dry["inputs"], "The dry source identity must match")
    OUTPUT.mkdir(parents=True, exist_ok=False)
    registration = publish(
        OUTPUT / "registration.json", dict(inputs=receipt(inputs), qa=prefix.reference(QA), dry=prefix.reference(DRY))
    )
    lifecycle = run_owned_workers(
        [dict(key="search", argv=[sys.executable, "-m", MODULE, "--worker"], log=OUTPUT / "worker/worker.log")],
        cwd=ROOT,
        timeout_seconds=600,
    )
    for item in lifecycle["workers"]:
        item["log_ref"] = prefix.reference(item.pop("log"))
    workers = publish(OUTPUT / "workers.json", lifecycle)
    summary = dict(
        complete=False,
        passed=False,
        control_qualified=False,
        registration=registration,
        workers=workers,
        physics_steps=0,
        world_model_calls=0,
        fresh_cases=0,
        automatic_retry=False,
    )
    try:
        lifecycle_check.require_closed_workers(lifecycle, ["search"])
        ref = prefix.reference(OUTPUT / "worker/result.json")
        result = prefix.read(ref)
        prefix.require(result["complete"] is result["passed"] is True, "The static search must complete")
        prefix.require(snapshot(ROOT)["sources"] == before, "Sources must remain unchanged")
        summary.update(complete=True, passed=True, worker_result=ref, sources_unchanged=True, stats=result["stats"])
    finally:
        print(publish(OUTPUT / "result.json", summary), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true")
    group.add_argument("--execute", action="store_true")
    group.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        dry_run()
    elif args.execute:
        execute()
    else:
        worker()


if __name__ == "__main__":
    main()
