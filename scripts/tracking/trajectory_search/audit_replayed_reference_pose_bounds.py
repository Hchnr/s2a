"""Audit first-pose bounds using completed, independently replayed feedback."""

import argparse
import hashlib
import json
import sys

import numpy as np

from s2a.geometry import inverse, pose_error
from s2a.infrastructure.owned_processes import run_owned_workers
from s2a.infrastructure.publication import publish_result as publish
from s2a.research.joint_control.tracking.stopped_pose_safety_bound import (
    error_directions,
    first_pose_linearization,
    safety_scopes,
    weighted_pose_bounds,
)
from s2a.research.trajectory_search.feasible_start_restoration import domain_check, linear_domain
from scripts.maintenance.run_cpu_tests import snapshot
from scripts.simulation.runtime import audit_reset_multitask_live_recovery as comparison
from scripts.simulation.runtime import registered_feedback_phase as lifecycle_check
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import reset_bowl_next_reference as reference
from scripts.simulation.runtime import reset_bowl_reference_tail as tail
from scripts.simulation.runtime import reset_live_two_window_replay as capture_check
from scripts.simulation.runtime import run_reset_bowl_next_reference as source
from scripts.tracking.trajectory_search import audit_early_quality_feasibility as static
from scripts.verification.reference_ports.check_stopped_pose_safety_bound import check_candidate

ROOT = prefix.REPO
CONFIG = ROOT / "configs/tracking/joint_control/replayed_reference_pose_bounds_20261112.json"
PLAN = ROOT / "plans/v0.0.1/tracking/joint_control/replayed_reference_pose_bounds_20261112.md"
QA = ROOT / "outputs/qa/replayed_reference_pose_bounds_v2_20261112/result.json"
DRY = ROOT / "outputs/replayed_reference_pose_bounds_dry_20261112/result.json"
OUTPUT = ROOT / "outputs/replayed_reference_pose_bounds_20261112"
MODULE = "scripts.tracking.trajectory_search.audit_replayed_reference_pose_bounds"
PRODUCER = {
    "path": str(ROOT / "outputs/reset_bowl_step88_reference_20261112/producer/result.json"),
    "sha256": "2f52da105d77fed05b45984fea791b9bf29922e634693d5a218cc46994d9b4e1",
}
CHECKER = {
    "path": str(ROOT / "outputs/reset_bowl_step88_reference_20261112/checker/result.json"),
    "sha256": "e3b82504ae60627f0bd29107661130d9635cae5f4ea373a5d13067d404cfca8c",
}
FIXED = dict(
    schema="s2a.replayed-reference-pose-bounds.v1",
    case_id="task0_init0_s42_raw",
    seed=42,
    reference_start_step=88,
    decision_steps=[104, 106, 107, 108],
    position_weights=[0.0, 0.25, 0.5, 0.75, 1.0],
    normalized_numerical_allowance=1e-7,
    max_static_models=1,
    max_forward_queries=4,
    max_seed_rollouts=4,
    max_support_lp_calls=40,
    max_independent_first_responses=4,
    max_independent_transition_steps=100,
    physics_steps=0,
    world_model_calls=0,
    fresh_cases=0,
    automatic_retry=False,
)


def completed_plan_input(period, decision_step):
    """Bind a shifted plan only to the immediately preceding completed period."""
    if type(decision_step) is not int or decision_step < 1 or period.get("step") != decision_step - 1:
        raise ValueError("The immediately preceding completed period is required")
    previous = np.asarray(period["command"], dtype=float)
    plan = np.asarray(period["decision"]["refined_plan"], dtype=float)
    history = period["completed_feedback"]["policy_history"]["past_inner_error"]
    profile = np.asarray(history["inner_position_error_rad"], dtype=float)
    if (
        previous.shape != (7,)
        or plan.shape != (8, 6)
        or profile.shape != (25, 7)
        or not all(np.isfinite(value).all() for value in (previous, plan, profile))
        or not np.array_equal(previous[:6], plan[0])
        or np.any(np.abs(plan) > 1 + 1e-12)
        or np.any(profile < 0)
        or np.any(np.diff(profile, axis=0) < 0)
        or history.get("available") is not True
        or history.get("future_guarantee") is not False
        or history.get("last_completed_step") != decision_step - 1
        or history.get("total_completed_periods") != decision_step
    ):
        raise ValueError("The saved plan, previous command and causal inner history must agree")
    shifted = np.vstack((plan[1:], plan[-1]))
    if np.max(np.abs(np.diff(np.vstack((previous[:6], shifted)), axis=0))) > 0.25 + 1e-12:
        raise ValueError("The completed shifted plan must retain the original command rate")
    return shifted, profile.copy()


def reference_window(poses, reference_start, decision_step):
    """Select only eight points from the reference already generated at start."""
    poses = np.asarray(poses, dtype=float)
    if type(reference_start) is not int or type(decision_step) is not int or poses.shape != (40, 4, 4):
        raise ValueError("A generated H40 and integer reference clock are required")
    cursor = decision_step - reference_start
    if not 0 <= cursor <= 32 or not np.isfinite(poses).all():
        raise ValueError("The requested eight points must lie within the generated reference")
    return poses[cursor : cursor + 8].copy()


def sources_digest(sources):
    return hashlib.sha256(json.dumps(sources, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_inputs():
    prefix.environment()
    config_ref = prefix.reference(CONFIG)
    config = prefix.read(config_ref)
    prefix.require(
        type(config) is dict
        and set(config) == set(FIXED)
        and all(type(config[k]) is type(v) and config[k] == v for k, v in FIXED.items()),
        "The fixed retrospectively selected cases and diagnostic budgets are required",
    )
    original = source.load_inputs()
    owners = [prefix.read(ref) for ref in (PRODUCER, CHECKER)]
    branches = []
    for phase, owner in zip(("producer", "checker"), owners, strict=True):
        prefix.require(
            owner["complete"] is owner["passed"] is owner["sources_unchanged"] is True
            and owner["control_qualified"] is False
            and owner["phase"] == phase
            and owner["counts"]["control_steps"] == 108
            and len(owner["branch_refs"]) == 1,
            "The independently closed 108-period branches are required",
        )
        branches.append(prefix.read(owner["branch_refs"][0]))
    producer, checker = branches
    source.summarize_branch(producer, "producer", original["config"])
    source.summarize_branch(checker, "checker", original["config"], producer=producer)
    p, c = (branch["runtime"]["continuation"] for branch in branches)
    produced = reference.prefix_periods(p["prefix88"]) + tail.tail_periods(p["new_reference"])
    replayed = reference.prefix_periods(c["prefix88"], replay=True) + tail.tail_periods(c["new_reference"])
    differences = comparison.compare_periods(produced, replayed)
    prefix.require(
        len(produced) == len(replayed) == 108
        and all(value == 0 for value in differences.values())
        and producer["case"]["case_id"] == checker["case"]["case_id"] == config["case_id"]
        and p["new_reference"]["stop"]["at_step"] == 108
        and p["new_reference"]["stop"]["next_action_dispatched"] is False
        and c["new_reference"]["stop"] == p["new_reference"]["stop"],
        "The same completed producer and checker periods and recorded stop are required",
    )
    case = original["previous"]["base"]["case"]
    capture = original["capture88"]
    for branch in branches:
        capture_check.compare_capture(capture, branch["runtime"]["step88_reference_capture"]["capture"])
    endpoint = capture["current_endpoint"]
    snapshot88 = dict(step=88, sim_time_s=endpoint["sim_time"], state_sha256=endpoint["sim_state_sha256"])
    poses, _openings = reference.decode_reference(
        snapshot88,
        capture,
        {"source_pose": capture["current_source_pose"]},
        case["tool"],
        lambda: (original["reference88"], original["envelope88"]),
    )
    for step in config["decision_steps"]:
        completed_plan_input(produced[step - 1], step)
        reference_window(poses, config["reference_start_step"], step)
    refs = dict(
        config=config_ref,
        plan=prefix.reference(PLAN),
        producer=PRODUCER,
        checker=CHECKER,
        producer_branch=owners[0]["branch_refs"][0],
        checker_branch=owners[1]["branch_refs"][0],
        reference=original["refs"]["response"],
        envelope=original["refs"]["envelope"],
    )
    return dict(config=config, case=case, source=original, producer=producer, periods=produced, poses=poses, refs=refs)


def input_receipt(inputs):
    return dict(
        refs=inputs["refs"], steps=inputs["config"]["decision_steps"], source_sha256=inputs["source"]["source_sha256"]
    )


def analyze(inputs, stats):
    case, config = inputs["case"], inputs["config"]
    static.preload_software_gl(case, inputs["source"]["source_sha256"])
    import mujoco

    def prohibit_step(*_args, **_kwargs):
        raise RuntimeError("This static diagnostic cannot advance physics")

    for name in ("mj_step", "mj_step1", "mj_step2"):
        setattr(mujoco, name, prohibit_step)
    stats["static_models_attempted"] += 1
    model = mujoco.MjModel.from_xml_string(case["record_xml"])
    stats["static_models_completed"] += 1
    data = mujoco.MjData(model)
    tool = inverse(np.asarray(case["controller"]["source_from_site"])) @ case["tool"]
    initial_joint = inputs["producer"]["runtime"]["runtime_initial_joint"]
    outcomes = []
    for step in config["decision_steps"]:
        prior = inputs["periods"][step - 1]
        seed, profile = completed_plan_input(prior, step)
        targets = reference_window(inputs["poses"], config["reference_start_step"], step)
        static_case = dict(
            case,
            spec={"prefix_steps": step - 1},
            saved={"sim_states": np.asarray([prior["actual_state"]])},
            commands=np.asarray([prior["command"]]),
            initial_joint=np.asarray(initial_joint),
        )
        periods = [prior, {"step": step, "decision": {"refined_plan": seed}}]
        current = static.restored_context(model, data, static_case, periods, 1, stats)
        np.testing.assert_array_equal(current["state"], prior["actual_state"])
        np.testing.assert_array_equal(current["previous"], prior["command"])
        rollout, interface = current["rollout"], current["interface"]
        seed_error = pose_error(targets[0], rollout["pose"][0] @ tool)
        domain = linear_domain(
            current["model"],
            seed,
            current["previous"][:6],
            interface,
            current["limits"][:, 0],
            current["limits"][:, 1],
            profile,
            rollout,
            terminal_normals=np.empty((0, 7)),
        )
        scopes = safety_scopes(domain, seed)
        scale = (interface.output_max - interface.output_min) / (interface.input_max - interface.input_min)
        seed_q = rollout["joint_position"][0]
        command_jacobian = rollout["joint_position_jacobian"][0, :, :6] * scale
        linearization = first_pose_linearization(
            current["site"],
            tool,
            current["jacobian"],
            seed_q - current["q"],
            command_jacobian,
            seed[0],
            scopes["box"]["lower"],
            scopes["box"]["upper"],
            targets[0],
        )
        seed_tcp = rollout["pose"][0] @ tool
        np.testing.assert_allclose(linearization["seed_tcp_position"], seed_tcp[:3, 3], atol=1e-10, rtol=0)
        np.testing.assert_allclose(linearization["seed_tcp_rotation"], seed_tcp[:3, :3], atol=1e-10, rtol=0)
        position_direction, rotation_direction = error_directions(targets[0, :3, :3], seed_error)

        def account(name, event):
            if name != "box":
                stats[f"support_lp_{event}"] += 1
                prefix.require(stats["support_lp_attempted"] <= config["max_support_lp_calls"], "LP budget")

        bounds = weighted_pose_bounds(
            linearization,
            position_direction,
            rotation_direction,
            scopes,
            config["position_weights"],
            allowance=config["normalized_numerical_allowance"],
            on_support=account,
        )
        stats["independent_first_responses_charged"] += 1
        stats["independent_transition_steps_charged"] += 25
        verified = check_candidate(
            current["model"],
            current["velocity"],
            interface,
            current["site"],
            tool,
            targets[0],
            current["limits"][:, 0],
            current["limits"][:, 1],
            seed_q,
            command_jacobian,
            seed,
            seed_error,
            linearization,
            scopes,
            bounds,
        )
        stats["independent_first_responses_completed"] += verified["independent_first_responses"]
        stats["independent_transition_steps_completed"] += verified["independent_transition_steps"]
        safety = domain_check(seed, domain, interface)
        if step == 108:
            recorded = inputs["producer"]["runtime"]["segment_receipts"][0]["causal_shift_fallback_traces"][108]
            np.testing.assert_allclose(seed_error, recorded["fallback_first_two_tcp_errors"][0], atol=1e-9, rtol=0)
            prefix.require(safety == recorded["fallback_safety"], "Saved fallback safety must reconstruct exactly")
        outcomes.append(
            dict(
                decision_step=step,
                previous_state_sha256=prior["endpoint_receipt"]["sim_state_sha256"],
                seed_first_tcp_error=seed_error,
                shifted_plan_safety=safety,
                linearization=linearization,
                bounds=bounds,
                independent_verification=verified,
                excludes_first_pose_caps={
                    name: value["best_lower_bound"] > 1 for name, value in bounds["scopes"].items()
                },
            )
        )
    return outcomes


def dry_run():
    inputs = load_inputs()
    print(
        publish(
            DRY,
            dict(
                complete=True,
                passed=True,
                inputs=input_receipt(inputs),
                sources_sha256=sources_digest(snapshot(ROOT)["sources"]),
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
    prefix.require(input_receipt(inputs) == registration["inputs"], "Registered sources must match")
    stats = dict(
        static_models_attempted=0,
        static_models_completed=0,
        forward_queries_attempted=0,
        forward_queries_completed=0,
        baseline_rollouts=0,
        max_qpos_restore_difference=0.0,
        support_lp_attempted=0,
        support_lp_completed=0,
        independent_first_responses_charged=0,
        independent_first_responses_completed=0,
        independent_transition_steps_charged=0,
        independent_transition_steps_completed=0,
    )
    result = dict(complete=False, passed=False, control_qualified=False, stats=stats)
    try:
        outcomes = analyze(inputs, stats)
        prefix.require(
            stats["static_models_attempted"] == stats["static_models_completed"] == 1
            and stats["forward_queries_attempted"] == stats["forward_queries_completed"] == 4
            and stats["baseline_rollouts"] == 4
            and stats["support_lp_attempted"] == stats["support_lp_completed"] == 40
            and stats["independent_first_responses_charged"] == stats["independent_first_responses_completed"] == 4
            and stats["independent_transition_steps_charged"] == stats["independent_transition_steps_completed"] == 100,
            "Registered static computation counts must match actual completions",
        )
        result.update(complete=True, passed=True, outcomes=outcomes)
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
        "The complete CPU regression and zero-model dry are required",
    )
    before = snapshot(ROOT)["sources"]
    prefix.require(
        before == prefix.read(qa["source_before_ref"])["sources"] and sources_digest(before) == dry["sources_sha256"],
        "Frozen CPU, dry and current source identity",
    )
    inputs = load_inputs()
    prefix.require(input_receipt(inputs) == dry["inputs"], "The dry input identity must match")
    OUTPUT.mkdir(parents=True, exist_ok=False)
    registration = publish(
        OUTPUT / "registration.json",
        dict(inputs=input_receipt(inputs), qa=prefix.reference(QA), dry=prefix.reference(DRY)),
    )
    lifecycle = run_owned_workers(
        [dict(key="bounds", argv=[sys.executable, "-m", MODULE, "--worker"], log=OUTPUT / "worker/worker.log")],
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
        lifecycle_check.require_closed_workers(lifecycle, ["bounds"])
        ref = prefix.reference(OUTPUT / "worker/result.json")
        result = prefix.read(ref)
        prefix.require(result["complete"] is result["passed"] is True, "The bounded static worker must finish")
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
