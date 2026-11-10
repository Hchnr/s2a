"""Compare first-pose outer bounds at four independently replayed stops."""

import argparse
import sys

import numpy as np

from s2a.geometry import inverse
from s2a.infrastructure.owned_processes import run_owned_workers
from s2a.infrastructure.publication import publish_result as publish
from s2a.references.generated_twist import reference_world_twist
from s2a.research.joint_control.tracking.stopped_pose_safety_bound import (
    error_directions,
    first_pose_linearization,
    safety_scopes,
    weighted_pose_bounds,
)
from s2a.research.trajectory_search.feasible_start_restoration import linear_domain
from scripts.maintenance.run_cpu_tests import snapshot
from scripts.simulation.runtime import prepare_stopped_pose_safety_bound as binding
from scripts.simulation.runtime import registered_feedback_phase as lifecycle_check
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.tracking.trajectory_search import audit_early_quality_feasibility as previous
from scripts.tracking.trajectory_search import compare_multitask_feasibility_first as comparison
from scripts.verification.reference_ports.check_stopped_pose_safety_bound import check_candidate

REPO = prefix.REPO
QA = REPO / "outputs/qa/stopped_pose_safety_bound_20261111/result.json"
DRY = REPO / "outputs/stopped_pose_safety_bound_dry_20261111/result.json"
OUTPUT = REPO / "outputs/stopped_pose_safety_bound_20261111"
INDICES = (2, 6, 10, 11)


def analyze(case, inputs, stats):
    previous.preload_software_gl(case, inputs["source_sha256"])
    import mujoco

    def prohibited(*_args, **_kwargs):
        raise RuntimeError("Static pose-bound analysis forbids physical integration")

    for name in ("mj_step", "mj_step1", "mj_step2"):
        setattr(mujoco, name, prohibited)
    stats["model_initializations_attempted"] += 1
    model = mujoco.MjModel.from_xml_string(case["record_xml"])
    stats["model_initializations_completed"] += 1
    data = mujoco.MjData(model)
    offset = len(case["feedback_periods"])
    periods = case["feedback_periods"] + [
        {"step": case["bound_spec"]["step"], "decision": {"refined_plan": case["bound_seed"]}}
    ]
    current = previous.restored_context(model, data, case, periods, offset, stats)
    last = case["feedback_periods"][-1]
    np.testing.assert_array_equal(current["state"], last["actual_state"])
    np.testing.assert_array_equal(current["previous"], last["command"])
    np.testing.assert_array_equal(current["plan"], case["bound_seed"])
    site_from_tcp = inverse(np.asarray(case["controller"]["source_from_site"])) @ case["tool"]
    cursor = case["intervention_spec"]["cursor"] + offset
    targets = case["targets"][cursor : cursor + 8]
    twists = np.asarray(
        [reference_world_twist(case["targets"], i, case["anchor"], 0.05)["twist"] for i in range(cursor, cursor + 8)]
    )
    prefix.require(targets.shape == (8, 4, 4) and twists.shape == (8, 6), "Already-generated reference window")
    args = (
        current["model"],
        current["velocity"],
        current["plan"],
        current["previous"][:6],
        current["interface"],
        current["site"],
        site_from_tcp,
        targets,
        twists,
        current["limits"][:, 0],
        current["limits"][:, 1],
    )
    fingerprint = comparison.input_fingerprint(args, case["bound_profile"])
    if case["previous_fingerprint"] is not None:
        prefix.require(
            fingerprint == case["previous_fingerprint"], "The first two stops retain their original model inputs"
        )
    rollout, interface = current["rollout"], current["interface"]
    domain = linear_domain(
        current["model"],
        current["plan"],
        current["previous"][:6],
        interface,
        current["limits"][:, 0],
        current["limits"][:, 1],
        case["bound_profile"],
        rollout,
        terminal_normals=np.empty((0, 7)),
    )
    scopes = safety_scopes(domain, current["plan"])
    scale = (interface.output_max - interface.output_min) / (interface.input_max - interface.input_min)
    seed_q = rollout["joint_position"][0]
    command_jacobian = rollout["joint_position_jacobian"][0, :, :6] * scale
    linearization = first_pose_linearization(
        current["site"],
        site_from_tcp,
        current["jacobian"],
        seed_q - current["q"],
        command_jacobian,
        current["plan"][0],
        scopes["box"]["lower"],
        scopes["box"]["upper"],
        targets[0],
    )
    seed_tcp = rollout["pose"][0] @ site_from_tcp
    np.testing.assert_allclose(linearization["seed_tcp_position"], seed_tcp[:3, 3], atol=1e-10, rtol=0)
    np.testing.assert_allclose(linearization["seed_tcp_rotation"], seed_tcp[:3, :3], atol=1e-10, rtol=0)
    position_direction, rotation_direction = error_directions(targets[0, :3, :3], case["closest_error"])

    def account(name, event):
        if name != "box":
            stats[f"support_lp_{event}"] += 1
            prefix.require(stats["support_lp_attempted"] <= 10, "Per-case support LP budget")

    bounds = weighted_pose_bounds(
        linearization,
        position_direction,
        rotation_direction,
        scopes,
        inputs["config"]["position_weights"],
        allowance=inputs["config"]["normalized_numerical_allowance"],
        on_support=account,
    )
    stats["independent_first_responses_charged"] += 1
    stats["independent_transition_steps_charged"] += 25
    verification = check_candidate(
        current["model"],
        current["velocity"],
        interface,
        current["site"],
        site_from_tcp,
        targets[0],
        current["limits"][:, 0],
        current["limits"][:, 1],
        seed_q,
        command_jacobian,
        case["closest_commands"],
        case["closest_error"],
        linearization,
        scopes,
        bounds,
    )
    stats["independent_first_responses_completed"] += verification["independent_first_responses"]
    stats["independent_transition_steps_completed"] += verification["independent_transition_steps"]
    return dict(
        case=case["bound_spec"],
        seed_scope=case["seed_scope"],
        first_target_cursor=cursor,
        input_sha256=fingerprint,
        linearization=linearization,
        bounds=bounds,
        independently_checked_candidate=verification,
        excludes_first_pose_caps={name: value["best_lower_bound"] > 1 for name, value in bounds["scopes"].items()},
        scope="floating-point outer relaxation of frozen first-pose and joint model; not a physical feasibility claim",
    )


def worker(index):
    prefix.environment()
    registration = prefix.read(prefix.reference(OUTPUT / "registration.json"))
    inputs = binding.load_inputs()
    prefix.require(
        binding.input_receipt(inputs) == registration["inputs"]
        and binding.source_refs() == registration["source_refs"],
        "Frozen stop, reference and source identity",
    )
    case = next(item for item in inputs["cases"] if item["bound_spec"]["case_index"] == index)
    stats = dict(
        model_initializations_attempted=0,
        model_initializations_completed=0,
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
    result = dict(complete=False, passed=False, control_qualified=False, case=case["bound_spec"], stats=stats)
    try:
        data = analyze(case, inputs, stats)
        prefix.require(
            stats["model_initializations_attempted"]
            == stats["model_initializations_completed"]
            == stats["forward_queries_attempted"]
            == stats["forward_queries_completed"]
            == stats["baseline_rollouts"]
            == stats["independent_first_responses_charged"]
            == stats["independent_first_responses_completed"]
            == 1
            and stats["support_lp_attempted"] == stats["support_lp_completed"] == 10
            and stats["independent_transition_steps_charged"] == stats["independent_transition_steps_completed"] == 25,
            "Exact per-case static and independent response costs",
        )
        result.update(complete=True, passed=True, data=data)
    except BaseException as error:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        print(publish(OUTPUT / f"case_{index:02d}" / "result.json", result), flush=True)


def execute():
    prefix.environment()
    qa_ref, dry_ref = prefix.reference(QA), prefix.reference(DRY)
    qa, dry = prefix.read(qa_ref), prefix.read(dry_ref)
    prefix.require(
        qa["complete"] is qa["passed"] is qa["source_unchanged"] is qa["actual_wait"] is qa["reaped"] is True,
        "Closed CPU QA",
    )
    before = snapshot(REPO)["sources"]
    prefix.require(before == prefix.read(qa["source_before_ref"])["sources"], "QA source identity")
    inputs, sources = binding.load_inputs(), binding.source_refs()
    prefix.require(
        dry["complete"] is dry["passed"] is True
        and dry["inputs"] == binding.input_receipt(inputs)
        and dry["source_refs"] == sources,
        "Zero-model dry input and source identity",
    )
    OUTPUT.mkdir(parents=True, exist_ok=False)
    registration = publish(
        OUTPUT / "registration.json",
        dict(inputs=binding.input_receipt(inputs), source_refs=sources, qa_ref=qa_ref, dry_ref=dry_ref),
    )
    keys = [f"case_{index:02d}" for index in INDICES]
    specs = [
        dict(
            key=key,
            argv=[
                sys.executable,
                "-m",
                "scripts.tracking.trajectory_search.audit_stopped_pose_safety_bound",
                "--worker",
                str(index),
            ],
            log=OUTPUT / key / "worker.log",
        )
        for key, index in zip(keys, INDICES, strict=True)
    ]
    lifecycle = run_owned_workers(specs, cwd=REPO, timeout_seconds=inputs["config"]["worker_timeout_seconds"])
    for item in lifecycle["workers"]:
        item["log_ref"] = prefix.reference(item.pop("log"))
    workers = publish(OUTPUT / "workers.json", lifecycle)
    available = [
        prefix.reference(OUTPUT / key / "result.json") for key in keys if (OUTPUT / key / "result.json").is_file()
    ]
    result = dict(
        complete=False,
        passed=False,
        control_qualified=False,
        registration_ref=registration,
        workers_ref=workers,
        available_case_refs=available,
        physics_steps=0,
        world_model_calls=0,
        fresh_cases=0,
        automatic_retry=False,
    )
    try:
        lifecycle_check.require_closed_workers(lifecycle, keys)
        case_refs = [prefix.reference(OUTPUT / key / "result.json") for key in keys]
        cases = [prefix.read(ref) for ref in case_refs]
        prefix.require(all(item["complete"] is item["passed"] is True for item in cases), "All four bound cases closed")
        stats_keys = (
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
        counts = {key: sum(item["stats"][key] for item in cases) for key in stats_keys}
        prefix.require(
            counts["model_initializations_completed"]
            == counts["forward_queries_completed"]
            == counts["baseline_rollouts"]
            == counts["independent_first_responses_charged"]
            == counts["independent_first_responses_completed"]
            == 4
            and counts["support_lp_attempted"] == counts["support_lp_completed"] == 40
            and counts["independent_transition_steps_charged"]
            == counts["independent_transition_steps_completed"]
            == 100,
            "Exact total static, support and independent response costs",
        )
        for path, digest in inputs["source_sha256"].items():
            prefix.checked({"path": path, "sha256": digest})
        prefix.require(
            binding.source_refs() == sources and snapshot(REPO)["sources"] == before, "All sources remained unchanged"
        )
        result.update(complete=True, passed=True, sources_unchanged=True, case_refs=case_refs, counts=counts)
    finally:
        print(publish(OUTPUT / "result.json", result), flush=True)


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
                    source_refs=binding.source_refs(),
                    model_initializations=0,
                    physics_steps=0,
                ),
            )
        )
    elif args.run:
        execute()
    else:
        worker(args.worker)


if __name__ == "__main__":
    main()
