"""Compare two unchanged quality-constrained searches at four causal states."""

import argparse
import hashlib
import sys
import time
from pathlib import Path

import numpy as np

from s2a.geometry import inverse
from s2a.infrastructure.owned_processes import run_owned_workers
from s2a.infrastructure.publication import publish_result as publish
from s2a.infrastructure.publication import result_json
from s2a.references.generated_twist import reference_world_twist
from s2a.research.joint_control.tracking.early_quality_diagnostic import (
    completed_error_profile,
    independently_check_witness,
)
from s2a.research.trajectory_search.feasibility_first_joint_plan import solve_interior_plan as feasibility_solve
from s2a.research.trajectory_search.quality_capped_inner_joint_plan import solve_interior_plan as original_solve
from scripts.maintenance.run_cpu_tests import snapshot
from scripts.simulation.runtime import prepare_multitask_feasibility_first as binding
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.tracking.trajectory_search import audit_early_quality_feasibility as previous

REPO = prefix.REPO
OUTPUT = REPO / "outputs/multitask_feasibility_first_20261111"
DRY = REPO / "outputs/multitask_feasibility_first_dry_20261111/result.json"
QA = REPO / "outputs/qa/multitask_feasibility_first_20261111/result.json"


def source_refs():
    return (
        binding.source_refs()
        + previous.source_refs()
        + [
            prefix.reference(path)
            for path in (
                Path(__file__),
                REPO / "tests/research/native_runtime/test_multitask_feasibility_first_runtime.py",
            )
        ]
    )


def input_fingerprint(args, profile):
    """Hash every solver input, including dynamics, limits and OSC ranges."""
    interface = args[4]
    payload = dict(
        model=args[0],
        velocity=args[1],
        seed=args[2],
        previous=args[3],
        interface={key: getattr(interface, key) for key in ("input_min", "input_max", "output_min", "output_max")},
        site=args[5],
        tool=args[6],
        targets=args[7],
        twists=args[8],
        lower=args[9],
        upper=args[10],
        past_inner_error_rad=profile,
    )
    return hashlib.sha256(result_json(payload).encode()).hexdigest()


def compare_solvers(args, context, initial_joint, profile, stats, *, solvers=None, auditor=independently_check_witness):
    """Compare equal inputs and accept only independently audited model witnesses."""
    if solvers is None:
        solvers = (("original", original_solve), ("feasibility_first", feasibility_solve))
    if [name for name, _ in solvers] != ["original", "feasibility_first"]:
        raise ValueError("Exactly the registered original and feasibility-first solvers are required")
    fingerprint = input_fingerprint(args, profile)
    result = {}
    for label, solver in solvers:
        stats["optimizer_calls_attempted"] += 1
        started = time.monotonic()
        solved = solver(*args, past_inner_error_rad=profile)
        elapsed = time.monotonic() - started
        stats["optimizer_calls_completed"] += 1
        evaluations = int(solved["response_evaluations"])
        prefix.require(0 <= evaluations <= 160, "Unchanged per-solver response budget")
        stats["optimizer_responses"] += evaluations
        prefix.require(input_fingerprint(args, profile) == fingerprint, "A solver modified shared causal inputs")
        verification, audit_error = None, None
        if solved.get("witness_found") is True:
            stats["independent_endpoint_queries_charged"] += 200
            stats["independent_transition_steps_charged"] += 2600
            try:
                verification = auditor(
                    solved,
                    context["mass"],
                    context["jacobian"],
                    context["contract"]["kp"],
                    context["contract"]["kd"],
                    context["q"],
                    context["velocity"],
                    initial_joint,
                    context["interface"],
                    context["previous"][:6],
                    context["site"],
                    args[6],
                    args[7][0],
                    context["limits"][:, 0],
                    context["limits"][:, 1],
                    past_inner_error_rad=profile,
                )
                prefix.require(
                    verification.get("passed") is True
                    and verification.get("first_pose_caps_satisfied") is True
                    and verification.get("first_pose_caps_enforced") is True
                    and verification.get("braking_constraint_enforced") is True
                    and verification.get("independent_endpoint_queries") == 200
                    and verification.get("independent_transition_steps") == 2600,
                    "Full independent original constraint and first-pose audit",
                )
            except (AssertionError, ValueError) as error:
                audit_error = {"type": type(error).__name__, "message": str(error)}
                verification = None
        prefix.require(input_fingerprint(args, profile) == fingerprint, "Independent audit modified shared inputs")
        result[label] = {
            "solver_result": {key: value for key, value in solved.items() if key != "rollout"},
            "independent_verification": verification,
            "independent_audit_error": audit_error,
            "accepted_model_witness": verification is not None,
            "elapsed_seconds": elapsed,
            "input_sha256": fingerprint,
        }
    original = result["original"]["solver_result"]
    candidate = result["feasibility_first"]["solver_result"]
    for key in (
        "terminal_normals",
        "first_pose_caps",
        "joint_margin_rad",
        "braking_time_s",
        "requested_position_margin_rad",
    ):
        prefix.require((key in original) == (key in candidate), f"Both solvers must report {key}")
        if key in original:
            prefix.require(np.array_equal(original[key], candidate[key]), f"Unequal solver constraint: {key}")
    return {"input_sha256": fingerprint, "solvers": result}


def analyze(case, inputs, stats):
    """Reconstruct one current model without advancing the physical timeline."""
    previous.preload_software_gl(case, inputs["source_sha256"])
    import mujoco

    def prohibited(*_args, **_kwargs):
        raise RuntimeError("Static comparison forbids physical integration")

    for name in ("mj_step", "mj_step1", "mj_step2"):
        setattr(mujoco, name, prohibited)
    stats["model_initializations_attempted"] += 1
    model = mujoco.MjModel.from_xml_string(case["record_xml"])
    stats["model_initializations_completed"] += 1
    data = mujoco.MjData(model)
    spec = case["comparison_spec"]
    offset, start = spec["completed_feedback_steps"], case["spec"]["prefix_steps"]
    if spec["kind"] == "stopped_quality":
        stopped, periods = case["stop_inputs"], case["feedback_periods"]
        synthetic = periods + [{"step": stopped["stop_step"], "decision": {"refined_plan": stopped["seed"]}}]
    else:
        stopped, periods = None, case["feedback_periods"]
        synthetic = periods
    cache = {}

    def context(index):
        if index not in cache:
            cache[index] = previous.restored_context(model, data, case, synthetic, index, stats)
        return cache[index]

    profile = completed_error_profile(
        periods, start, offset, lambda index: context(index)["rollout"]["inner_joint_position"][:25]
    )
    if stopped is not None:
        np.testing.assert_allclose(
            profile["inner_position_error_rad"], stopped["past_inner_error_rad"], atol=1e-10, rtol=0
        )
    current = context(offset)
    if stopped is not None:
        np.testing.assert_array_equal(current["state"], stopped["state"])
        np.testing.assert_array_equal(current["previous"], stopped["previous"])
    site_from_tcp = inverse(np.asarray(case["controller"]["source_from_site"])) @ case["tool"]
    cursor = case["intervention_spec"]["cursor"] + offset
    targets = case["targets"][cursor : cursor + 8]
    twists = np.asarray(
        [
            reference_world_twist(case["targets"], index, case["anchor"], 0.05)["twist"]
            for index in range(cursor, cursor + 8)
        ]
    )
    prefix.require(targets.shape == (8, 4, 4) and twists.shape == (8, 6), "Complete previously generated target window")
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
    comparison = compare_solvers(args, current, case["initial_joint"], profile["inner_position_error_rad"], stats)
    return dict(
        case=spec,
        seed_scope=case["seed_scope"],
        current_reference_cursor=cursor,
        past_error_scope="recomputed_completed_feedback" if offset else "zero_unknown_at_warning_start",
        past_inner_error_rad=profile["inner_position_error_rad"],
        current_joint_margin_rad=float(
            np.min(np.minimum(current["q"] - current["limits"][:, 0], current["limits"][:, 1] - current["q"]))
        ),
        comparison=comparison,
        scope="frozen free-model comparison only; no new action or physical qualification",
    )


def worker(index):
    prefix.environment()
    registration = prefix.read(prefix.reference(OUTPUT / "registration.json"))
    inputs = binding.load_inputs()
    prefix.require(
        binding.input_receipt(inputs) == registration["inputs"] and source_refs() == registration["source_refs"],
        "Frozen comparison sources and causal inputs",
    )
    case = next(item for item in inputs["cases"] if item["comparison_spec"]["case_index"] == index)
    stats = dict(
        model_initializations_attempted=0,
        model_initializations_completed=0,
        forward_queries_attempted=0,
        forward_queries_completed=0,
        baseline_rollouts=0,
        max_qpos_restore_difference=0.0,
        optimizer_calls_attempted=0,
        optimizer_calls_completed=0,
        optimizer_responses=0,
        independent_endpoint_queries_charged=0,
        independent_transition_steps_charged=0,
    )
    result = dict(
        complete=False,
        passed=False,
        control_qualified=False,
        case=case["comparison_spec"],
        stats=stats,
        automatic_retry=False,
    )
    try:
        data = analyze(case, inputs, stats)
        expected_forwards = case["comparison_spec"]["completed_feedback_steps"] + 1
        prefix.require(
            stats["model_initializations_completed"] == 1
            and stats["forward_queries_completed"] == stats["baseline_rollouts"] == expected_forwards
            and stats["optimizer_calls_completed"] == 2
            and stats["optimizer_responses"] <= 320
            and stats["independent_endpoint_queries_charged"] <= 400
            and stats["independent_transition_steps_charged"] <= 5200,
            "Per-case static model, solver and independent audit costs",
        )
        result.update(complete=True, passed=True, data=data)
    except BaseException as error:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        print(publish(OUTPUT / f"case_{index:02d}" / "result.json", result), flush=True)


def execute():
    prefix.environment()
    qa, dry = prefix.read(prefix.reference(QA)), prefix.read(prefix.reference(DRY))
    prefix.require(
        qa["complete"] is qa["passed"] is qa["source_unchanged"] is qa["actual_wait"] is qa["reaped"] is True,
        "Frozen comparison QA closed",
    )
    before = snapshot(REPO)["sources"]
    prefix.require(before == prefix.read(qa["source_before_ref"])["sources"], "QA source identity")
    inputs, sources = binding.load_inputs(), source_refs()
    prefix.require(
        dry["complete"] is dry["passed"] is True
        and dry["inputs"] == binding.input_receipt(inputs)
        and dry["source_refs"] == sources,
        "Zero-model dry identity",
    )
    OUTPUT.mkdir(parents=True, exist_ok=False)
    registration = publish(
        OUTPUT / "registration.json", dict(inputs=binding.input_receipt(inputs), source_refs=sources)
    )
    specs = [
        dict(
            key=f"case_{item['case_index']:02d}",
            argv=[
                sys.executable,
                "-m",
                "scripts.tracking.trajectory_search.compare_multitask_feasibility_first",
                "--worker",
                str(item["case_index"]),
            ],
            log=OUTPUT / f"case_{item['case_index']:02d}" / "worker.log",
        )
        for item in binding.CASES
    ]
    lifecycle = run_owned_workers(specs, cwd=REPO, timeout_seconds=inputs["config"]["worker_timeout_seconds"])
    for item in lifecycle["workers"]:
        item["log_ref"] = prefix.reference(item.pop("log"))
    workers = publish(OUTPUT / "workers.json", lifecycle)
    available = [
        prefix.reference(OUTPUT / f"case_{item['case_index']:02d}" / "result.json")
        for item in binding.CASES
        if (OUTPUT / f"case_{item['case_index']:02d}" / "result.json").is_file()
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
        prefix.require(
            not lifecycle["launch_error"]
            and len(lifecycle["workers"]) == 4
            and all(
                item["returncode"] == 0 and item["actual_wait"] and item["reaped"] and not item["signal_sent"]
                for item in lifecycle["workers"]
            ),
            "All comparison workers exited and were reaped",
        )
        case_refs = [
            prefix.reference(OUTPUT / f"case_{item['case_index']:02d}" / "result.json") for item in binding.CASES
        ]
        cases = [prefix.read(ref) for ref in case_refs]
        prefix.require(all(item["complete"] is item["passed"] is True for item in cases), "All comparisons closed")
        keys = (
            "model_initializations_completed",
            "forward_queries_completed",
            "baseline_rollouts",
            "optimizer_calls_completed",
            "optimizer_responses",
            "independent_endpoint_queries_charged",
            "independent_transition_steps_charged",
        )
        counts = {key: sum(item["stats"][key] for item in cases) for key in keys}
        prefix.require(
            counts["model_initializations_completed"] == 4
            and counts["forward_queries_completed"] == counts["baseline_rollouts"] == 14
            and counts["optimizer_calls_completed"] == 8
            and counts["optimizer_responses"] <= 1280
            and counts["independent_endpoint_queries_charged"] <= 1600
            and counts["independent_transition_steps_charged"] <= 20800,
            "Registered complete numerical budget",
        )
        for path, digest in inputs["source_sha256"].items():
            prefix.checked({"path": path, "sha256": digest})
        prefix.require(
            source_refs() == sources and snapshot(REPO)["sources"] == before, "Comparison source identity unchanged"
        )
        result.update(complete=True, passed=True, sources_unchanged=True, case_refs=case_refs, counts=counts)
    finally:
        print(publish(OUTPUT / "result.json", result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true")
    group.add_argument("--run", action="store_true")
    group.add_argument("--worker", type=int, choices=[2, 6, 10, 11])
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
        execute()
    else:
        worker(args.worker)


if __name__ == "__main__":
    main()
