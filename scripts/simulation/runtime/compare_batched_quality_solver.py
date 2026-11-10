"""Compare unchanged quality optimization with two explicit rollout implementations."""

import argparse
import platform
import sys
import time
from pathlib import Path

import numpy as np

from s2a.controllers.osc import OSCInterface
from s2a.infrastructure.owned_processes import run_owned_workers
from s2a.infrastructure.publication import publish_result as publish
from s2a.research.joint_control.preview.batched_free_joint_rollout import BatchedFreeJointRollout
from s2a.research.joint_control.preview.interior_joint_preview import build_joint_preview
from s2a.research.joint_control.tracking.early_quality_diagnostic import independently_check_witness
from s2a.research.trajectory_search.quality_capped_inner_joint_plan import solve_interior_plan
from scripts.maintenance.run_cpu_tests import snapshot
from scripts.simulation.runtime import prepare_batched_quality_solver as binding
from scripts.simulation.runtime import prepare_quality_solver_runtime_profile as saved
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix

REPO = prefix.REPO
QA = REPO / "outputs/qa/batched_quality_solver_20261111/result.json"
DRY = REPO / "outputs/batched_quality_solver_dry_20261111/result.json"
ROOT = REPO / "outputs/batched_quality_solver_20261111"


def source_refs():
    return binding.source_refs() + [
        prefix.reference(path)
        for path in (Path(__file__), REPO / "tests/research/native_runtime/test_batched_quality_solver_comparison.py")
    ]


def timed(call):
    wall, cpu = time.perf_counter(), time.process_time()
    value = call()
    return value, {"wall_seconds": time.perf_counter() - wall, "process_cpu_seconds": time.process_time() - cpu}


def record_audit(call, ledger):
    """Keep ordinary audit rejection as evidence while preserving errors."""
    ledger["independent_audits_attempted"] += 1
    try:
        verified = call()
    except (AssertionError, ValueError) as error:
        verified = {"passed": False, "error": {"type": type(error).__name__, "message": str(error)}}
    ledger["independent_audits_completed"] += 1
    ledger["independent_audits_passed" if verified.get("passed") is True else "independent_audits_failed"] += 1
    return verified


def compare_runs(first, second, *, atol):
    """Report differences; a mismatch is evidence, not a reason to rerun."""
    same_status = first["status"] == second["status"] and first["witness_found"] == second["witness_found"]
    result = {
        "same_status_and_witness": same_status,
        "response_evaluation_difference": second["response_evaluations"] - first["response_evaluations"],
        "same_optimizer_status": first.get("optimizer_success") == second.get("optimizer_success")
        and first.get("iterations") == second.get("iterations"),
        "both_independent_audits_passed": bool(
            (first.get("independent_audit") or {}).get("passed") is True
            and (second.get("independent_audit") or {}).get("passed") is True
        ),
        "planned_command_max_abs_difference": None,
        "first_tcp_error_max_abs_difference": None,
        "objective_abs_difference": None,
        "equivalent_at_tolerance": False,
    }
    if first["witness_found"] is True and second["witness_found"] is True:
        plan_diff = float(
            np.max(np.abs(np.asarray(first["planned_commands"]) - np.asarray(second["planned_commands"])))
        )
        error_diff = float(np.max(np.abs(np.asarray(first["first_tcp_error"]) - np.asarray(second["first_tcp_error"]))))
        objective_diff = float(abs(first["objective"] - second["objective"]))
        result.update(
            planned_command_max_abs_difference=plan_diff,
            first_tcp_error_max_abs_difference=error_diff,
            objective_abs_difference=objective_diff,
            equivalent_at_tolerance=bool(
                same_status
                and result["response_evaluation_difference"] == 0
                and result["same_optimizer_status"]
                and result["both_independent_audits_passed"]
                and max(plan_diff, error_diff, objective_diff) <= atol
            ),
        )
    return result


def run_sample(sample, config, ledger):
    """Run both solvers on one frozen local model and audit every witness."""
    model, model_time = timed(
        lambda: build_joint_preview(
            sample["mass"],
            sample["jacobian"],
            sample["kp"],
            sample["kd"],
            sample["q"],
            sample["initial_joint"],
            null_kp=10.0,
            dt=0.002,
        )
    )
    candidate, prepare_time = timed(lambda: BatchedFreeJointRollout(model))
    contract = sample["contract"]
    interface = OSCInterface(*(contract[key] for key in ("input_min", "input_max", "output_min", "output_max")))

    def batched_rollout(actual_model, velocity, deltas, site, lower, upper, *, enforce_limits):
        prefix.require(actual_model is model, "Prepared rollout is bound to the current frozen model")
        return candidate.rollout(velocity, deltas, site, lower, upper, enforce_limits=enforce_limits)

    def solve(mode):
        ledger["solver_calls_attempted"] += 1
        kwargs = {"past_inner_error_rad": sample["past_inner_error_rad"]}
        if mode == "batched":
            kwargs["rollout_fn"] = batched_rollout
        result = solve_interior_plan(
            model,
            sample["velocity"],
            sample["seed"],
            sample["previous"],
            interface,
            sample["site"],
            sample["site_from_tcp"],
            sample["targets"],
            sample["twists"],
            sample["lower"],
            sample["upper"],
            **kwargs,
        )
        count = result["response_evaluations"]
        prefix.require(
            type(count) is int and 0 <= count <= config["max_response_evaluations_per_call"],
            "Original per-call solver response budget",
        )
        ledger["solver_calls_completed"] += 1
        ledger["response_evaluations_completed"] += count
        return result

    def audit(result):
        return record_audit(
            lambda: independently_check_witness(
                result,
                sample["mass"],
                sample["jacobian"],
                sample["kp"],
                sample["kd"],
                sample["q"],
                sample["velocity"],
                sample["initial_joint"],
                interface,
                sample["previous"],
                sample["site"],
                sample["site_from_tcp"],
                sample["targets"][0],
                sample["lower"],
                sample["upper"],
                past_inner_error_rad=sample["past_inner_error_rad"],
            ),
            ledger,
        )

    records = {"sequential": [], "batched": []}
    order = []
    for repetition in range(config["repetitions_per_implementation"]):
        modes = ("sequential", "batched") if (sample["step"] + repetition) % 2 == 0 else ("batched", "sequential")
        for mode in modes:
            result, solve_time = timed(lambda mode=mode: solve(mode))
            verified, audit_time = (None, None)
            if result["witness_found"] is True:
                verified, audit_time = timed(lambda result=result: audit(result))
            records[mode].append(
                {
                    "status": result["status"],
                    "witness_found": result["witness_found"],
                    "response_evaluations": result["response_evaluations"],
                    "optimizer_success": result["optimizer_success"],
                    "iterations": result.get("iterations"),
                    "optimizer_message": result.get("optimizer_message"),
                    "objective": result.get("objective"),
                    "first_tcp_error": result.get("first_tcp_error"),
                    "planned_commands": result.get("planned_commands"),
                    "minimum_tightened_inner_slack_rad": result.get("minimum_tightened_inner_slack_rad"),
                    "minimum_braking_margin_rad": result.get("minimum_braking_margin_rad"),
                    "solve_time": solve_time,
                    "independent_audit_time": audit_time,
                    "independent_audit": verified,
                }
            )
            order.append(mode)
    tolerance = config["comparison_atol"]
    return {
        "step": sample["step"],
        "sample_sha256": saved.sample_fingerprint(sample),
        "model_build_time": model_time,
        "batched_prepare_time": prepare_time,
        "execution_order": order,
        "records": records,
        "paired_comparisons": [
            compare_runs(a, b, atol=tolerance) for a, b in zip(records["sequential"], records["batched"], strict=True)
        ],
        "within_implementation_comparisons": {
            mode: compare_runs(rows[0], rows[1], atol=tolerance) for mode, rows in records.items()
        },
    }


def worker(step):
    prefix.environment()
    directory = ROOT / f"step_{step}"
    registration = prefix.read(prefix.reference(ROOT / "registration.json"))
    inputs = binding.load_inputs()
    prefix.require(
        binding.input_receipt(inputs) == registration["inputs"] and source_refs() == registration["source_refs"],
        "Frozen solver comparison inputs and source",
    )
    prefix.require(step in inputs["config"]["steps"], "One registered solver comparison sample")
    ledger = {
        "solver_calls_attempted": 0,
        "solver_calls_completed": 0,
        "response_evaluations_completed": 0,
        "independent_audits_attempted": 0,
        "independent_audits_completed": 0,
        "independent_audits_passed": 0,
        "independent_audits_failed": 0,
    }
    result = {
        "complete": False,
        "passed": False,
        "control_qualified": False,
        "step": step,
        "ledger": ledger,
        "automatic_retry": False,
    }
    try:
        result["data"] = run_sample(inputs["samples"][str(step)], inputs["config"], ledger)
        result.update(complete=True, passed=True)
    except BaseException as error:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        print(publish(directory / "result.json", result), flush=True)


def execute():
    prefix.environment()
    qa, dry = prefix.read(prefix.reference(QA)), prefix.read(prefix.reference(DRY))
    prefix.require(
        qa["complete"] is qa["passed"] is qa["source_unchanged"] is qa["actual_wait"] is qa["reaped"] is True,
        "Frozen solver comparison CPU QA closed",
    )
    before = snapshot(REPO)["sources"]
    prefix.require(before == prefix.read(qa["source_before_ref"])["sources"], "CPU QA source identity")
    inputs, sources = binding.load_inputs(), source_refs()
    prefix.require(
        dry["complete"] is dry["passed"] is True
        and dry["inputs"] == binding.input_receipt(inputs)
        and dry["source_refs"] == sources,
        "Zero-model dry preserves solver inputs",
    )
    ROOT.mkdir(parents=True, exist_ok=False)
    registration_ref = publish(
        ROOT / "registration.json", {"inputs": binding.input_receipt(inputs), "source_refs": sources}
    )
    specs = [
        {
            "key": str(step),
            "argv": [
                sys.executable,
                "-m",
                "scripts.simulation.runtime.compare_batched_quality_solver",
                "--worker",
                str(step),
            ],
            "log": ROOT / f"step_{step}/worker.log",
        }
        for step in inputs["config"]["steps"]
    ]
    lifecycle = run_owned_workers(specs, cwd=REPO, timeout_seconds=inputs["config"]["worker_timeout_seconds"])
    for item in lifecycle["workers"]:
        item["log_ref"] = prefix.reference(item.pop("log"))
    workers_ref = publish(ROOT / "workers.json", lifecycle)
    paths = {str(step): ROOT / f"step_{step}/result.json" for step in inputs["config"]["steps"]}
    result = {
        "complete": False,
        "passed": False,
        "control_qualified": False,
        "registration_ref": registration_ref,
        "workers_ref": workers_ref,
        "available_sample_refs": {step: prefix.reference(path) for step, path in paths.items() if path.is_file()},
        "physical_steps": 0,
        "world_model_calls": 0,
        "fresh_cases": 0,
        "automatic_retry": False,
        "python": sys.version,
        "platform": platform.platform(),
    }
    try:
        prefix.require(
            not lifecycle["launch_error"]
            and len(lifecycle["workers"]) == inputs["config"]["max_workers"]
            and all(
                item["returncode"] == 0 and item["actual_wait"] and item["reaped"] and not item["signal_sent"]
                for item in lifecycle["workers"]
            ),
            "Registered solver comparison workers closed and were reaped",
        )
        sample_refs = {step: prefix.reference(path) for step, path in paths.items()}
        samples = [prefix.read(sample_refs[step]) for step in sample_refs]
        prefix.require(all(sample["complete"] is sample["passed"] is True for sample in samples), "Complete samples")
        totals = {key: sum(sample["ledger"][key] for sample in samples) for key in samples[0]["ledger"]}
        prefix.require(
            totals["solver_calls_attempted"] == totals["solver_calls_completed"] == inputs["config"]["max_solver_calls"]
            and totals["response_evaluations_completed"] <= inputs["config"]["max_response_evaluations"]
            and totals["independent_audits_attempted"] == totals["independent_audits_completed"]
            and totals["independent_audits_passed"] + totals["independent_audits_failed"]
            == totals["independent_audits_completed"]
            and totals["independent_audits_completed"] <= inputs["config"]["max_independent_audits"],
            "Registered solver and audit budgets",
        )
        for path, digest in inputs["source_sha256"].items():
            prefix.checked({"path": path, "sha256": digest})
        prefix.require(
            source_refs() == sources and snapshot(REPO)["sources"] == before, "Unchanged solver sources and inputs"
        )
        result.update(complete=True, passed=True, sources_unchanged=True, sample_refs=sample_refs, counts=totals)
    finally:
        print(publish(ROOT / "result.json", result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true")
    group.add_argument("--execute", action="store_true")
    group.add_argument("--worker", type=int)
    args = parser.parse_args()
    prefix.environment()
    if args.dry_run:
        print(
            publish(
                DRY,
                {
                    "complete": True,
                    "passed": True,
                    "source_refs": source_refs(),
                    "inputs": binding.input_receipt(binding.load_inputs()),
                    "model_initializations": 0,
                    "physics_steps": 0,
                    "world_model_calls": 0,
                    "fresh_cases": 0,
                },
            ),
            flush=True,
        )
    elif args.execute:
        execute()
    else:
        worker(args.worker)


if __name__ == "__main__":
    main()
