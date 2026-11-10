"""Profile the unchanged quality solver on three saved causal development states."""

import argparse
import cProfile
import json
import platform
import pstats
import sys
import time
from pathlib import Path

from s2a.controllers.osc import OSCInterface
from s2a.infrastructure.owned_processes import run_owned_workers
from s2a.infrastructure.publication import publish_result as publish
from s2a.infrastructure.publication import result_json
from s2a.research.gripper.causal_loaded_history import compare_recorded
from s2a.research.joint_control.preview.interior_joint_preview import build_joint_preview
from s2a.research.joint_control.tracking.early_quality_diagnostic import independently_check_witness
from s2a.research.trajectory_search.quality_capped_inner_joint_plan import solve_interior_plan
from scripts.maintenance.run_cpu_tests import snapshot
from scripts.simulation.runtime import prepare_quality_solver_runtime_profile as binding
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix

REPO = prefix.REPO
QA = REPO / "outputs/qa/quality_solver_runtime_profile_20261111/result.json"
DRY = REPO / "outputs/quality_solver_runtime_profile_dry_20261111/result.json"
ROOT = REPO / "outputs/quality_solver_runtime_profile_20261111"


def source_refs():
    return binding.source_refs() + [
        prefix.reference(path)
        for path in (Path(__file__), REPO / "tests/research/native_runtime/test_quality_solver_runtime_profile.py")
    ]


def profile_rows(profiler):
    """Store self and cumulative time, without claiming additive nested times."""
    stats = pstats.Stats(profiler)
    rows = [
        {
            "file": filename,
            "line": line,
            "function": name,
            "primitive_calls": primitive,
            "total_calls": calls,
            "self_seconds": self_time,
            "cumulative_seconds": cumulative,
        }
        for (filename, line, name), (primitive, calls, self_time, cumulative, _callers) in stats.stats.items()
    ]
    return sorted(rows, key=lambda row: (-row["self_seconds"], row["file"], row["line"], row["function"]))


def timed(call):
    wall, cpu = time.perf_counter(), time.process_time()
    value = call()
    return value, {"wall_seconds": time.perf_counter() - wall, "process_cpu_seconds": time.process_time() - cpu}


def run_sample(
    sample,
    config,
    ledger,
    *,
    builder=build_joint_preview,
    solver=solve_interior_plan,
    auditor=independently_check_witness,
):
    """Measure identical inputs once without and once with Python profiling."""
    model, build_time = timed(
        lambda: builder(
            sample["mass"],
            sample["jacobian"],
            sample["kp"],
            sample["kd"],
            sample["q"],
            sample["initial_joint"],
            null_kp=config["null_kp"],
            dt=config["inner_dt_s"],
        )
    )
    contract = sample["contract"]
    interface = OSCInterface(*(contract[key] for key in ("input_min", "input_max", "output_min", "output_max")))

    def solve():
        ledger["solver_calls_attempted"] += 1
        result = solver(
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
            past_inner_error_rad=sample["past_inner_error_rad"],
        )
        count = result["response_evaluations"]
        prefix.require(
            type(count) is int and 0 <= count <= config["max_response_evaluations_per_call"],
            "Original per-call solver response budget",
        )
        ledger["solver_calls_completed"] += 1
        ledger["response_evaluations_completed"] += count
        return result

    first, unprofiled_time = timed(solve)
    profiler = cProfile.Profile()

    def profiled_solve():
        profiler.enable()
        try:
            return solve()
        finally:
            profiler.disable()

    second, profiled_time = timed(profiled_solve)
    difference = compare_recorded(json.loads(result_json(first)), json.loads(result_json(second)), atol=1e-10)
    audit, audit_time = None, None
    if first["witness_found"] is True:
        ledger["independent_audits_attempted"] += 1
        audit, audit_time = timed(
            lambda: auditor(
                first,
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
            )
        )
        prefix.require(audit["passed"] is True, "Original independent witness audit passed")
        ledger["independent_audits_completed"] += 1
    return {
        "step": sample["step"],
        "sample_sha256": binding.sample_fingerprint(sample),
        "model_build_time": build_time,
        "unprofiled_solver_time": unprofiled_time,
        "profiled_solver_time": profiled_time,
        "independent_audit_time": audit_time,
        "solver_status": first["status"],
        "witness_found": first["witness_found"],
        "response_evaluations_per_call": first["response_evaluations"],
        "optimizer_success": first["optimizer_success"],
        "iterations": first.get("iterations"),
        "planned_commands": first.get("planned_commands"),
        "first_tcp_error": first.get("first_tcp_error"),
        "repeated_result_max_difference": difference,
        "independent_audit": audit,
        "profile_rows": profile_rows(profiler),
    }


def worker(step):
    prefix.environment()
    directory = ROOT / f"step_{step}"
    registration = prefix.read(prefix.reference(ROOT / "registration.json"))
    inputs = binding.load_inputs()
    prefix.require(
        binding.input_receipt(inputs) == registration["inputs"] and source_refs() == registration["source_refs"],
        "Frozen quality profile inputs and source",
    )
    prefix.require(step in inputs["config"]["steps"], "One registered timing sample")
    ledger = {
        "solver_calls_attempted": 0,
        "solver_calls_completed": 0,
        "response_evaluations_completed": 0,
        "independent_audits_attempted": 0,
        "independent_audits_completed": 0,
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
        "Frozen quality profile CPU QA closed",
    )
    before = snapshot(REPO)["sources"]
    prefix.require(before == prefix.read(qa["source_before_ref"])["sources"], "CPU QA source identity")
    inputs, sources = binding.load_inputs(), source_refs()
    prefix.require(
        dry["complete"] is dry["passed"] is True
        and dry["inputs"] == binding.input_receipt(inputs)
        and dry["source_refs"] == sources,
        "Zero-model dry preserves timing inputs",
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
                "scripts.simulation.runtime.profile_quality_solver_runtime",
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
            "Registered profile workers closed and were reaped",
        )
        sample_refs = {step: prefix.reference(path) for step, path in paths.items()}
        samples = [prefix.read(sample_refs[step]) for step in sample_refs]
        prefix.require(all(sample["complete"] is sample["passed"] is True for sample in samples), "Complete samples")
        totals = {key: sum(sample["ledger"][key] for sample in samples) for key in samples[0]["ledger"]}
        prefix.require(
            totals["solver_calls_attempted"] == totals["solver_calls_completed"] == inputs["config"]["max_solver_calls"]
            and totals["response_evaluations_completed"] <= inputs["config"]["max_response_evaluations"]
            and totals["independent_audits_attempted"] == totals["independent_audits_completed"]
            and totals["independent_audits_completed"] <= inputs["config"]["max_independent_audits"],
            "Registered solver and audit budgets",
        )
        for path, digest in inputs["source_sha256"].items():
            prefix.checked({"path": path, "sha256": digest})
        prefix.require(
            source_refs() == sources and snapshot(REPO)["sources"] == before, "Unchanged timing sources and inputs"
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
