"""Compare free-model batched and sequential rollouts on fixed saved states."""

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
from s2a.research.joint_control.preview.interior_joint_preview import build_joint_preview, rollout_joint_preview
from scripts.maintenance.run_cpu_tests import snapshot
from scripts.simulation.runtime import prepare_batched_joint_rollout as binding
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix

REPO = prefix.REPO
QA = REPO / "outputs/qa/batched_joint_rollout_20261111/result.json"
DRY = REPO / "outputs/batched_joint_rollout_dry_20261111/result.json"
ROOT = REPO / "outputs/batched_joint_rollout_20261111"


def source_refs():
    return binding.source_refs() + [
        prefix.reference(path)
        for path in (Path(__file__), REPO / "tests/research/native_runtime/test_batched_joint_rollout_benchmark.py")
    ]


def compare_outputs(original, candidate, *, atol=1e-10):
    """Check every field, retaining discrete equality and per-field error."""
    prefix.require(set(original) == set(candidate), "Identical complete rollout output fields")
    differences = {}
    for key in original:
        a, b = np.asarray(original[key]), np.asarray(candidate[key])
        prefix.require(a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all(), f"Finite {key} shape")
        if a.dtype.kind in "biu" or b.dtype.kind in "biu":
            np.testing.assert_array_equal(a, b, err_msg=key)
            differences[key] = 0.0
        else:
            np.testing.assert_allclose(a, b, atol=atol, rtol=0, err_msg=key)
            differences[key] = float(np.max(np.abs(a - b))) if a.size else 0.0
    return differences


def timed(call):
    wall, cpu = time.perf_counter(), time.process_time()
    value = call()
    return value, {"wall_seconds": time.perf_counter() - wall, "process_cpu_seconds": time.process_time() - cpu}


def benchmark_sample(sample, config, ledger):
    """Alternate unchanged inputs and compare every warm and timed result."""
    model, model_time = timed(
        lambda: build_joint_preview(
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
    candidate, prepare_time = timed(lambda: BatchedFreeJointRollout(model))
    contract = sample["contract"]
    interface = OSCInterface(*(contract[key] for key in ("input_min", "input_max", "output_min", "output_max")))
    deltas = np.asarray([interface.decode(command) for command in sample["seed"]])

    def call(mode):
        ledger[f"{mode}_calls_attempted"] += 1
        args = (sample["velocity"], deltas, sample["site"], sample["lower"], sample["upper"])
        value = (
            rollout_joint_preview(model, *args, enforce_limits=False)
            if mode == "sequential"
            else candidate.rollout(*args, enforce_limits=False)
        )
        ledger[f"{mode}_calls_completed"] += 1
        return value

    first, _ = timed(lambda: call("sequential"))
    second, _ = timed(lambda: call("batched"))
    maximum = compare_outputs(first, second, atol=config["comparison_atol"])
    timings = {"sequential": [], "batched": []}
    for iteration in range(config["timed_calls_per_implementation"]):
        order = ("sequential", "batched") if iteration % 2 == 0 else ("batched", "sequential")
        values = {}
        for mode in order:
            values[mode], elapsed = timed(lambda mode=mode: call(mode))
            timings[mode].append(elapsed)
        difference = compare_outputs(values["sequential"], values["batched"], atol=config["comparison_atol"])
        maximum = {key: max(maximum[key], difference[key]) for key in maximum}
    summaries = {}
    for mode, rows in timings.items():
        wall = np.asarray([row["wall_seconds"] for row in rows])
        summaries[mode] = {
            "wall_median_seconds": float(np.median(wall)),
            "wall_p90_seconds": float(np.quantile(wall, 0.9)),
            "wall_min_seconds": float(np.min(wall)),
            "wall_max_seconds": float(np.max(wall)),
        }
    return {
        "step": sample["step"],
        "sample_sha256": binding.saved.sample_fingerprint(sample),
        "model_build_time": model_time,
        "batched_prepare_time": prepare_time,
        "timings": timings,
        "timing_summaries": summaries,
        "max_abs_difference_by_field": maximum,
        "compared_pairs": config["warmup_calls_per_implementation"] + config["timed_calls_per_implementation"],
    }


def worker(step):
    prefix.environment()
    directory = ROOT / f"step_{step}"
    registration = prefix.read(prefix.reference(ROOT / "registration.json"))
    inputs = binding.load_inputs()
    prefix.require(
        binding.input_receipt(inputs) == registration["inputs"] and source_refs() == registration["source_refs"],
        "Frozen rollout benchmark inputs and source",
    )
    prefix.require(step in inputs["config"]["steps"], "One registered benchmark sample")
    ledger = {
        "sequential_calls_attempted": 0,
        "sequential_calls_completed": 0,
        "batched_calls_attempted": 0,
        "batched_calls_completed": 0,
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
        result["data"] = benchmark_sample(inputs["samples"][str(step)], inputs["config"], ledger)
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
        "Frozen rollout CPU QA closed",
    )
    before = snapshot(REPO)["sources"]
    prefix.require(before == prefix.read(qa["source_before_ref"])["sources"], "CPU QA source identity")
    inputs, sources = binding.load_inputs(), source_refs()
    prefix.require(
        dry["complete"] is dry["passed"] is True
        and dry["inputs"] == binding.input_receipt(inputs)
        and dry["source_refs"] == sources,
        "Zero-model dry preserves rollout inputs",
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
                "scripts.simulation.runtime.benchmark_batched_joint_rollout",
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
        "solver_calls": 0,
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
            "Registered rollout benchmark workers closed and were reaped",
        )
        sample_refs = {step: prefix.reference(path) for step, path in paths.items()}
        samples = [prefix.read(sample_refs[step]) for step in sample_refs]
        prefix.require(all(sample["complete"] is sample["passed"] is True for sample in samples), "Complete samples")
        totals = {key: sum(sample["ledger"][key] for sample in samples) for key in samples[0]["ledger"]}
        half = inputs["config"]["max_rollout_calls"] // 2
        prefix.require(all(totals[key] == half for key in totals), "Exact registered rollout comparison budget")
        for path, digest in inputs["source_sha256"].items():
            prefix.checked({"path": path, "sha256": digest})
        prefix.require(
            source_refs() == sources and snapshot(REPO)["sources"] == before, "Unchanged rollout sources and inputs"
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
