"""Run a separately budgeted, source-pinned live-feedback comparison repair."""

import argparse
import sys
from pathlib import Path

from s2a.infrastructure.owned_processes import run_owned_workers
from s2a.infrastructure.publication import publish_result as publish
from scripts.maintenance.run_cpu_tests import snapshot
from scripts.simulation.runtime import prepare_early_tabletop_live_feedback_repair as admission
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_causal_stream_recorded_physical as physical
from scripts.simulation.runtime import run_early_tabletop_live_feedback as previous
from scripts.simulation.runtime import run_quality_rate_repair_recorded_feedback as clone_support

ROOT, OUTPUT, DRY, QA = admission.ROOT, admission.OUTPUT, admission.DRY, admission.QA
MODULE = "scripts.simulation.runtime.run_early_tabletop_live_feedback_repair"


def source_refs():
    return previous.source_refs() + [
        prefix.reference(path)
        for path in (
            admission.CONFIG,
            admission.PLAN,
            Path(__file__).resolve(),
            ROOT / "scripts/simulation/runtime/prepare_early_tabletop_live_feedback_repair.py",
            ROOT / "tests/research/native_runtime/test_early_tabletop_live_feedback_repair_admission.py",
            ROOT / "tests/research/native_runtime/test_early_tabletop_live_feedback_repair.py",
        )
    ]


def worker(mode, phase):
    run = clone_support.clone(previous.worker, {"OUTPUT": OUTPUT, "admission": admission, "source_refs": source_refs})
    return run(mode, phase)


def cumulative_cost(native_steps, new_commands, config):
    """Count the earlier capture and failed attempt in the registered total."""
    if type(native_steps) is not int or type(new_commands) is not int or native_steps < 0 or new_commands < 0:
        raise ValueError("Nonnegative actual counts are required")
    total = config["prior_capture_native_steps"] + config["prior_failed_native_steps"] + native_steps
    commands = 4 + new_commands
    prefix.require(
        native_steps <= config["repair_max_native_steps"]
        and total <= config["total_max_native_steps"]
        and commands <= config["total_max_new_commands"],
        "The separate repair and cumulative budgets must hold",
    )
    return {
        "repair_native_steps": native_steps,
        "experiment_native_steps": total,
        "repair_new_commands": new_commands,
        "experiment_new_commands": commands,
    }


def dry_run():
    inputs = admission.load_inputs()
    print(
        publish(
            DRY,
            dict(
                complete=True,
                passed=True,
                inputs=admission.input_receipt(inputs),
                source_refs=source_refs(),
                native_models=0,
                physics_steps=0,
                world_model_calls=0,
                fresh_cases=0,
            ),
        ),
        flush=True,
    )


def execute():
    inputs = admission.load_inputs()
    item, base = inputs["early_tabletop_live_feedback_repair"], inputs["early_tabletop_live_feedback"]
    sources, before = source_refs(), snapshot(ROOT)["sources"]
    qa, dry = prefix.read(prefix.reference(QA)), prefix.read(prefix.reference(DRY))
    prefix.require(
        qa["complete"] is qa["passed"] is qa["all_tests_passed"] is qa["source_unchanged"] is True
        and qa["actual_wait"] is qa["reaped"] is True
        and qa["signal_sent"] is False
        and qa["configured_parallelism"] == 32
        and qa["skipped_count"] == qa["xfail_count"] == qa["xpass_count"] == 0
        and not qa["collection"]["collection_errors"]
        and prefix.read(qa["source_before_ref"])["sources"] == before
        and dry["complete"] is dry["passed"] is True
        and dry["inputs"] == admission.input_receipt(inputs)
        and dry["source_refs"] == sources
        and dry["native_models"] == dry["physics_steps"] == dry["world_model_calls"] == 0,
        "Source-matched 32-worker QA and zero-work dry are required",
    )
    OUTPUT.mkdir(parents=True, exist_ok=False)
    result = dict(
        complete=False,
        passed=False,
        control_qualified=False,
        automatic_retry=False,
        new_world_model_calls=0,
        fresh_cases=0,
        groups=[],
        prior_capture_native_steps=12000,
        prior_failed_native_steps=6000,
        prior_model_requests=2,
        prior_denoise_calls=70,
    )
    try:
        env = physical._cpu_environment(OUTPUT)
        env["MUJOCO_GL"] = "osmesa"

        def register(mode, phase, producer_ref):
            directory = OUTPUT / mode / phase
            (directory / "simulation").mkdir(parents=True)
            publish(
                directory / "registration.json",
                dict(
                    mode=mode,
                    phase=phase,
                    inputs=admission.input_receipt(inputs),
                    source_refs=sources,
                    producer_ref=producer_ref,
                ),
            )
            return dict(
                key=f"{mode}-{phase}",
                argv=[sys.executable, "-m", MODULE, "--worker", mode, phase],
                log=directory / "worker.log",
            )

        def group(name, specs):
            with physical._temporary_environment(env):
                lifecycle = run_owned_workers(specs, cwd=ROOT, timeout_seconds=item["config"]["worker_timeout_seconds"])
            result["groups"].append({"name": name, "workers_ref": publish(OUTPUT / f"{name}_workers.json", lifecycle)})
            prefix.require(
                lifecycle["launch_error"] is None
                and len(lifecycle["workers"]) == len(specs)
                and all(
                    row["actual_wait"] is row["reaped"] is True
                    and row["returncode"] == 0
                    and row["signal_sent"] is False
                    for row in lifecycle["workers"]
                ),
                f"Every {name} worker must exit and be reaped",
            )

        group("producers", [register(mode, "producer", None) for mode in item["config"]["modes"]])
        refs, producers, summaries = {}, {}, {}
        for mode in item["config"]["modes"]:
            refs[mode] = prefix.reference(OUTPUT / mode / "producer/simulation/result.json")
            producers[mode] = prefix.read(refs[mode])
            summaries[mode] = {
                "producer": previous.summarize(producers[mode], mode=mode, phase="producer", config=base["config"])
            }
        group("checkers", [register(mode, "checker", refs[mode]) for mode in item["config"]["modes"]])
        checker_refs = {}
        for mode in item["config"]["modes"]:
            checker_refs[mode] = prefix.reference(OUTPUT / mode / "checker/simulation/result.json")
            checker = prefix.read(checker_refs[mode])
            summaries[mode]["checker"] = previous.summarize(checker, mode=mode, phase="checker", config=base["config"])
            previous.check_pair(producers[mode], checker)
        total = sum(
            summary[phase]["native_steps"] for summary in summaries.values() for phase in ("producer", "checker")
        )
        new = sum(summary["producer"]["newly_selected_commands"] for summary in summaries.values())
        costs = cumulative_cost(total, new, item["config"])
        prefix.require(
            source_refs() == sources and snapshot(ROOT)["sources"] == before, "Repair source identity changed"
        )
        for path, digest in item["source_sha256"].items():
            prefix.checked({"path": path, "sha256": digest})
        result.update(
            complete=True,
            passed=True,
            sources_unchanged=True,
            producer_refs=refs,
            checker_refs=checker_refs,
            summaries=summaries,
            **costs,
        )
    except BaseException as error:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        partial = []
        for mode in item["config"]["modes"]:
            for phase in ("producer", "checker"):
                path = OUTPUT / mode / phase / "simulation/result.json"
                if path.is_file():
                    ref = prefix.reference(path)
                    branch = prefix.read(ref)
                    runtime = branch.get("runtime", {})
                    partial.append(
                        {
                            "mode": mode,
                            "phase": phase,
                            "ref": ref,
                            "native_steps_completed": runtime.get("counters", {}).get("mj_step", {}).get("completed"),
                            "newly_selected_commands": runtime.get("newly_selected_commands"),
                        }
                    )
        result["partial_branches"] = partial
        raise
    finally:
        print(publish(OUTPUT / "result.json", result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true")
    group.add_argument("--execute", action="store_true")
    group.add_argument("--worker", nargs=2, metavar=("MODE", "PHASE"))
    args = parser.parse_args()
    if args.dry_run:
        dry_run()
    elif args.execute:
        execute()
    else:
        worker(*args.worker)


if __name__ == "__main__":
    main()
