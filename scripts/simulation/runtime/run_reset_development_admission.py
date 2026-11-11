"""Verify twelve actual CPU resets and first images without dispatching actions."""

import argparse
import os
import sys
import tempfile
from pathlib import Path

from s2a.infrastructure.native_environment import restore_cv2_environment
from s2a.infrastructure.owned_processes import run_owned_workers
from s2a.infrastructure.publication import publish_result as publish
from scripts.maintenance.run_cpu_tests import snapshot
from scripts.simulation.runtime import prepare_reset_development as preparation
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_causal_stream_recorded_physical as physical

ROOT = prefix.REPO
CONFIG = ROOT / "configs/tracking/joint_control/reset_native_admission_20261111.json"
PLAN = preparation.PLAN
OUTPUT = ROOT / "outputs/reset_native_admission_20261111"
DRY = ROOT / "outputs/reset_native_admission_dry_20261111/result.json"
QA = ROOT / "outputs/qa/reset_native_admission_20261111/result.json"
MODULE = "scripts.simulation.runtime.run_reset_development_admission"


def validate_config(config):
    integer_keys = (
        "max_workers",
        "max_models",
        "max_contexts",
        "max_control_steps",
        "max_native_steps",
        "world_model_calls",
        "fresh_cases",
        "worker_timeout_seconds",
    )
    if any(type(config.get(key)) is not int for key in integer_keys) or config.get("automatic_retry") is not False:
        raise ValueError("Reset budgets must use explicit integers and forbid retries")
    if config != {
        "schema": "s2a.reset-native-admission.v1",
        "case_indices": list(range(12)),
        "task_ids": [8, 9, 6],
        "sampling_seeds": [42, 43],
        "variants": ["raw", "terminal058"],
        "max_workers": 12,
        "max_models": 36,
        "max_contexts": 36,
        "max_control_steps": 0,
        "max_native_steps": 0,
        "world_model_calls": 0,
        "fresh_cases": 0,
        "worker_timeout_seconds": 1200,
        "automatic_retry": False,
    }:
        raise ValueError("The exact twelve-case reset-only budget is required")


def source_refs():
    return [
        prefix.reference(path)
        for path in (
            CONFIG,
            PLAN,
            Path(__file__).resolve(),
            Path(preparation.__file__).resolve(),
            ROOT / "src/s2a/research/native_runtime/development_reset_runtime.py",
            ROOT / "src/s2a/research/native_runtime/development_prefix_runtime.py",
            ROOT / "src/s2a/research/native_runtime/development_prefix_replay.py",
            ROOT / "src/s2a/research/native_runtime/reset_endpoint.py",
            ROOT / "src/s2a/research/wam/training/native_observations.py",
            ROOT / "src/s2a/backends/libero/cameras.py",
            ROOT / "src/s2a/backends/libero/simulation.py",
            ROOT / "tests/research/native_runtime/test_reset_development_admission.py",
            ROOT / "tests/research/native_runtime/test_development_reset_runtime.py",
            ROOT / "tests/research/native_runtime/test_prepare_reset_development.py",
            ROOT / "tests/research/native_runtime/test_reset_endpoint.py",
        )
    ]


def load_config():
    ref = prefix.reference(CONFIG)
    config = prefix.read(ref)
    validate_config(config)
    return ref, config


def require_no_simulator_import():
    modules = ("mujoco", "OpenGL", "robosuite", "libero")
    prefix.require(
        not any(name in sys.modules for name in modules), "Simulator must not load before OSMesa configuration"
    )


def case_receipt(inputs, index):
    receipt = preparation.input_receipt(inputs)
    selected = [item for item in receipt["cases"] if item["case"]["case_index"] == index]
    prefix.require(len(selected) == 1, "One original case per worker")
    return dict(receipt, cases=selected)


def dry_run():
    prefix.environment()
    require_no_simulator_import()
    config_ref, config = load_config()
    inputs = preparation.load_inputs(verify_assets=True)
    require_no_simulator_import()
    prefix.require([c["case"]["case_index"] for c in inputs["cases"]] == config["case_indices"], "Fixed case roster")
    print(
        publish(
            DRY,
            dict(
                complete=True,
                passed=True,
                config_ref=config_ref,
                inputs=preparation.input_receipt(inputs),
                source_refs=source_refs(),
                models=0,
                contexts=0,
                control_steps=0,
                physics_steps=0,
                world_model_calls=0,
                fresh_cases=0,
                automatic_retry=False,
            ),
        ),
        flush=True,
    )


def worker(index):
    directory = OUTPUT / f"case_{index:02d}"
    result = dict(
        complete=False,
        passed=False,
        control_qualified=False,
        case_index=index,
        automatic_retry=False,
        world_model_calls=0,
        fresh_cases=0,
    )
    try:
        temp, cache = directory / "tmp", directory / "cache"
        temp.mkdir()
        cache.mkdir()
        os.environ.update(TMPDIR=str(temp), XDG_CACHE_HOME=str(cache), NUMBA_CACHE_DIR=str(cache / "numba"))
        tempfile.tempdir = None
        prefix.environment()
        require_no_simulator_import()
        reg = prefix.read(prefix.reference(directory / "registration.json"))
        inputs = preparation.load_inputs(index, verify_assets=False)
        prefix.require(
            reg["inputs"] == case_receipt(inputs, index) and reg["source_refs"] == source_refs(),
            "The worker must bind its registered inputs and sources",
        )
        require_no_simulator_import()
        cv2 = inputs["cv2_admission"]
        result["cv2_admission"] = restore_cv2_environment(cv2["cv2_probe_ref"], cv2["cv2_runtime_source_refs"])
        require_no_simulator_import()
        from s2a.research.native_runtime.development_prefix_runtime import PrefixReplayFailure
        from s2a.research.native_runtime.development_reset_runtime import DevelopmentResetRuntime

        case = inputs["cases"][0]
        adapter = DevelopmentResetRuntime(
            expected=case["expected"],
            expected_receipt=case["expected_receipt"],
            chunk=case["chunk"],
            controller=case["controller"],
            tool=case["tool"],
        )
        try:
            runtime = adapter.run(
                case["config"],
                case["record"],
                case["record_xml"],
                directory / "runtime",
                inputs["asset_refs"],
                registered_seed=case["case"]["sampling_seed"],
            )
        except PrefixReplayFailure as error:
            result["runtime"] = error.receipt
            raise
        result.update(complete=True, passed=True, case=case["case"], runtime=runtime)
    except BaseException as error:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        print(publish(directory / "result.json", result), flush=True)


def execute():
    prefix.environment()
    require_no_simulator_import()
    config_ref, config = load_config()
    before, sources = snapshot(ROOT)["sources"], source_refs()
    qa, dry = prefix.read(prefix.reference(QA)), prefix.read(prefix.reference(DRY))
    inputs = preparation.load_inputs(verify_assets=True)
    prefix.require(
        qa["complete"] is qa["passed"] is qa["all_tests_passed"] is qa["source_unchanged"] is True
        and qa["actual_wait"] is qa["reaped"] is True
        and qa["signal_sent"] is False
        and qa["configured_parallelism"] == 32
        and qa["skipped_count"] == 0
        and not qa["collection"]["collection_errors"]
        and prefix.read(qa["source_before_ref"])["sources"] == before
        and dry["complete"] is dry["passed"] is True
        and dry["config_ref"] == config_ref
        and dry["source_refs"] == sources
        and dry["inputs"] == preparation.input_receipt(inputs)
        and dry["models"] == dry["contexts"] == dry["physics_steps"] == 0,
        "Source-matched CPU QA and zero-model dry are required",
    )
    OUTPUT.mkdir(exist_ok=False)
    result = dict(
        complete=False,
        passed=False,
        control_qualified=False,
        config_ref=config_ref,
        inputs=preparation.input_receipt(inputs),
        source_refs=sources,
        qa_ref=prefix.reference(QA),
        dry_ref=prefix.reference(DRY),
        automatic_retry=False,
        world_model_calls=0,
        fresh_cases=0,
    )
    try:
        specs = []
        for index in config["case_indices"]:
            directory = OUTPUT / f"case_{index:02d}"
            directory.mkdir()
            publish(directory / "registration.json", dict(inputs=case_receipt(inputs, index), source_refs=sources))
            specs.append(
                dict(
                    key=f"case_{index:02d}",
                    argv=[sys.executable, "-m", MODULE, "--worker", str(index)],
                    log=directory / "worker.log",
                )
            )
        env = physical._cpu_environment(OUTPUT)
        with physical._temporary_environment(env):
            lifecycle = run_owned_workers(specs, cwd=ROOT, timeout_seconds=config["worker_timeout_seconds"])
        result["workers_ref"] = publish(OUTPUT / "workers.json", lifecycle)
        prefix.require(
            lifecycle["launch_error"] is None
            and len(lifecycle["workers"]) == 12
            and all(
                row["actual_wait"] is row["reaped"] is True and row["returncode"] == 0 and row["signal_sent"] is False
                for row in lifecycle["workers"]
            ),
            "All reset workers must exit and be reaped",
        )
        refs = [prefix.reference(OUTPUT / f"case_{index:02d}/result.json") for index in config["case_indices"]]
        cases = [prefix.read(ref) for ref in refs]
        prefix.require(
            all(
                c["complete"] is c["passed"] is True
                and c["case_index"] == i
                and c["runtime"]["reset_observation_verified"] is True
                and c["runtime"]["reset_image_match"] is True
                and c["runtime"]["control_steps_completed"] == 0
                and c["runtime"]["counters"]["mj_step"]["attempted"] == 0
                and c["runtime"]["counters"]["model_construct"]["completed"] == 3
                and c["runtime"]["counters"]["context"]["completed"] == 3
                for i, c in enumerate(cases)
            ),
            "Every reset, image and zero-step budget must match",
        )
        prefix.require(
            snapshot(ROOT)["sources"] == before
            and source_refs() == sources
            and preparation.input_receipt(preparation.load_inputs(verify_assets=True))
            == preparation.input_receipt(inputs),
            "Sources or reviewed inputs changed during reset admission",
        )
        result.update(
            complete=True,
            passed=True,
            sources_unchanged=True,
            case_refs=refs,
            totals=dict(models=36, contexts=36, control_steps=0, physics_steps=0),
            image_matches=12,
            exact_state_matches=sum(c["runtime"]["reset_endpoint_comparison"]["state_bytes_equal"] for c in cases),
        )
    except BaseException as error:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        partial = []
        for index in config["case_indices"]:
            path = OUTPUT / f"case_{index:02d}/result.json"
            if path.is_file():
                ref = prefix.reference(path)
                branch = prefix.read(ref)
                partial.append(dict(case_index=index, ref=ref, counters=branch.get("runtime", {}).get("counters")))
        result["partial_cases"] = partial
        raise
    finally:
        print(publish(OUTPUT / "result.json", result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true")
    group.add_argument("--execute", action="store_true")
    group.add_argument("--worker", type=int, choices=range(12))
    args = parser.parse_args()
    if args.dry_run:
        dry_run()
    elif args.execute:
        execute()
    else:
        worker(args.worker)


if __name__ == "__main__":
    main()
