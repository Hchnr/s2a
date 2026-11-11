"""Re-run the independent reset checker with producer-matched endpoint schema."""

import argparse
import sys
from pathlib import Path

from s2a.infrastructure.native_environment import restore_cv2_environment
from s2a.infrastructure.owned_processes import run_owned_workers
from s2a.infrastructure.publication import publish_result as publish
from scripts.maintenance.run_cpu_tests import snapshot
from scripts.simulation.runtime import registered_feedback_phase
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import reset_live_second_admission as original_inputs
from scripts.simulation.runtime import reset_live_two_window_replay as original_replay
from scripts.simulation.runtime import run_causal_stream_recorded_physical as physical
from scripts.simulation.runtime import run_reset_live_second_window as original_owner
from scripts.simulation.runtime import run_reset_start_feedback as original
from scripts.simulation.runtime import run_segmented_reset_equivalence as isolation
from scripts.verification.response_models.audit_reset_start_feedback import compare_replay

ROOT = prefix.REPO
CONFIG = ROOT / "configs/tracking/references/reset_live_checker_repair_20261111.json"
PLAN = ROOT / "plans/v0.0.1/tracking/references/reset_live_checker_repair_20261111.md"
QA = ROOT / "outputs/qa/reset_live_checker_repair_20261111/result.json"
DRY = ROOT / "outputs/reset_live_checker_repair_dry_20261111/result.json"
OUTPUT = ROOT / "outputs/reset_live_checker_repair_20261111"
MODULE = "scripts.simulation.runtime.run_reset_live_checker_repair"
PRODUCER = {
    "path": str(ROOT / "outputs/reset_live_second_window_20261111/producer/result.json"),
    "sha256": "a4634066e9a4ed8c7b93eeda89501b8f66b0454b0fe854c5976cf46b2c104d0f",
}
FAILURE = {
    "path": str(ROOT / "outputs/reset_live_second_window_20261111/checker/result.json"),
    "sha256": "bbdcb22927763f9a7cb6bc460155fe70adc073cc7c3da1ed17017e9fad32c93a",
}
PARTIAL = {
    "path": str(ROOT / "outputs/reset_live_second_window_20261111/checker/simulation/result.json"),
    "sha256": "554e49aa1cdd870b8b2e51f8cddfe23dae54885f3db55d51ad54c7df4527a850",
}
FIXED = dict(
    schema="s2a.reset-live-checker-repair.v1",
    output=str(OUTPUT.relative_to(ROOT)),
    case_index=8,
    producer_sha256=PRODUCER["sha256"],
    failed_checker_sha256=FAILURE["sha256"],
    failed_branch_sha256=PARTIAL["sha256"],
    max_control_steps=16,
    max_native_steps=400,
    max_models=3,
    max_contexts=3,
    world_model_calls=0,
    fresh_cases=0,
    worker_timeout_seconds=1800,
    automatic_retry=False,
)


def validate_config(config):
    prefix.require(
        type(config) is dict
        and set(config) == set(FIXED)
        and all(type(config[k]) is type(v) and config[k] == v for k, v in FIXED.items()),
        "The exact repair roster and budget are required",
    )


def source_refs():
    names = (
        CONFIG,
        PLAN,
        Path(__file__).resolve(),
        ROOT / "scripts/simulation/runtime/reset_live_two_window_replay_v2.py",
        ROOT / "tests/research/native_runtime/test_reset_live_two_window_replay_v2.py",
        ROOT / "tests/research/native_runtime/test_reset_live_checker_repair.py",
    )
    return original_inputs.source_refs() + [prefix.reference(path) for path in names]


def load_inputs(*, verify_assets=True):
    config_ref = prefix.reference(CONFIG)
    config = prefix.read(config_ref)
    validate_config(config)
    inputs = original_inputs.load_inputs(verify_assets=verify_assets)
    producer, failure, partial = (prefix.read(ref) for ref in (PRODUCER, FAILURE, PARTIAL))
    lifecycle = prefix.read(failure["workers_ref"])
    prefix.require(
        producer["complete"] is producer["passed"] is producer["sources_unchanged"] is producer["gpu_released"] is True
        and producer["model_initializations"] == producer["model_requests_attempted"] == 1
        and producer["denoise_calls_attempted"] == 35
        and producer["counts"]["control_steps"] == 16
        and producer["counts"]["native_steps"] == 400
        and failure["complete"] is failure["passed"] is False
        and failure["available_branch_ref"] == PARTIAL
        and len(lifecycle["workers"]) == 1
        and lifecycle["workers"][0]["returncode"] == 1
        and lifecycle["workers"][0]["actual_wait"] is lifecycle["workers"][0]["reaped"] is True
        and lifecycle["workers"][0]["signal_sent"] is False
        and partial["complete"] is partial["passed"] is False
        and partial["error"]["message"] == "The independent current image or endpoint differs"
        and partial["runtime"]["cleanup_complete"] is True
        and partial["runtime"]["control_steps_completed"] == 8
        and partial["runtime"]["counters"]["mj_step"]["completed"] == 200,
        "The exact completed producer and retained failed checker are required",
    )
    branch, model = prefix.read(producer["branch_ref"]), prefix.read(producer["model_ref"])
    prefix.require(
        branch["complete"] is branch["passed"] is model["complete"] is model["passed"] is True
        and model["model_requests_completed"] == model["responses_sent"] == 1
        and model["denoise_calls_completed"] == 35,
        "The actual one-request model and producer must be closed",
    )
    old_periods = branch["runtime"]["continuation"]["first_window"]["periods"]
    partial_periods = partial["runtime"]["reset_first_window"]["periods"]
    difference = compare_replay(old_periods, partial_periods)
    prefix.require(
        all(value == 0 for value in difference.values()),
        "The failed checker must have exactly reproduced its first eight periods",
    )
    expected = branch["runtime"]["streaming_captures"][0]
    actual = partial["runtime"]["checker_stream_capture"]
    actual_minimal = dict(
        actual,
        current_endpoint={
            key: actual["current_endpoint"][key] for key in ("endpoint_step", "sim_time", "sim_state_sha256")
        },
    )
    original_replay.compare_capture(expected, actual_minimal)
    prefix.require(
        len(actual["current_endpoint"]) > len(actual_minimal["current_endpoint"]),
        "The original failure must be caused by extra checker endpoint metadata",
    )
    consumed = dict(inputs["source_sha256"])
    for ref in (
        config_ref,
        PRODUCER,
        FAILURE,
        PARTIAL,
        failure["workers_ref"],
        producer["branch_ref"],
        producer["model_ref"],
    ):
        prefix.checked(ref, consumed)
    return dict(
        original=inputs,
        config=config,
        config_ref=config_ref,
        producer=producer,
        failed=failure,
        partial=partial,
        branch=branch,
        model=model,
        source_sha256=consumed,
    )


def input_receipt(inputs):
    return dict(
        config_ref=inputs["config_ref"],
        original=original_inputs.input_receipt(inputs["original"]),
        producer_ref=PRODUCER,
        failure_ref=FAILURE,
        partial_ref=PARTIAL,
        source_sha256=inputs["source_sha256"],
    )


def dry_run():
    prefix.environment()
    inputs = load_inputs()
    print(
        publish(
            DRY,
            dict(
                complete=True,
                passed=True,
                inputs=input_receipt(inputs),
                source_refs=source_refs(),
                models=0,
                physics_steps=0,
                world_model_calls=0,
                fresh_cases=0,
                automatic_retry=False,
            ),
        ),
        flush=True,
    )


def check_qa_and_dry(inputs):
    qa, dry = prefix.read(prefix.reference(QA)), prefix.read(prefix.reference(DRY))
    prefix.require(
        qa["complete"] is qa["passed"] is qa["all_tests_passed"] is qa["source_unchanged"] is True
        and qa["actual_wait"] is qa["reaped"] is True
        and qa["signal_sent"] is False
        and qa["configured_parallelism"] == 32
        and qa["skipped_count"] == qa["xfail_count"] == qa["xpass_count"] == 0
        and not qa["collection"]["collection_errors"],
        "Complete source-matched repair QA is required",
    )
    before = snapshot(ROOT)["sources"]
    prefix.require(
        prefix.read(qa["source_before_ref"])["sources"] == before
        and dry["complete"] is dry["passed"] is True
        and dry["inputs"] == input_receipt(inputs)
        and dry["source_refs"] == source_refs(),
        "The repair dry and source identity must match",
    )
    return before


def worker():
    prefix.environment()
    registration = prefix.read(prefix.reference(OUTPUT / "registration.json"))
    inputs = load_inputs(verify_assets=False)
    prefix.require(
        registration["inputs"] == input_receipt(inputs) and registration["source_refs"] == source_refs(),
        "Frozen repair inputs required",
    )
    isolation.checker_isolation()
    directory = OUTPUT / "simulation"
    result = dict(
        complete=False, passed=False, control_qualified=False, phase="checker", automatic_retry=False, fresh_cases=0
    )
    try:
        old, case, producer, model = inputs["original"], inputs["original"]["case"], inputs["branch"], inputs["model"]
        cv2 = old["base"]["cv2_admission"]
        result["cv2_admission"] = restore_cv2_environment(cv2["cv2_probe_ref"], cv2["cv2_runtime_source_refs"])
        original.require_no_simulator_import()
        from s2a.research.native_runtime.development_prefix_runtime import PrefixReplayFailure
        from s2a.research.native_runtime.reset_two_window_runtime import ResetTwoWindowRuntime
        from scripts.simulation.runtime.reset_live_two_window_replay_v2 import run_checker

        adapter = ResetTwoWindowRuntime(
            expected=case["expected"],
            expected_receipt=case["expected_receipt"],
            chunk=case["chunk"],
            controller=case["controller"],
            tool=case["tool"],
        )
        model_response = prefix.read(model["responses"][0]["actual_response_ref"])
        model_registration = original_inputs.model_registration(
            old, original_inputs.OUTPUT / "producer/simulation/runtime"
        )
        runtime_output = directory / "runtime"

        def continuation(env, terminal, observers, receipt, *, progress=None):
            return run_checker(
                env,
                terminal,
                observers,
                receipt,
                case=case,
                producer=producer,
                registration=model_registration,
                model_response=model_response,
                runtime_output=runtime_output,
                progress=progress,
            )

        try:
            runtime = adapter.run_feedback(
                case["config"],
                case["record"],
                case["record_xml"],
                runtime_output,
                old["base"]["asset_refs"],
                registered_seed=case["case"]["sampling_seed"],
                continuation=continuation,
            )
        except PrefixReplayFailure as error:
            result["runtime"] = error.receipt
            raise
        unloaded = all(name not in sys.modules for name in isolation.FORBIDDEN)
        prefix.require(unloaded is True, "The checker may not load candidate policies")
        result.update(
            complete=True, passed=True, case=case["case"], runtime=runtime, candidate_modules_unloaded=unloaded
        )
    except BaseException as error:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        print(publish(directory / "result.json", result), flush=True)


def execute():
    prefix.environment()
    inputs = load_inputs()
    before = check_qa_and_dry(inputs)
    sources = source_refs()
    OUTPUT.mkdir(exist_ok=False)
    directory = OUTPUT / "simulation"
    directory.mkdir()
    registration = publish(OUTPUT / "registration.json", dict(inputs=input_receipt(inputs), source_refs=sources))
    result = dict(
        complete=False,
        passed=False,
        control_qualified=False,
        phase="checker",
        registration_ref=registration,
        world_model_calls=0,
        fresh_cases=0,
        automatic_retry=False,
    )
    try:
        env = physical._cpu_environment(directory)
        spec = dict(key="checker", argv=[sys.executable, "-m", MODULE, "--worker"], log=OUTPUT / "checker.log")
        with physical._temporary_environment(env):
            lifecycle = run_owned_workers([spec], cwd=ROOT, timeout_seconds=inputs["config"]["worker_timeout_seconds"])
        result["workers_ref"] = publish(OUTPUT / "workers.json", lifecycle)
        registered_feedback_phase.require_closed_workers(lifecycle, ["checker"])
        branch_ref = prefix.reference(directory / "result.json")
        branch = prefix.read(branch_ref)
        counts = original_owner.summarize_branch(branch, "checker", producer=inputs["branch"])
        prefix.require(
            counts["control_steps"] == 16
            and counts["native_steps"] == 400
            and counts["models"] == counts["contexts"] == 3,
            "The complete registered independent replay is required",
        )
        for path, digest in inputs["source_sha256"].items():
            prefix.checked({"path": path, "sha256": digest})
        prefix.require(source_refs() == sources and snapshot(ROOT)["sources"] == before, "Repair sources changed")
        result.update(
            complete=True,
            passed=True,
            sources_unchanged=True,
            branch_ref=branch_ref,
            counts=counts,
            original_failure_preserved=True,
        )
    finally:
        if (directory / "result.json").is_file():
            result["available_branch_ref"] = prefix.reference(directory / "result.json")
        print(publish(OUTPUT / "result.json", result), flush=True)


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
