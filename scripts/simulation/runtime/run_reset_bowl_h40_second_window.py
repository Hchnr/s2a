"""Own and independently replay one bounded second-reference physical window."""

import argparse
import sys
from pathlib import Path

import numpy as np

from s2a.infrastructure.native_environment import restore_cv2_environment
from s2a.infrastructure.owned_processes import run_owned_workers
from s2a.infrastructure.publication import publish_result as publish
from scripts.maintenance.run_cpu_tests import snapshot
from scripts.simulation.runtime import audit_reset_multitask_live_recovery as comparison
from scripts.simulation.runtime import registered_feedback_phase
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import reset_bowl_fixed_h40_admission as previous_admission
from scripts.simulation.runtime import run_causal_stream_recorded_physical as physical
from scripts.simulation.runtime import run_reset_start_feedback as original
from scripts.simulation.runtime import run_segmented_reset_equivalence as isolation
from scripts.wam.tracking.preview import gpu_resume_repro_20261111 as repro

ROOT = prefix.REPO
CONFIG = ROOT / "configs/tracking/references/reset_bowl_h40_second_window_20261112.json"
PLAN = ROOT / "plans/v0.0.1/tracking/references/reset_bowl_h40_second_window_20261112.md"
OUTPUT = ROOT / "outputs/reset_bowl_h40_second_window_20261112"
QA = ROOT / "outputs/qa/reset_bowl_h40_second_window_20261112/result.json"
DRY = ROOT / "outputs/reset_bowl_h40_second_window_dry_20261112/result.json"
CAPTURE = {
    "path": str(ROOT / "outputs/reset_bowl_h40_reobserve_capture_20261112/result.json"),
    "sha256": "456d3124b040537ab7572c0fa0d990c29fe77737499ea31f0eb47c26d334c161",
}
MODEL = {
    "path": str(ROOT / "outputs/reset_bowl_h40_reobserve_model_20261112/result.json"),
    "sha256": "3f0aab53e5e04d074a9b031edc05bc4bba380f72835b583a9fb79a50da1e769a",
}
FIXED_AUDIT = {
    "path": str(ROOT / "outputs/reset_bowl_fixed_h40_audit_20261112/result.json"),
    "sha256": "df5e3d2a7ff15fb432f22aad566e4b35c5ea092e1568b3552c05f52484442e8b",
}
FIXED = dict(
    schema="s2a.reset-bowl-h40-second-window.v1",
    output=str(OUTPUT.relative_to(ROOT)),
    case_id="task0_init0_s42_raw",
    seed=42,
    initial_reference_steps=48,
    second_reference_steps=8,
    max_control_steps_per_phase=56,
    max_native_steps_per_phase=1400,
    max_models_per_phase=3,
    max_contexts_per_phase=3,
    previous_control_steps=606,
    previous_native_steps=15150,
    previous_models=48,
    previous_contexts=48,
    previous_model_initializations=7,
    previous_requests_attempted=19,
    previous_requests_completed=16,
    previous_denoises=560,
    model_initializations=0,
    model_requests=0,
    fresh_cases=0,
    automatic_retry=False,
    worker_timeout_seconds=2400,
)


def source_refs():
    paths = (
        CONFIG,
        PLAN,
        Path(__file__).resolve(),
        ROOT / "scripts/simulation/runtime/reset_bowl_h40_second_window.py",
        ROOT / "src/s2a/research/native_runtime/reset_bounded_runtime.py",
        ROOT / "src/s2a/research/native_runtime/development_prefix_runtime_bounded.py",
        ROOT / "tests/research/native_runtime/test_reset_bounded_runtime.py",
        ROOT / "tests/research/native_runtime/test_reset_bowl_h40_second_window.py",
        ROOT / "tests/research/native_runtime/test_reset_bowl_h40_second_window_owner.py",
    )
    return previous_admission.source_refs() + [prefix.reference(path) for path in paths]


def load_inputs(*, verify_assets=True):
    prefix.environment()
    config = prefix.read(prefix.reference(CONFIG))
    prefix.require(
        type(config) is dict
        and set(config) == set(FIXED)
        and all(type(config[key]) is type(value) and config[key] == value for key, value in FIXED.items()),
        "The exact case and 56-period budgets are required",
    )
    base = previous_admission.load_inputs(verify_assets=verify_assets)
    capture, model, audit = (prefix.read(ref) for ref in (CAPTURE, MODEL, FIXED_AUDIT))
    prior = prefix.read(capture["inputs"]["previous_cost_audit"])
    prefix.require(
        all(
            item["complete"] is item["passed"] is True and item["control_qualified"] is False
            for item in (capture, model, audit)
        )
        and base["case"]["case"]["case_id"] == config["case_id"]
        and capture["cumulative_control_steps"] == config["previous_control_steps"]
        and capture["cumulative_native_steps"] == config["previous_native_steps"]
        and capture["cumulative_models"] == config["previous_models"]
        and capture["cumulative_contexts"] == config["previous_contexts"]
        and capture["inputs"]["producer_branch_ref"] == audit["producer_branch_ref"]
        and model["model_initializations"]
        == model["model_requests_attempted"]
        == model["model_requests_completed"]
        == 1
        and model["denoise_calls_completed"] == 35
        and model["gpu_released"] is True
        and model["physics_steps"] == 0
        and prior["cumulative_model_initializations"] + 1 == config["previous_model_initializations"]
        and prior["cumulative_requests_attempted"] + 1 == config["previous_requests_attempted"]
        and prior["cumulative_requests_completed"] + 1 == config["previous_requests_completed"]
        and prior["cumulative_denoises"] + 35 == config["previous_denoises"],
        "The audited physical trajectory, model output and cumulative costs are required",
    )
    producer_ref = audit["producer_branch_ref"]
    producer = prefix.read(producer_ref)
    reference, envelope = prefix.read(model["raw_response_ref"]), prefix.read(model["response_ref"])
    arrays = repro.response_arrays(reference)
    np.testing.assert_array_equal(arrays["action"], envelope["action"])
    np.testing.assert_array_equal(arrays["normalized_action"], envelope["normalized_action"])
    capture_image = capture["capture"]["canvas_png"]
    prefix.checked(capture_image)
    prefix.require(
        producer["complete"] is producer["passed"] is True
        and producer["runtime"]["continuation"]["fixed"]["complete"] is True
        and envelope["step"] == capture["capture"]["frame_index"] == 48
        and envelope["image_sha256"] == capture_image["sha256"]
        and envelope["state_sha256"] == capture["capture"]["current_endpoint"]["sim_state_sha256"]
        and envelope["sim_time_s"] == capture["capture"]["current_endpoint"]["sim_time"]
        and envelope["denoise_calls"] == 35,
        "The recorded second response must bind the audited endpoint and image",
    )
    return dict(
        config=config,
        base=base,
        capture=capture["capture"],
        previous_producer=producer,
        previous_producer_ref=producer_ref,
        reference=reference,
        envelope=envelope,
        reference_ref=model["raw_response_ref"],
        envelope_ref=model["response_ref"],
    )


def input_receipt(inputs):
    return dict(
        config=prefix.reference(CONFIG),
        plan=prefix.reference(PLAN),
        capture=CAPTURE,
        model=MODEL,
        fixed_audit=FIXED_AUDIT,
        previous_producer=inputs["previous_producer_ref"],
        response=inputs["reference_ref"],
        envelope=inputs["envelope_ref"],
        original=previous_admission.input_receipt(inputs["base"]),
    )


def dry_run():
    inputs = load_inputs()
    print(
        publish(
            DRY,
            dict(
                complete=True,
                passed=True,
                inputs=input_receipt(inputs),
                source_refs=source_refs(),
                native_models=0,
                physics_steps=0,
                model_initializations=0,
                model_requests=0,
                automatic_retry=False,
            ),
        ),
        flush=True,
    )


def check_qa_and_dry(inputs):
    qa, dry = prefix.read(prefix.reference(QA)), prefix.read(prefix.reference(DRY))
    before = snapshot(ROOT)["sources"]
    prefix.require(
        qa["complete"] is qa["passed"] is qa["all_tests_passed"] is qa["source_unchanged"] is True
        and qa["actual_wait"] is qa["reaped"] is True
        and qa["signal_sent"] is False
        and qa["configured_parallelism"] == 32
        and qa["skipped_count"] == qa["xfail_count"] == qa["xpass_count"] == 0
        and not qa["collection"]["collection_errors"]
        and prefix.read(qa["source_before_ref"])["sources"] == before
        and dry["complete"] is dry["passed"] is True
        and dry["inputs"] == input_receipt(inputs)
        and dry["source_refs"] == source_refs()
        and dry["native_models"] == dry["physics_steps"] == dry["model_initializations"] == dry["model_requests"] == 0,
        "A source-matched 32-worker regression and zero-physics dry are required",
    )
    return before


def worker(phase):
    directory = OUTPUT / phase / "simulation"
    result = dict(
        complete=False, passed=False, control_qualified=False, phase=phase, automatic_retry=False, fresh_cases=0
    )
    try:
        prefix.environment()
        registration = prefix.read(prefix.reference(OUTPUT / phase / "registration.json"))
        inputs = load_inputs(verify_assets=False)
        prefix.require(
            registration["phase"] == phase
            and registration["inputs"] == input_receipt(inputs)
            and registration["source_refs"] == source_refs(),
            "The worker must use the registered source and inputs",
        )
        producer = None
        if phase == "checker":
            isolation.checker_isolation()
            producer = prefix.read(registration["producer_branch_ref"])
            prefix.require(producer["complete"] is producer["passed"] is True, "A completed producer is required")
        cv2 = inputs["base"]["base"]["previous"]["base"]["cv2_admission"]
        result["cv2_admission"] = restore_cv2_environment(cv2["cv2_probe_ref"], cv2["cv2_runtime_source_refs"])
        original.require_no_simulator_import()
        from s2a.research.native_runtime.development_prefix_runtime_bounded import PrefixReplayFailure
        from s2a.research.native_runtime.reset_bounded_runtime import ResetBoundedRuntime
        from scripts.simulation.runtime import reset_bowl_h40_second_window as continuation_module

        case = inputs["base"]["case"]
        adapter = ResetBoundedRuntime(
            expected=case["expected"],
            expected_receipt=case["expected_receipt"],
            chunk=case["chunk"],
            controller=case["controller"],
            tool=case["tool"],
        )
        runtime_output = directory / "runtime"

        def continuation(env, terminal, observers, receipt, *, progress=None):
            common = dict(
                case=case,
                expected_capture=inputs["base"]["expected_capture"],
                load_reference=lambda: previous_admission.load_reference(inputs["base"]),
                runtime_output=runtime_output,
                expected_second_capture=inputs["capture"],
                load_second_reference=lambda: (inputs["reference"], inputs["envelope"]),
                progress=progress,
            )
            if phase == "producer":
                return continuation_module.run_producer(
                    env,
                    terminal,
                    observers,
                    receipt,
                    baseline=inputs["base"]["baseline"],
                    previous_producer=inputs["previous_producer"],
                    **common,
                )
            return continuation_module.run_checker(env, terminal, observers, receipt, producer=producer, **common)

        try:
            runtime = adapter.run_feedback(
                case["config"],
                case["record"],
                case["record_xml"],
                runtime_output,
                inputs["base"]["base"]["previous"]["base"]["asset_refs"],
                registered_seed=42,
                continuation=continuation,
                max_control_steps=56,
            )
        except PrefixReplayFailure as error:
            result["runtime"] = error.receipt
            raise
        unloaded = all(name not in sys.modules for name in isolation.FORBIDDEN)
        prefix.require(phase != "checker" or unloaded is True, "The checker cannot load candidate policies")
        result.update(
            complete=True, passed=True, case=case["case"], runtime=runtime, candidate_modules_unloaded=unloaded
        )
    except BaseException as error:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        print(publish(directory / "result.json", result), flush=True)


def summarize_branch(branch, phase, *, producer=None):
    """Count actual dispatches and preserve any pre-dispatch stop."""
    from scripts.simulation.runtime import reset_bowl_h40_second_window as continuation_module

    runtime = branch["runtime"]
    continuation = runtime["continuation"]
    if phase == "producer":
        first = continuation_module.periods(continuation)
        second = continuation["second"]["periods"]
        stop = continuation["second"]["stop"]
    else:
        prefix.require(
            producer is not None and branch["candidate_modules_unloaded"] is True, "Isolated checker required"
        )
        first = continuation["first_window"]["periods"] + [
            row for segment in continuation["replays"] for row in segment["periods"]
        ]
        second = continuation["second"]["periods"]
        stop = continuation["second"]["reported_producer_stop"]
    all_periods = first + second
    count = len(all_periods)
    counters = runtime["counters"]
    prefix.require(
        branch["complete"] is branch["passed"] is runtime["complete"] is runtime["passed"] is True
        and runtime["cleanup_complete"] is runtime["reset_observation_verified"] is runtime["reset_image_match"] is True
        and runtime["previous_saved_command"] is None
        and 8 <= count <= 56
        and [row["step"] for row in all_periods] == list(range(count))
        and len(first) <= 48
        and len(second) <= 8
        and (not second or len(first) == 48)
        and runtime["control_steps_attempted"] == runtime["control_steps_completed"] == count
        and counters["mj_step"]["attempted"] == counters["mj_step"]["completed"] == count * 25
        and counters["model_construct"]["completed"] == counters["context"]["completed"] == 3
        and all(len(row["actual_inner"]) == 25 for row in all_periods),
        "All actual periods, inner steps, models and contexts must match the registered limit",
    )
    if stop is not None:
        prefix.require(
            type(stop) is dict
            and type(stop.get("reason")) is str
            and bool(stop["reason"])
            and stop.get("next_action_dispatched") is False
            and (
                ("at_step" in stop and "after_step" not in stop and stop["at_step"] == count)
                or ("after_step" in stop and "at_step" not in stop and stop["after_step"] + 1 == count)
            ),
            "A stop must immediately follow the actual dispatched periods",
        )
    errors = np.asarray([row["tcp_error"] for row in all_periods], dtype=float)
    prefix.require(errors.shape == (count, 6) and np.isfinite(errors).all(), "Finite TCP errors required")
    if phase == "producer":
        prefix.require(runtime["newly_selected_commands"] == count, "The producer must count selected actions")
        receipts = runtime["segment_receipts"]
        budget = dict(
            quality_responses=sum(row["quality_refiner"]["quality_response_evaluations"] for row in receipts),
            baseline_responses=sum(row["quality_refiner"]["baseline_response_evaluations"] for row in receipts),
            gripper_sequences=sum(row["gripper_proposals"]["candidate_sequences"] for row in receipts),
            endpoint_queries=sum(row["quality_refiner"]["endpoint_queries_charged"] for row in receipts),
            transition_steps=sum(row["quality_refiner"]["transition_steps_charged"] for row in receipts),
        )
        prefix.require(
            all(type(value) is int and value >= 0 for value in budget.values())
            and budget["quality_responses"] <= 56 * 320
            and budget["baseline_responses"] <= 56 * 480
            and budget["gripper_sequences"] <= 56 * 6561
            and budget["endpoint_queries"] <= 11200
            and budget["transition_steps"] <= 145600,
            "The physical branch exceeds the original per-period candidate budgets",
        )
    else:
        budget = None
        prefix.require(runtime["newly_selected_commands"] == 0, "The checker cannot select actions")
        expected = producer["runtime"]["continuation"]
        expected_all = continuation_module.periods(expected) + expected["second"]["periods"]
        prefix.require(
            len(expected_all) == count and continuation["second"]["stop_recomputed"] is False,
            "The checker must cover exactly the producer's actions without re-solving stops",
        )
    return dict(
        control_steps=count,
        native_steps=count * 25,
        models=3,
        contexts=3,
        second_reference_steps=len(second),
        stop=stop,
        candidate_budget=budget,
        pose_caps_satisfied=sum(bool(row["pose_caps_satisfied"]) for row in all_periods),
        position_rmse_m=float(np.sqrt(np.mean(np.sum(errors[:, :3] ** 2, axis=1)))),
        rotation_rmse_rad=float(np.sqrt(np.mean(np.sum(errors[:, 3:] ** 2, axis=1)))),
        minimum_inner_margin_rad=float(min(row["minimum_inner_margin_rad"] for row in all_periods)),
        target_grasp_periods=sum(bool(row["target_grasp"]) for row in all_periods),
        task_success_periods=sum(bool(row["task_success"]) for row in all_periods),
        second_target_grasp_periods=sum(bool(row["target_grasp"]) for row in second),
        second_task_success_periods=sum(bool(row["task_success"]) for row in second),
    )


def execute(phase):
    prefix.environment()
    inputs = load_inputs()
    before = check_qa_and_dry(inputs)
    sources = source_refs()
    producer_ref = producer = prior = None
    if phase == "checker":
        prior = prefix.read(prefix.reference(OUTPUT / "producer/result.json"))
        prefix.require(prior["complete"] is prior["passed"] is prior["sources_unchanged"] is True, "Producer required")
        producer_ref = prior["branch_ref"]
        producer = prefix.read(producer_ref)
    directory = OUTPUT / phase
    directory.mkdir(parents=True, exist_ok=False)
    simulation = directory / "simulation"
    simulation.mkdir()
    registration = publish(
        directory / "registration.json",
        dict(phase=phase, inputs=input_receipt(inputs), source_refs=sources, producer_branch_ref=producer_ref),
    )
    result = dict(
        complete=False,
        passed=False,
        control_qualified=False,
        phase=phase,
        registration_ref=registration,
        model_initializations=0,
        model_requests=0,
        fresh_cases=0,
        automatic_retry=False,
    )
    try:
        environment = physical._cpu_environment(simulation)
        spec = dict(
            key=phase,
            argv=[
                sys.executable,
                "-m",
                "scripts.simulation.runtime.run_reset_bowl_h40_second_window",
                "--worker",
                phase,
            ],
            log=directory / "worker.log",
        )
        with physical._temporary_environment(environment):
            lifecycle = run_owned_workers([spec], cwd=ROOT, timeout_seconds=inputs["config"]["worker_timeout_seconds"])
        result["workers_ref"] = publish(directory / "workers.json", lifecycle)
        registered_feedback_phase.require_closed_workers(lifecycle, [phase])
        branch_ref = prefix.reference(simulation / "result.json")
        branch = prefix.read(branch_ref)
        prefix.require(branch["case"]["case_id"] == inputs["config"]["case_id"], "The physical case differs")
        counts = summarize_branch(branch, phase, producer=producer)
        if phase == "checker":
            from scripts.simulation.runtime import reset_bowl_h40_second_window as continuation_module

            expected = producer["runtime"]["continuation"]
            replayed = branch["runtime"]["continuation"]
            produced = continuation_module.periods(expected) + expected["second"]["periods"]
            checked = (
                replayed["first_window"]["periods"]
                + [row for segment in replayed["replays"] for row in segment["periods"]]
                + replayed["second"]["periods"]
            )
            result["independent_max_absolute_difference"] = comparison.compare_periods(produced, checked)
            prefix.require(counts["control_steps"] == prior["counts"]["control_steps"], "Checker count differs")
        prefix.require(
            snapshot(ROOT)["sources"] == before and source_refs() == sources, "Sources changed during execution"
        )
        result.update(complete=True, passed=True, sources_unchanged=True, branch_ref=branch_ref, counts=counts)
        if prior is not None:
            result.update(
                cumulative_control_steps=inputs["config"]["previous_control_steps"]
                + prior["counts"]["control_steps"]
                + counts["control_steps"],
                cumulative_native_steps=inputs["config"]["previous_native_steps"]
                + prior["counts"]["native_steps"]
                + counts["native_steps"],
                cumulative_models=inputs["config"]["previous_models"] + prior["counts"]["models"] + counts["models"],
                cumulative_contexts=inputs["config"]["previous_contexts"]
                + prior["counts"]["contexts"]
                + counts["contexts"],
            )
    except BaseException as error:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        branch_path = simulation / "result.json"
        if branch_path.is_file():
            result["available_branch_ref"] = prefix.reference(branch_path)
        print(publish(directory / "result.json", result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true")
    group.add_argument("--phase", choices=("producer", "checker"))
    group.add_argument("--worker", choices=("producer", "checker"))
    args = parser.parse_args()
    if args.dry_run:
        dry_run()
    elif args.phase:
        execute(args.phase)
    else:
        worker(args.worker)


if __name__ == "__main__":
    main()
