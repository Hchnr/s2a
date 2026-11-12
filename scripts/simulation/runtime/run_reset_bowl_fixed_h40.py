"""Own one fixed-reference bowl probe and its independent CPU replay."""

import argparse
import sys

import numpy as np

from s2a.infrastructure.native_environment import restore_cv2_environment
from s2a.infrastructure.owned_processes import run_owned_workers
from s2a.infrastructure.publication import publish_result as publish
from scripts.maintenance.run_cpu_tests import snapshot
from scripts.simulation.runtime import audit_reset_multitask_live_recovery as comparison
from scripts.simulation.runtime import registered_feedback_phase
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import reset_bowl_fixed_h40_admission as admission
from scripts.simulation.runtime import run_causal_stream_recorded_physical as physical
from scripts.simulation.runtime import run_reset_start_feedback as original
from scripts.simulation.runtime import run_segmented_reset_equivalence as isolation

ROOT, OUTPUT, QA, DRY = admission.ROOT, admission.OUTPUT, admission.QA, admission.DRY
MODULE = "scripts.simulation.runtime.run_reset_bowl_fixed_h40"


def dry_run():
    inputs = admission.load_inputs()
    result = dict(
        complete=True,
        passed=True,
        inputs=admission.input_receipt(inputs),
        source_refs=admission.source_refs(),
        native_models=0,
        physics_steps=0,
        model_initializations=0,
        model_requests=0,
        fresh_cases=0,
        automatic_retry=False,
    )
    print(publish(DRY, result), flush=True)


def check_qa_and_dry(inputs):
    qa, dry = prefix.read(prefix.reference(QA)), prefix.read(prefix.reference(DRY))
    prefix.require(
        qa["complete"] is qa["passed"] is qa["all_tests_passed"] is qa["source_unchanged"] is True
        and qa["actual_wait"] is qa["reaped"] is True
        and qa["signal_sent"] is False
        and qa["configured_parallelism"] == 32
        and qa["skipped_count"] == qa["xfail_count"] == qa["xpass_count"] == 0
        and not qa["collection"]["collection_errors"],
        "A complete 32-worker source-matched regression is required",
    )
    before = snapshot(ROOT)["sources"]
    prefix.require(before == prefix.read(qa["source_before_ref"])["sources"], "The QA source snapshot differs")
    prefix.require(
        dry["complete"] is dry["passed"] is True
        and dry["inputs"] == admission.input_receipt(inputs)
        and dry["source_refs"] == admission.source_refs()
        and dry["native_models"] == dry["physics_steps"] == dry["model_initializations"] == dry["model_requests"] == 0,
        "A matching zero-physics, zero-model dry is required",
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
        inputs = admission.load_inputs(verify_assets=False)
        prefix.require(
            registration["phase"] == phase
            and registration["inputs"] == admission.input_receipt(inputs)
            and registration["source_refs"] == admission.source_refs(),
            "The worker inputs and source identities must match registration",
        )
        producer = None
        if phase == "checker":
            isolation.checker_isolation()
            producer = prefix.read(registration["producer_branch_ref"])
            prefix.require(
                producer["complete"] is producer["passed"] is True, "A closed fixed-reference producer is required"
            )
        cv2 = inputs["base"]["previous"]["base"]["cv2_admission"]
        result["cv2_admission"] = restore_cv2_environment(cv2["cv2_probe_ref"], cv2["cv2_runtime_source_refs"])
        original.require_no_simulator_import()
        from s2a.research.native_runtime.development_prefix_runtime_48 import PrefixReplayFailure
        from s2a.research.native_runtime.reset_multi_window_runtime import ResetMultiWindowRuntime

        case = inputs["case"]
        adapter = ResetMultiWindowRuntime(
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
                expected_capture=inputs["expected_capture"],
                load_reference=lambda: admission.load_reference(inputs),
                runtime_output=runtime_output,
                progress=progress,
            )
            if phase == "producer":
                from scripts.simulation.runtime.reset_bowl_fixed_h40 import run_producer

                return run_producer(env, terminal, observers, receipt, baseline=inputs["baseline"], **common)
            from scripts.simulation.runtime.reset_bowl_fixed_h40_replay import run_checker

            return run_checker(env, terminal, observers, receipt, producer=producer, **common)

        try:
            runtime = adapter.run_feedback(
                case["config"],
                case["record"],
                case["record_xml"],
                runtime_output,
                inputs["base"]["previous"]["base"]["asset_refs"],
                registered_seed=42,
                continuation=continuation,
            )
        except PrefixReplayFailure as error:
            result["runtime"] = error.receipt
            raise
        unloaded = all(name not in sys.modules for name in isolation.FORBIDDEN)
        prefix.require(phase != "checker" or unloaded is True, "The checker must not load candidate strategy modules")
        result.update(
            complete=True, passed=True, case=case["case"], runtime=runtime, candidate_modules_unloaded=unloaded
        )
    except BaseException as error:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        print(publish(directory / "result.json", result), flush=True)


def summarize_branch(branch, phase, *, producer=None):
    runtime, continuation = branch["runtime"], branch["runtime"]["continuation"]
    first = continuation["first_window"]["periods"]
    prefix.require(
        phase in ("producer", "checker")
        and branch["complete"] is branch["passed"] is runtime["complete"] is runtime["passed"] is True
        and runtime["cleanup_complete"] is runtime["reset_observation_verified"] is runtime["reset_image_match"] is True
        and runtime["previous_saved_command"] is None
        and len(first) == 8
        and [row["step"] for row in first] == list(range(8)),
        "The verified reset and eight initial physical periods are required",
    )
    budget = None
    if phase == "producer":
        fixed = continuation["fixed"]
        segments = fixed["segments"]
        prefix.require(
            len(segments) <= 5
            and fixed["actual_model_requests"] == 0
            and fixed["automatic_retry"] is False
            and (
                (fixed["complete"] is True and fixed["stop"] is None)
                or (fixed["complete"] is False and fixed["stop"] is not None)
            ),
            "The fixed-reference run must finish or retain an explicit stop",
        )
        periods = []
        for index, segment in enumerate(segments):
            current = segment["continuation"]["periods"]
            prefix.require(
                segment["start_step"] == 8 + 8 * index
                and segment["reference_offset"] == 8 * index
                and segment["generated_endpoints"] == 40 - 8 * index
                and segment["terminal_hold_endpoints"] == 8 * index
                and segment["continuation"]["steps_completed"] == len(current)
                and 0 <= len(current) <= 8
                and (index == len(segments) - 1 or len(current) == 8),
                "The physical segments must preserve generated targets and marked terminal holds",
            )
            periods.extend(current)
        prefix.require(not fixed["complete"] or len(periods) == 40, "A complete reference contains 40 actual periods")
        receipts, starts = runtime["segment_receipts"], runtime["segment_trace_starts"]
        prefix.require(
            len(receipts) == len(starts) == 1 + len(segments)
            and starts[0] == dict(shared=0, causal=0)
            and runtime["newly_selected_commands"] == 8 + len(periods),
            "Every actual selected action and segment history must be recorded",
        )
        for key, traces in (
            ("shared", receipts[0]["shared_shift_traces"]),
            ("causal", receipts[0]["causal_shift_fallback_traces"]),
        ):
            values = [row[key] for row in starts]
            prefix.require(
                values == sorted(values) and all(type(value) is int and 0 <= value <= len(traces) for value in values),
                "Solver history boundaries must be ordered",
            )
        budget = dict(
            quality_responses=sum(row["quality_refiner"]["quality_response_evaluations"] for row in receipts),
            baseline_responses=sum(row["quality_refiner"]["baseline_response_evaluations"] for row in receipts),
            gripper_sequences=sum(row["gripper_proposals"]["candidate_sequences"] for row in receipts),
            endpoint_queries=sum(row["quality_refiner"]["endpoint_queries_charged"] for row in receipts),
            transition_steps=sum(row["quality_refiner"]["transition_steps_charged"] for row in receipts),
        )
        prefix.require(
            all(type(value) is int and value >= 0 for value in budget.values())
            and budget["quality_responses"] <= 48 * 320
            and budget["baseline_responses"] <= 48 * 480
            and budget["gripper_sequences"] <= 48 * 6561
            and budget["endpoint_queries"] <= 9600
            and budget["transition_steps"] <= 124800,
            "The physical branch exceeds the registered candidate budgets",
        )
        stop = fixed["stop"]
    else:
        prefix.require(
            producer is not None
            and branch["candidate_modules_unloaded"] is True
            and runtime["newly_selected_commands"] == 0
            and continuation["stop_reason_independently_verified"] is False,
            "The isolated checker cannot select actions or recompute solver stops",
        )
        fixed = producer["runtime"]["continuation"]["fixed"]
        expected = [row for segment in fixed["segments"] for row in segment["continuation"]["periods"]]
        periods = [row for replay in continuation["replays"] for row in replay["periods"]]
        prefix.require(
            len(continuation["replays"]) == len(fixed["segments"])
            and len(periods) == len(expected)
            and continuation["reported_producer_stop"] == fixed["stop"],
            "The checker must replay the exact actual fixed-reference segments",
        )
        stop = continuation["reported_producer_stop"]
    all_periods = first + periods
    count = len(all_periods)
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
    counters = runtime["counters"]
    prefix.require(
        count <= 48
        and [row["step"] for row in all_periods] == list(range(count))
        and runtime["control_steps_attempted"] == runtime["control_steps_completed"] == count
        and counters["mj_step"]["attempted"] == counters["mj_step"]["completed"] == count * 25
        and counters["model_construct"]["completed"] == counters["context"]["completed"] == 3
        and all(len(row["actual_inner"]) == 25 for row in all_periods),
        "All actual models, contexts and physical samples must match the registered budget",
    )
    error = np.asarray([row["tcp_error"] for row in all_periods], dtype=float)
    prefix.require(error.shape == (count, 6) and np.isfinite(error).all(), "Finite measured TCP errors required")
    scores = [row["completed_feedback"]["score"] for row in periods]
    return dict(
        control_steps=count,
        native_steps=count * 25,
        models=3,
        contexts=3,
        generated_reference_steps=len(periods),
        last_generated_endpoint_index=None if not periods else len(periods) - 1,
        model_requests=0,
        stop=stop,
        candidate_budget=budget,
        pose_caps_satisfied=sum(bool(row["pose_caps_satisfied"]) for row in all_periods),
        position_rmse_m=float(np.sqrt(np.mean(np.sum(error[:, :3] ** 2, axis=1)))),
        rotation_rmse_rad=float(np.sqrt(np.mean(np.sum(error[:, 3:] ** 2, axis=1)))),
        minimum_inner_margin_rad=float(min(row["minimum_inner_margin_rad"] for row in all_periods)),
        minimum_reference_opening=None if not scores else float(min(row["opening_reference"] for row in scores)),
        minimum_actual_opening=None if not scores else float(min(row["opening"] for row in scores)),
        target_grasp_periods=sum(bool(row["target_grasp"]) for row in all_periods),
        task_success_periods=sum(bool(row["task_success"]) for row in all_periods),
    )


def compare_old_prefix(branch, old):
    current, prior = branch["runtime"]["continuation"], old["runtime"]["continuation"]
    current_first, prior_first = current["first_window"]["periods"], prior["first_window"]["periods"]
    current_live = [row for segment in current["fixed"]["segments"] for row in segment["continuation"]["periods"]][:8]
    prior_live = prior["stream"]["segments"][0]["continuation"]["periods"][: len(current_live)]
    result = {}
    for name, left, right in (
        ("reset_first_window", current_first, prior_first),
        ("same_reference_prefix", current_live, prior_live),
    ):
        try:
            result[name] = dict(
                periods=len(left), matched=True, max_absolute_difference=comparison.compare_periods(left, right)
            )
        except (ValueError, AssertionError) as error:
            result[name] = dict(periods=len(left), matched=False, reason=str(error))
    return result


def execute(phase):
    prefix.environment()
    prefix.require(phase in ("producer", "checker"), "A registered CPU phase is required")
    inputs = admission.load_inputs()
    before = check_qa_and_dry(inputs)
    sources = admission.source_refs()
    producer_ref = producer = prior = None
    if phase == "checker":
        prior = prefix.read(prefix.reference(OUTPUT / "producer/result.json"))
        prefix.require(
            prior["complete"] is prior["passed"] is prior["sources_unchanged"] is True,
            "A closed fixed-reference producer is required before independent replay",
        )
        producer_ref = prior["branch_ref"]
        producer = prefix.read(producer_ref)
    directory = OUTPUT / phase
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "simulation").mkdir()
    registration = publish(
        directory / "registration.json",
        dict(
            phase=phase, inputs=admission.input_receipt(inputs), source_refs=sources, producer_branch_ref=producer_ref
        ),
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
        environment = physical._cpu_environment(directory / "simulation")
        spec = dict(key=phase, argv=[sys.executable, "-m", MODULE, "--worker", phase], log=directory / "worker.log")
        with physical._temporary_environment(environment):
            lifecycle = run_owned_workers([spec], cwd=ROOT, timeout_seconds=inputs["config"]["worker_timeout_seconds"])
        result["workers_ref"] = publish(directory / "workers.json", lifecycle)
        registered_feedback_phase.require_closed_workers(lifecycle, [phase])
        branch_ref = prefix.reference(directory / "simulation/result.json")
        branch = prefix.read(branch_ref)
        prefix.require(
            branch["case"]["case_id"] == inputs["config"]["case_id"],
            "The physical result belongs to the registered case",
        )
        counts = summarize_branch(branch, phase, producer=producer)
        if phase == "checker":
            prefix.require(
                counts["control_steps"] == prior["counts"]["control_steps"]
                and counts["native_steps"] == prior["counts"]["native_steps"],
                "Independent replay must cover exactly the producer's dispatched periods",
            )
        else:
            result["old_reobservation_prefix_comparison"] = compare_old_prefix(branch, inputs["old_producer"])
        for path, digest in inputs["source_sha256"].items():
            prefix.checked({"path": path, "sha256": digest})
        prefix.require(
            admission.source_refs() == sources and snapshot(ROOT)["sources"] == before,
            "Sources changed during execution",
        )
        result.update(complete=True, passed=True, sources_unchanged=True, branch_ref=branch_ref, counts=counts)
    except BaseException as error:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        branch_path = directory / "simulation/result.json"
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
