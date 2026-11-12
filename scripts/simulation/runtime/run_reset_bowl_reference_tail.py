"""Own a bounded, independently replayed continuation of a saved bowl reference."""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

from s2a.infrastructure.native_environment import restore_cv2_environment
from s2a.infrastructure.publication import publish_result as publish
from scripts.simulation.runtime import audit_reset_multitask_live_recovery as comparison
from scripts.simulation.runtime import registered_feedback_phase as phase_owner
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import reset_bowl_fixed_h40_admission as original_admission
from scripts.simulation.runtime import run_causal_stream_recorded_physical as physical
from scripts.simulation.runtime import run_reset_bowl_h40_second_window as previous
from scripts.simulation.runtime import run_reset_start_feedback as original
from scripts.simulation.runtime import run_segmented_reset_equivalence as isolation

ROOT = prefix.REPO
CONFIG = ROOT / "configs/tracking/references/reset_bowl_h40_tail_20261112.json"
PLAN = ROOT / "plans/v0.0.1/tracking/references/finite_reference_tail_20261112.md"
OUTPUT = ROOT / "outputs/reset_bowl_h40_tail_20261112"
QA = ROOT / "outputs/qa/reset_bowl_h40_tail_20261112/result.json"
DRY = ROOT / "outputs/reset_bowl_h40_tail_dry_20261112/result.json"
PRODUCER = {
    "path": str(ROOT / "outputs/reset_bowl_h40_second_window_20261112/producer/result.json"),
    "sha256": "0cf05b609d1d1ee189413fda27158af81e19a1fd5f8ce307542e28ccc9b3e25a",
}
CHECKER = {
    "path": str(ROOT / "outputs/reset_bowl_h40_second_window_20261112/checker/result.json"),
    "sha256": "4d763b9fcebcf5b04696716ff0f81fd8849035134b966afa88ab2d22dc4fab9a",
}
FIXED = dict(
    schema="s2a.reset-bowl-h40-tail.v1",
    output=str(OUTPUT.relative_to(ROOT)),
    case_id="task0_init0_s42_raw",
    seed=42,
    verified_prefix_steps=56,
    reference_start_step=48,
    reference_consumed=8,
    max_tail_segments=4,
    max_control_steps_per_phase=88,
    max_native_steps_per_phase=2200,
    max_models_per_phase=3,
    max_contexts_per_phase=3,
    previous_control_steps=718,
    previous_native_steps=17950,
    previous_models=54,
    previous_contexts=54,
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
        ROOT / "scripts/simulation/runtime/reset_bowl_reference_tail.py",
        ROOT / "src/s2a/research/native_runtime/finite_reference_tail.py",
        ROOT / "tests/research/native_runtime/test_finite_reference_tail.py",
        ROOT / "tests/research/native_runtime/test_reset_bowl_reference_tail.py",
        ROOT / "tests/research/native_runtime/test_reset_bowl_reference_tail_owner.py",
    )
    return previous.source_refs() + [prefix.reference(path) for path in paths]


def load_inputs():
    prefix.environment()
    config = prefix.read(prefix.reference(CONFIG))
    prefix.require(
        type(config) is dict
        and set(config) == set(FIXED)
        and all(type(config[key]) is type(value) and config[key] == value for key, value in FIXED.items()),
        "The exact saved-reference continuation and budgets are required",
    )
    base = previous.load_inputs()
    producer, checker = prefix.read(PRODUCER), prefix.read(CHECKER)
    prefix.require(
        all(item["complete"] is item["passed"] is item["sources_unchanged"] is True for item in (producer, checker))
        and producer["control_qualified"] is checker["control_qualified"] is False
        and producer["counts"]["control_steps"] == checker["counts"]["control_steps"] == 56
        and producer["counts"]["second_reference_steps"] == checker["counts"]["second_reference_steps"] == 8
        and producer["counts"]["stop"] is checker["counts"]["stop"] is None
        and all(value == 0 for value in checker["independent_max_absolute_difference"].values())
        and checker["cumulative_control_steps"] == config["previous_control_steps"]
        and checker["cumulative_native_steps"] == config["previous_native_steps"]
        and checker["cumulative_models"] == config["previous_models"]
        and checker["cumulative_contexts"] == config["previous_contexts"]
        and base["config"]["previous_model_initializations"] == config["previous_model_initializations"]
        and base["config"]["previous_requests_attempted"] == config["previous_requests_attempted"]
        and base["config"]["previous_requests_completed"] == config["previous_requests_completed"]
        and base["config"]["previous_denoises"] == config["previous_denoises"],
        "The independently replayed 56-period prefix and prior costs are required",
    )
    producer_branch = prefix.read(producer["branch_ref"])
    checker_branch = prefix.read(checker["branch_ref"])
    prefix.require(
        producer_branch["complete"] is producer_branch["passed"] is True
        and checker_branch["complete"] is checker_branch["passed"] is True
        and checker_branch["candidate_modules_unloaded"] is True
        and producer_branch["case"]["case_id"] == checker_branch["case"]["case_id"] == config["case_id"],
        "The exact producer, isolated checker and case are required",
    )
    refs = dict(
        config=prefix.reference(CONFIG),
        plan=prefix.reference(PLAN),
        producer=PRODUCER,
        checker=CHECKER,
        producer_branch=producer["branch_ref"],
        checker_branch=checker["branch_ref"],
    )
    consumed = dict(base["base"]["source_sha256"])
    for ref in refs.values():
        prefix.checked(ref, consumed)
    return dict(config=config, previous=base, refs=refs, previous_producer=producer_branch, source_sha256=consumed)


def input_receipt(inputs):
    return dict(
        refs=inputs["refs"], previous=previous.input_receipt(inputs["previous"]), source_sha256=inputs["source_sha256"]
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
    prefix.require(
        qa["complete"] is qa["passed"] is qa["all_tests_passed"] is qa["source_unchanged"] is True
        and qa["actual_wait"] is qa["reaped"] is True
        and qa["signal_sent"] is False
        and qa["configured_parallelism"] == 32
        and qa["skipped_count"] == qa["xfail_count"] == qa["xpass_count"] == 0
        and not qa["collection"]["collection_errors"]
        and dry["complete"] is dry["passed"] is True
        and dry["inputs"] == input_receipt(inputs)
        and dry["source_refs"] == source_refs()
        and dry["native_models"] == dry["physics_steps"] == dry["model_initializations"] == dry["model_requests"] == 0,
        "A complete 32-worker regression and zero-physics dry are required",
    )


def branch_key(index, mode):
    prefix.require(index == 0 and mode == "raw", "One registered bowl case is required")
    return "case_00"


def worker(phase, index, mode):
    key = branch_key(index, mode)
    directory = OUTPUT / phase / key
    result = dict(
        complete=False, passed=False, control_qualified=False, phase=phase, automatic_retry=False, fresh_cases=0
    )
    try:
        os.environ.update(physical._cpu_environment(directory))
        prefix.environment()
        inputs = load_inputs()
        registration = prefix.read(prefix.reference(OUTPUT / phase / "registration.json"))
        prefix.require(
            registration["phase"] == phase
            and registration["inputs"] == input_receipt(inputs)
            and registration["source_refs"] == source_refs(),
            "The worker must use the registered sources and inputs",
        )
        producer = None
        if phase == "checker":
            isolation.checker_isolation()
            producer = prefix.read(registration["producer_branch_refs"][key])
            prefix.require(producer["complete"] is producer["passed"] is True, "The completed producer is required")
        base = inputs["previous"]
        cv2 = base["base"]["base"]["previous"]["base"]["cv2_admission"]
        result["cv2_admission"] = restore_cv2_environment(cv2["cv2_probe_ref"], cv2["cv2_runtime_source_refs"])
        original.require_no_simulator_import()
        from s2a.research.native_runtime.development_prefix_runtime_bounded import PrefixReplayFailure
        from s2a.research.native_runtime.reset_bounded_runtime import ResetBoundedRuntime
        from scripts.simulation.runtime import reset_bowl_reference_tail as tail

        case = base["base"]["case"]
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
                expected_capture=base["base"]["expected_capture"],
                load_reference=lambda: original_admission.load_reference(base["base"]),
                runtime_output=runtime_output,
                expected_second_capture=base["capture"],
                load_second_reference=lambda: (base["reference"], base["envelope"]),
                progress=progress,
            )
            if phase == "producer":
                return tail.run_producer(
                    env,
                    terminal,
                    observers,
                    receipt,
                    baseline=base["base"]["baseline"],
                    previous_producer=base["previous_producer"],
                    previous_56_producer=inputs["previous_producer"],
                    **common,
                )
            return tail.run_checker(env, terminal, observers, receipt, producer=producer, **common)

        try:
            runtime = adapter.run_feedback(
                case["config"],
                case["record"],
                case["record_xml"],
                runtime_output,
                base["base"]["base"]["previous"]["base"]["asset_refs"],
                registered_seed=42,
                continuation=continuation,
                max_control_steps=88,
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


def summarize_branch(branch, phase, config, *, producer=None):
    from scripts.simulation.runtime import reset_bowl_reference_tail as tail_module

    runtime = branch["runtime"]
    continuation = runtime["continuation"]
    prefix_periods = tail_module.prefix_periods(continuation["prefix"], replay=phase == "checker")
    tail_periods = tail_module.tail_periods(continuation["tail"])
    all_periods = prefix_periods + tail_periods
    count = len(all_periods)
    counters = runtime["counters"]
    prefix.require(
        phase in ("producer", "checker")
        and branch["complete"] is branch["passed"] is runtime["complete"] is runtime["passed"] is True
        and runtime["cleanup_complete"] is runtime["reset_observation_verified"] is runtime["reset_image_match"] is True
        and runtime["previous_saved_command"] is None
        and 8 <= count <= config["max_control_steps_per_phase"]
        and [row["step"] for row in all_periods] == list(range(count))
        and len(prefix_periods) <= 56
        and len(tail_periods) <= 32
        and (not tail_periods or len(prefix_periods) == 56)
        and runtime["control_steps_attempted"] == runtime["control_steps_completed"] == count
        and counters["mj_step"]["attempted"] == counters["mj_step"]["completed"] == count * 25
        and counters["model_construct"]["completed"] == counters["context"]["completed"] == 3
        and all(len(row["actual_inner"]) == 25 for row in all_periods),
        "All actual periods, inner steps, models and contexts must match the registered limits",
    )
    stop = continuation["tail"]["stop"]
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
            "The recorded stop must immediately follow the actual dispatches",
        )
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
            and budget["quality_responses"] <= 88 * 320
            and budget["baseline_responses"] <= 88 * 480
            and budget["gripper_sequences"] <= 88 * 6561
            and budget["endpoint_queries"] <= 17600
            and budget["transition_steps"] <= 228800,
            "The physical branch exceeds the original per-period candidate budgets",
        )
    else:
        budget = None
        prefix.require(
            producer is not None
            and branch["candidate_modules_unloaded"] is True
            and runtime["newly_selected_commands"] == 0
            and continuation["tail"]["stop_recomputed"] is False,
            "An isolated checker cannot select actions or recompute the producer stop",
        )
    errors = np.asarray([row["tcp_error"] for row in all_periods], dtype=float)
    prefix.require(errors.shape == (count, 6) and np.isfinite(errors).all(), "Finite TCP errors are required")
    return dict(
        control_steps=count,
        native_steps=count * 25,
        models=3,
        contexts=3,
        prefix_steps=len(prefix_periods),
        tail_steps=len(tail_periods),
        stop=stop,
        candidate_budget=budget,
        pose_caps_satisfied=sum(bool(row["pose_caps_satisfied"]) for row in all_periods),
        position_rmse_m=float(np.sqrt(np.mean(np.sum(errors[:, :3] ** 2, axis=1)))),
        rotation_rmse_rad=float(np.sqrt(np.mean(np.sum(errors[:, 3:] ** 2, axis=1)))),
        minimum_inner_margin_rad=float(min(row["minimum_inner_margin_rad"] for row in all_periods)),
        target_grasp_periods=sum(bool(row["target_grasp"]) for row in all_periods),
        task_success_periods=sum(bool(row["task_success"]) for row in all_periods),
        tail_target_grasp_periods=sum(bool(row["target_grasp"]) for row in tail_periods),
        tail_task_success_periods=sum(bool(row["task_success"]) for row in tail_periods),
    )


def summarize_phase(branches, phase, config):
    from scripts.simulation.runtime import reset_bowl_reference_tail as tail_module

    prefix.require(len(branches) == 1 and branches[0]["case"]["case_id"] == config["case_id"], "One case is required")
    branch = branches[0]
    producer = None
    if phase == "checker":
        prior = prefix.read(prefix.reference(OUTPUT / "producer/result.json"))
        producer = prefix.read(prior["branch_refs"][0])
    counts = summarize_branch(branch, phase, config, producer=producer)
    if phase == "checker":
        produced = producer["runtime"]["continuation"]
        replayed = branch["runtime"]["continuation"]
        actual = tail_module.prefix_periods(produced["prefix"]) + tail_module.tail_periods(produced["tail"])
        checked = tail_module.prefix_periods(replayed["prefix"], replay=True) + tail_module.tail_periods(
            replayed["tail"]
        )
        counts["independent_max_absolute_difference"] = comparison.compare_periods(actual, checked)
        prior_counts = prior["counts"]
        prefix.require(counts["control_steps"] == prior_counts["control_steps"], "Checker dispatch count differs")
        counts.update(
            cumulative_control_steps=config["previous_control_steps"]
            + prior_counts["control_steps"]
            + counts["control_steps"],
            cumulative_native_steps=config["previous_native_steps"]
            + prior_counts["native_steps"]
            + counts["native_steps"],
            cumulative_models=config["previous_models"] + prior_counts["models"] + counts["models"],
            cumulative_contexts=config["previous_contexts"] + prior_counts["contexts"] + counts["contexts"],
        )
    return counts


def execute(phase):
    inputs = load_inputs()
    check_qa_and_dry(inputs)
    phase_owner.execute_phase(
        phase,
        root=OUTPUT,
        qa_path=QA,
        dry_path=DRY,
        branches=[(0, "raw")],
        worker_module="scripts.simulation.runtime.run_reset_bowl_reference_tail",
        branch_key=branch_key,
        load_inputs=load_inputs,
        input_receipt=input_receipt,
        source_refs=source_refs,
        summarize_phase=summarize_phase,
        config_key="config",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true")
    group.add_argument("--phase", choices=("producer", "checker"))
    group.add_argument("--worker", nargs=3, metavar=("PHASE", "INDEX", "MODE"))
    args = parser.parse_args()
    if args.dry_run:
        dry_run()
    elif args.phase:
        execute(args.phase)
    else:
        worker(args.worker[0], int(args.worker[1]), args.worker[2])


if __name__ == "__main__":
    main()
