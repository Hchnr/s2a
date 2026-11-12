"""Own and independently replay the next verified bowl H40 reference."""

import argparse
from pathlib import Path

import numpy as np

from s2a.infrastructure.publication import publish_result as publish
from scripts.simulation.runtime import audit_reset_multitask_live_recovery as comparison
from scripts.simulation.runtime import registered_feedback_phase as phase_owner
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_reset_bowl_reference_tail as previous
from scripts.wam.tracking.preview import gpu_resume_repro_20261111 as repro

ROOT = prefix.REPO
CONFIG = ROOT / "configs/tracking/references/reset_bowl_step88_reference_20261112.json"
PLAN = ROOT / "plans/v0.0.1/tracking/references/reset_bowl_step88_reference_20261112.md"
OUTPUT = ROOT / "outputs/reset_bowl_step88_reference_20261112"
QA = ROOT / "outputs/qa/reset_bowl_step88_reference_20261112/result.json"
DRY = ROOT / "outputs/reset_bowl_step88_reference_dry_20261112/result.json"
CAPTURE = {
    "path": str(ROOT / "outputs/reset_bowl_step88_capture_20261112/result.json"),
    "sha256": "8a8175014b84494291498e9c8b706b9450d0ee5fabef320b6d1014a28c41553c",
}
MODEL = {
    "path": str(ROOT / "outputs/reset_bowl_step88_model_20261112/result.json"),
    "sha256": "19c97d3a85b73e17ff656ca471cc001dce66cbd0fa9498df895ff7b26df326c9",
}
PRODUCER88 = {
    "path": str(ROOT / "outputs/reset_bowl_h40_tail_20261112/producer/result.json"),
    "sha256": "006c62083fa238862f238a5e017aed2be41dbdc0d7e0d0a546c7bb1955c6b4ef",
}
FIXED = dict(
    schema="s2a.reset-bowl-step88-reference.v1",
    output=str(OUTPUT.relative_to(ROOT)),
    case_id="task0_init0_s42_raw",
    seed=42,
    verified_prefix_steps=88,
    reference_start_step=88,
    reference_consumed=0,
    max_reference_segments=5,
    max_control_steps_per_phase=128,
    max_native_steps_per_phase=3200,
    max_models_per_phase=3,
    max_contexts_per_phase=3,
    previous_control_steps=982,
    previous_native_steps=24550,
    previous_models=63,
    previous_contexts=63,
    previous_model_initializations=8,
    previous_requests_attempted=20,
    previous_requests_completed=17,
    previous_denoises=595,
    model_initializations=0,
    model_requests=0,
    fresh_cases=0,
    automatic_retry=False,
    worker_timeout_seconds=2400,
)
_OLD_LOAD, _OLD_RECEIPT, _OLD_SOURCES = previous.load_inputs, previous.input_receipt, previous.source_refs


def source_refs():
    paths = (
        CONFIG,
        PLAN,
        Path(__file__).resolve(),
        ROOT / "scripts/simulation/runtime/reset_bowl_next_reference.py",
        ROOT / "src/s2a/research/native_runtime/verified_terminal_capture.py",
        ROOT / "tests/research/native_runtime/test_reset_bowl_next_reference.py",
        ROOT / "tests/research/native_runtime/test_reset_bowl_next_reference_owner.py",
    )
    return _OLD_SOURCES() + [prefix.reference(path) for path in paths]


def load_inputs():
    prefix.environment()
    config = prefix.read(prefix.reference(CONFIG))
    prefix.require(
        type(config) is dict
        and set(config) == set(FIXED)
        and all(type(config[key]) is type(value) and config[key] == value for key, value in FIXED.items()),
        "The exact step-88 reference and 128-period limits are required",
    )
    base = _OLD_LOAD()
    capture, model, producer = (prefix.read(ref) for ref in (CAPTURE, MODEL, PRODUCER88))
    prefix.require(
        all(
            item["complete"] is item["passed"] is item["sources_unchanged"] is True
            for item in (capture, model, producer)
        )
        and capture["control_qualified"] is model["control_qualified"] is producer["control_qualified"] is False
        and capture["counts"]["control_steps"] == producer["counts"]["control_steps"] == 88
        and capture["counts"]["stop"] is producer["counts"]["stop"] is None
        and all(value == 0 for value in capture["independent_max_absolute_difference"].values())
        and capture["inputs"]["producer_branch"] == producer["branch_refs"][0]
        and capture["cumulative_control_steps"] == config["previous_control_steps"]
        and capture["cumulative_native_steps"] == config["previous_native_steps"]
        and capture["cumulative_models"] == config["previous_models"]
        and capture["cumulative_contexts"] == config["previous_contexts"]
        and model["model_initializations"]
        == model["model_requests_attempted"]
        == model["model_requests_completed"]
        == 1
        and model["denoise_calls_completed"] == 35
        and model["gpu_released"] is True
        and model["physics_steps"] == 0
        and base["config"]["previous_model_initializations"] + 1 == config["previous_model_initializations"]
        and base["config"]["previous_requests_attempted"] + 1 == config["previous_requests_attempted"]
        and base["config"]["previous_requests_completed"] + 1 == config["previous_requests_completed"]
        and base["config"]["previous_denoises"] + 35 == config["previous_denoises"],
        "The verified prefix, capture, model output and cumulative costs are required",
    )
    producer_ref = producer["branch_refs"][0]
    producer_branch = prefix.read(producer_ref)
    reference, envelope = prefix.read(model["raw_response_ref"]), prefix.read(model["response_ref"])
    arrays = repro.response_arrays(reference)
    np.testing.assert_array_equal(arrays["action"], envelope["action"])
    np.testing.assert_array_equal(arrays["normalized_action"], envelope["normalized_action"])
    image = capture["terminal_capture"]["capture"]
    prefix.checked(image["canvas_png"])
    prefix.require(
        producer_branch["complete"] is producer_branch["passed"] is True
        and producer_branch["case"]["case_id"] == config["case_id"]
        and envelope["step"] == image["frame_index"] == 88
        and envelope["image_sha256"] == image["canvas_png"]["sha256"]
        and envelope["state_sha256"] == image["current_endpoint"]["sim_state_sha256"]
        and envelope["sim_time_s"] == image["current_endpoint"]["sim_time"]
        and envelope["denoise_calls"] == 35,
        "The model response must bind the verified step-88 observation",
    )
    refs = dict(
        config=prefix.reference(CONFIG),
        plan=prefix.reference(PLAN),
        capture=CAPTURE,
        model=MODEL,
        producer=PRODUCER88,
        producer_branch=producer_ref,
        response=model["raw_response_ref"],
        envelope=model["response_ref"],
    )
    consumed = dict(base["source_sha256"])
    for ref in refs.values():
        prefix.checked(ref, consumed)
    return dict(
        config=config,
        prior_inputs=base,
        previous=base["previous"],
        previous_producer=base["previous_producer"],
        previous_88_producer=producer_branch,
        capture88=image,
        reference88=reference,
        envelope88=envelope,
        refs=refs,
        source_sha256=consumed,
    )


def input_receipt(inputs):
    return dict(refs=inputs["refs"], prior=_OLD_RECEIPT(inputs["prior_inputs"]), source_sha256=inputs["source_sha256"])


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


def bind_worker(inputs):
    """Extend the audited phase worker while preserving its original source."""
    from s2a.research.native_runtime import reset_bounded_runtime as bounded
    from scripts.simulation.runtime import reset_bowl_next_reference as next_reference
    from scripts.simulation.runtime import reset_bowl_reference_tail as old_tail

    old_producer, old_checker, old_adapter = old_tail.run_producer, old_tail.run_checker, bounded.ResetBoundedRuntime

    class Adapter128(old_adapter):
        def run_feedback(self, *args, max_control_steps, **kwargs):
            prefix.require(max_control_steps == 88, "The inherited worker must request its original bound")
            return super().run_feedback(*args, max_control_steps=128, **kwargs)

    def run_producer(*args, **kwargs):
        return next_reference.run_producer(
            *args,
            previous_88_producer=inputs["previous_88_producer"],
            expected_step88_capture=inputs["capture88"],
            load_step88_reference=lambda: (inputs["reference88"], inputs["envelope88"]),
            prefix_runner=old_producer,
            **kwargs,
        )

    def run_checker(*args, **kwargs):
        return next_reference.run_checker(
            *args,
            expected_step88_capture=inputs["capture88"],
            load_step88_reference=lambda: (inputs["reference88"], inputs["envelope88"]),
            prefix_runner=old_checker,
            **kwargs,
        )

    previous.OUTPUT, previous.load_inputs = OUTPUT, lambda: inputs
    previous.input_receipt, previous.source_refs = input_receipt, source_refs
    old_tail.run_producer, old_tail.run_checker = run_producer, run_checker
    bounded.ResetBoundedRuntime = Adapter128


def worker(phase, index, mode):
    inputs = load_inputs()
    bind_worker(inputs)
    previous.worker(phase, index, mode)


def summarize_branch(branch, phase, config, *, producer=None):
    from scripts.simulation.runtime import reset_bowl_next_reference as next_reference
    from scripts.simulation.runtime import reset_bowl_reference_tail as tail

    runtime = branch["runtime"]
    continuation = runtime["continuation"]
    prefix_rows = next_reference.prefix_periods(continuation["prefix88"], replay=phase == "checker")
    new_rows = tail.tail_periods(continuation["new_reference"])
    rows = prefix_rows + new_rows
    count, counters = len(rows), runtime["counters"]
    prefix.require(
        phase in ("producer", "checker")
        and branch["complete"] is branch["passed"] is runtime["complete"] is runtime["passed"] is True
        and runtime["cleanup_complete"] is runtime["reset_observation_verified"] is runtime["reset_image_match"] is True
        and runtime["previous_saved_command"] is None
        and 8 <= count <= config["max_control_steps_per_phase"]
        and [row["step"] for row in rows] == list(range(count))
        and len(prefix_rows) <= 88
        and len(new_rows) <= 40
        and (not new_rows or len(prefix_rows) == 88)
        and runtime["control_steps_attempted"] == runtime["control_steps_completed"] == count
        and counters["mj_step"]["attempted"] == counters["mj_step"]["completed"] == count * 25
        and counters["model_construct"]["completed"] == counters["context"]["completed"] == 3
        and all(len(row["actual_inner"]) == 25 for row in rows),
        "All actual periods, inner steps, models and contexts must match registration",
    )
    stop = continuation["new_reference"]["stop"]
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
            and budget["quality_responses"] <= 128 * 320
            and budget["baseline_responses"] <= 128 * 480
            and budget["gripper_sequences"] <= 128 * 6561
            and budget["endpoint_queries"] <= 25600
            and budget["transition_steps"] <= 332800,
            "The physical branch exceeds the original per-period candidate budgets",
        )
    else:
        budget = None
        prefix.require(
            producer is not None
            and branch["candidate_modules_unloaded"] is True
            and runtime["newly_selected_commands"] == 0
            and continuation["new_reference"]["stop_recomputed"] is False,
            "An isolated checker cannot select actions or recompute the producer stop",
        )
    errors = np.asarray([row["tcp_error"] for row in rows], dtype=float)
    prefix.require(errors.shape == (count, 6) and np.isfinite(errors).all(), "Finite TCP errors are required")
    return dict(
        control_steps=count,
        native_steps=count * 25,
        models=3,
        contexts=3,
        prefix_steps=len(prefix_rows),
        new_reference_steps=len(new_rows),
        stop=stop,
        candidate_budget=budget,
        pose_caps_satisfied=sum(bool(row["pose_caps_satisfied"]) for row in rows),
        position_rmse_m=float(np.sqrt(np.mean(np.sum(errors[:, :3] ** 2, axis=1)))),
        rotation_rmse_rad=float(np.sqrt(np.mean(np.sum(errors[:, 3:] ** 2, axis=1)))),
        minimum_inner_margin_rad=float(min(row["minimum_inner_margin_rad"] for row in rows)),
        target_grasp_periods=sum(bool(row["target_grasp"]) for row in rows),
        task_success_periods=sum(bool(row["task_success"]) for row in rows),
        new_target_grasp_periods=sum(bool(row["target_grasp"]) for row in new_rows),
        new_task_success_periods=sum(bool(row["task_success"]) for row in new_rows),
    )


def summarize_phase(branches, phase, config):
    from scripts.simulation.runtime import reset_bowl_next_reference as next_reference
    from scripts.simulation.runtime import reset_bowl_reference_tail as tail

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
        actual = next_reference.prefix_periods(produced["prefix88"]) + tail.tail_periods(produced["new_reference"])
        checked = next_reference.prefix_periods(replayed["prefix88"], replay=True) + tail.tail_periods(
            replayed["new_reference"]
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
        worker_module="scripts.simulation.runtime.run_reset_bowl_next_reference",
        branch_key=previous.branch_key,
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
