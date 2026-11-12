"""Own and independently replay a triggered current-image reference."""

import argparse
from pathlib import Path

import numpy as np

from s2a.infrastructure.publication import publish_result as publish
from scripts.simulation.runtime import audit_reset_multitask_live_recovery as comparison
from scripts.simulation.runtime import causal_replan_step107_execution as continuation
from scripts.simulation.runtime import registered_feedback_phase as phase_owner
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import reset_bowl_reference_tail as tail
from scripts.simulation.runtime import run_reset_bowl_next_reference as original
from scripts.tracking.trajectory_search import audit_replayed_reference_pose_bounds as bound
from scripts.tracking.trajectory_search import audit_replayed_reference_pose_bounds_v2 as source
from scripts.wam.tracking.preview import gpu_resume_repro_20261111 as repro

ROOT = prefix.REPO
CONFIG = ROOT / "configs/tracking/references/causal_replan_step107_execution_20261112.json"
PLAN = ROOT / "plans/v0.0.1/tracking/references/causal_replan_step107_execution_20261112.md"
OUTPUT = ROOT / "outputs/causal_replan_step107_execution_20261112"
QA = ROOT / "outputs/qa/causal_replan_step107_execution_20261112/result.json"
DRY = ROOT / "outputs/causal_replan_step107_execution_dry_20261112/result.json"
MODULE = "scripts.simulation.runtime.run_causal_replan_step107_execution"
CAPTURE = {
    "path": str(ROOT / "outputs/causal_replan_step107_capture_20261112/result.json"),
    "sha256": "44d1b205e85815cab0b215f1cdfa6d3cba73671f1d1c6135b318dabe04759656",
}
MODEL = {
    "path": str(ROOT / "outputs/causal_replan_step107_model_20261112/result.json"),
    "sha256": "46dc2c1d1ce1d5150ccede4cad68f695e535eba15b8a10fbebb9ff304e4302d0",
}
FIXED = dict(
    schema="s2a.causal-replan-step107-execution.v1",
    case_id="task0_init0_s42_raw",
    seed=42,
    trigger_reference_start=88,
    replan_step=107,
    state_sha256="04b8eea4cd6e611302b68ff036594129808fec0bb88cab97f4e55106ee4c148d",
    image_sha256="f7cc70d689471bbc96865f825dc44bc9e1c2607bf63bfa56b6114780ec2129a3",
    max_reference_segments=5,
    max_control_steps_per_phase=147,
    max_native_steps_per_phase=3675,
    max_models_per_phase=3,
    max_contexts_per_phase=3,
    previous_control_steps=1305,
    previous_native_steps=32625,
    previous_models=72,
    previous_contexts=72,
    previous_static_models=2,
    previous_model_initializations=9,
    previous_requests_attempted=21,
    previous_requests_completed=18,
    previous_denoises=630,
    model_initializations=0,
    model_requests=0,
    fresh_cases=0,
    automatic_retry=False,
    worker_timeout_seconds=3600,
)


def source_refs():
    paths = (
        CONFIG,
        PLAN,
        Path(__file__).resolve(),
        ROOT / "scripts/simulation/runtime/causal_replan_step107_execution.py",
        ROOT / "src/s2a/research/joint_control/preview/causal_reference_replan.py",
        ROOT / "src/s2a/research/native_runtime/finite_reference_tail.py",
        ROOT / "tests/research/native_runtime/test_causal_replan_step107_execution.py",
        ROOT / "tests/research/native_runtime/test_causal_replan_step107_execution_owner.py",
        ROOT / "tests/research/joint_control/preview/test_causal_reference_replan.py",
    )
    return original.source_refs() + [prefix.reference(path) for path in paths]


def load_inputs():
    prefix.environment()
    config_ref = prefix.reference(CONFIG)
    config = prefix.read(config_ref)
    prefix.require(
        type(config) is dict
        and set(config) == set(FIXED)
        and all(type(config[key]) is type(value) and config[key] == value for key, value in FIXED.items()),
        "The exact causal-trigger reference and physical budgets are required",
    )
    base = source.load_inputs()
    capture, model = prefix.read(CAPTURE), prefix.read(MODEL)
    capture_registration = prefix.read(capture["capture_registration_ref"])
    image, snap = capture["terminal_capture"]["capture"], capture["terminal_capture"]["snapshot"]
    prefix.require(
        capture["complete"] is capture["passed"] is capture["sources_unchanged"] is True
        and model["complete"] is model["passed"] is model["sources_unchanged"] is model["gpu_released"] is True
        and capture["control_qualified"] is model["control_qualified"] is False
        and capture_registration["inputs"]["bound_inputs"] == bound.input_receipt(base)
        and capture["counts"]["control_steps"] == config["replan_step"]
        and capture["counts"]["stop"] == dict(reason="requested_observation", at_step=107, next_action_dispatched=False)
        and all(value == 0 for value in capture["independent_max_absolute_difference"].values())
        and capture["cumulative_control_steps"] == config["previous_control_steps"]
        and capture["cumulative_native_steps"] == config["previous_native_steps"]
        and capture["cumulative_models"] == config["previous_models"]
        and capture["cumulative_contexts"] == config["previous_contexts"]
        and model["model_initializations"]
        == model["model_requests_attempted"]
        == model["model_requests_completed"]
        == 1
        and model["denoise_calls_completed"] == 35
        and model["physics_steps"] == 0
        and snap["step"] == image["frame_index"] == config["replan_step"]
        and snap["state_sha256"] == config["state_sha256"]
        and image["canvas_png"]["sha256"] == config["image_sha256"],
        "The independently replayed current image and completed model request are required",
    )
    prefix.checked(image["canvas_png"])
    raw, envelope = prefix.read(model["raw_response_ref"]), prefix.read(model["response_ref"])
    arrays = repro.response_arrays(raw)
    np.testing.assert_array_equal(arrays["action"], envelope["action"])
    np.testing.assert_array_equal(arrays["normalized_action"], envelope["normalized_action"])
    prefix.require(
        envelope["step"] == config["replan_step"]
        and envelope["state_sha256"] == config["state_sha256"]
        and envelope["image_sha256"] == config["image_sha256"]
        and envelope["sim_time_s"] == snap["sim_time_s"]
        and envelope["denoise_calls"] == 35,
        "The saved response must bind the exact current image, state and time",
    )
    refs = dict(
        config=config_ref,
        plan=prefix.reference(PLAN),
        capture=CAPTURE,
        model=MODEL,
        raw_response=model["raw_response_ref"],
        envelope=model["response_ref"],
        producer=base["refs"]["producer_branch"],
    )
    consumed = dict(base["source"]["source_sha256"])
    for ref in refs.values():
        prefix.checked(ref, consumed)
    return dict(
        config=config,
        base=base,
        previous=base["source"]["previous"],
        previous_producer=base["source"]["previous_producer"],
        capture=image,
        raw=raw,
        envelope=envelope,
        refs=refs,
        source_sha256=consumed,
    )


def input_receipt(inputs):
    return dict(refs=inputs["refs"], prior=bound.input_receipt(inputs["base"]), source_sha256=inputs["source_sha256"])


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
        "A completed CPU regression and zero-model dry are required",
    )


def bind_worker(inputs):
    """Extend the existing isolated worker with the triggered continuation."""
    from s2a.research.native_runtime import reset_bounded_runtime as bounded

    original_adapter = bounded.ResetBoundedRuntime
    original.OUTPUT = OUTPUT
    original.bind_worker(inputs["base"]["source"])
    old_producer, old_checker = tail.run_producer, tail.run_checker

    class Adapter147(original_adapter):
        def run_feedback(self, *args, max_control_steps, **kwargs):
            prefix.require(max_control_steps == 88, "The inherited worker must request its original bound")
            return super().run_feedback(*args, max_control_steps=147, **kwargs)

    def produce(*args, **kwargs):
        return continuation.run_producer(
            *args,
            prior_producer=inputs["base"]["producer"],
            expected_replan_capture=inputs["capture"],
            load_new_reference=lambda: (inputs["raw"], inputs["envelope"]),
            prefix_runner=old_producer,
            **kwargs,
        )

    def check(*args, **kwargs):
        return continuation.run_checker(
            *args,
            expected_replan_capture=inputs["capture"],
            load_new_reference=lambda: (inputs["raw"], inputs["envelope"]),
            prefix_runner=old_checker,
            **kwargs,
        )

    bounded.ResetBoundedRuntime = Adapter147
    tail.run_producer, tail.run_checker = produce, check
    original.previous.OUTPUT = OUTPUT
    original.previous.load_inputs = lambda: inputs
    original.previous.input_receipt, original.previous.source_refs = input_receipt, source_refs


def worker(phase, index, mode):
    import sys

    inputs = load_inputs()
    bind_worker(inputs)
    original.previous.worker(phase, index, mode)
    if phase == "checker":
        prefix.require(
            "s2a.research.joint_control.preview.causal_reference_replan" not in sys.modules,
            "The checker must not import the new candidate policy",
        )


def summarize_branch(branch, phase, config):
    runtime = branch["runtime"]
    sequence = runtime["continuation"]
    prior = continuation.prefix_periods(sequence["prefix107"], replay=phase == "checker")
    new = tail.tail_periods(sequence["replanned"])
    rows, counters = prior + new, runtime["counters"]
    count = len(rows)
    prefix.require(
        phase in ("producer", "checker")
        and branch["complete"] is branch["passed"] is runtime["complete"] is runtime["passed"] is True
        and runtime["cleanup_complete"] is runtime["reset_observation_verified"] is runtime["reset_image_match"] is True
        and runtime["previous_saved_command"] is None
        and len(prior) == config["replan_step"]
        and len(new) <= 40
        and [row["step"] for row in rows] == list(range(count))
        and runtime["control_steps_attempted"] == runtime["control_steps_completed"] == count
        and counters["mj_step"]["attempted"] == counters["mj_step"]["completed"] == 25 * count
        and counters["model_construct"]["completed"] == config["max_models_per_phase"]
        and counters["context"]["completed"] == config["max_contexts_per_phase"]
        and all(len(row["actual_inner"]) == 25 for row in rows),
        "Actual dispatches, endpoint history and simulation costs must match registration",
    )
    prefix_stop = sequence["prefix107"]["new_reference"]["stop"]
    prefix.require(
        prefix_stop["reason"] == "requested_observation"
        and prefix_stop["at_step"] == config["replan_step"]
        and prefix_stop["next_action_dispatched"] is False
        and prefix_stop["trigger"]["replan"] is True
        and prefix_stop["trigger"]["current_step"] == config["replan_step"]
        and runtime["replan_capture"]["snapshot"]["state_sha256"] == config["state_sha256"]
        and runtime["replan_capture"]["capture"]["canvas_png"]["sha256"] == config["image_sha256"],
        "The replan must follow the actual causal trigger",
    )
    stop = sequence["replanned"]["stop"]
    prefix.require(
        sequence["replanned"]["generated_periods"] == len(new)
        and (
            (stop is None and sequence["replanned"]["complete"] is True and len(new) == 40)
            or (stop is not None and sequence["replanned"]["complete"] is False and len(new) < 40)
        ),
        "The consumed generated-reference length must agree with its completion or stop",
    )
    if stop is not None:
        prefix.require(
            stop.get("next_action_dispatched") is False
            and (
                (stop.get("at_step") == count and "after_step" not in stop)
                or (stop.get("after_step") == count - 1 and "at_step" not in stop)
            ),
            "Any new stop must follow the actual dispatched periods",
        )
    if phase == "checker":
        prefix.require(
            branch["candidate_modules_unloaded"] is True
            and runtime["newly_selected_commands"] == 0
            and sequence["replanned"]["stop_recomputed"] is False,
            "The checker cannot select actions or recompute producer stops",
        )
        budget = None
    else:
        prefix.require(runtime["newly_selected_commands"] == count, "Every producer action must be counted")
        budget = dict(
            quality_responses=sum(
                row["quality_refiner"]["quality_response_evaluations"] for row in runtime["segment_receipts"]
            ),
            baseline_responses=sum(
                row["quality_refiner"]["baseline_response_evaluations"] for row in runtime["segment_receipts"]
            ),
            gripper_sequences=sum(
                row["gripper_proposals"]["candidate_sequences"] for row in runtime["segment_receipts"]
            ),
            endpoint_queries=sum(
                row["quality_refiner"]["endpoint_queries_charged"] for row in runtime["segment_receipts"]
            ),
            transition_steps=sum(
                row["quality_refiner"]["transition_steps_charged"] for row in runtime["segment_receipts"]
            ),
        )
        maximum = config["max_control_steps_per_phase"]
        prefix.require(
            all(type(value) is int and value >= 0 for value in budget.values())
            and budget["quality_responses"] <= maximum * 320
            and budget["baseline_responses"] <= maximum * 480
            and budget["gripper_sequences"] <= maximum * 6561
            and budget["endpoint_queries"] <= maximum * 200
            and budget["transition_steps"] <= maximum * 2600,
            "Candidate work must remain within the original per-period limits",
        )
    errors = np.asarray([row["tcp_error"] for row in rows], dtype=float)
    prefix.require(errors.shape == (count, 6) and np.isfinite(errors).all(), "Finite observed TCP errors required")
    return dict(
        control_steps=count,
        native_steps=count * 25,
        models=config["max_models_per_phase"],
        contexts=config["max_contexts_per_phase"],
        prefix_steps=len(prior),
        new_reference_steps=len(new),
        replan_trigger=prefix_stop["trigger"],
        stop=stop,
        candidate_budget=budget,
        pose_caps_satisfied=sum(bool(row["pose_caps_satisfied"]) for row in rows),
        position_rmse_m=float(np.sqrt(np.mean(np.sum(errors[:, :3] ** 2, axis=1)))),
        rotation_rmse_rad=float(np.sqrt(np.mean(np.sum(errors[:, 3:] ** 2, axis=1)))),
        minimum_inner_margin_rad=float(min(row["minimum_inner_margin_rad"] for row in rows)),
        target_grasp_periods=sum(bool(row["target_grasp"]) for row in rows),
        task_success_periods=sum(bool(row["task_success"]) for row in rows),
        new_target_grasp_periods=sum(bool(row["target_grasp"]) for row in new),
        new_task_success_periods=sum(bool(row["task_success"]) for row in new),
    )


def summarize_phase(branches, phase, config):
    prefix.require(len(branches) == 1 and branches[0]["case"]["case_id"] == config["case_id"], "One case required")
    branch = branches[0]
    counts = summarize_branch(branch, phase, config)
    if phase == "checker":
        owner = prefix.read(prefix.reference(OUTPUT / "producer/result.json"))
        producer = prefix.read(owner["branch_refs"][0])
        produced, replayed = (item["runtime"]["continuation"] for item in (producer, branch))
        actual = continuation.prefix_periods(produced["prefix107"]) + tail.tail_periods(produced["replanned"])
        checked = continuation.prefix_periods(replayed["prefix107"], replay=True) + tail.tail_periods(
            replayed["replanned"]
        )
        counts["independent_max_absolute_difference"] = comparison.compare_periods(actual, checked)
        prior = owner["counts"]
        prefix.require(counts["control_steps"] == prior["control_steps"], "Producer and checker counts differ")
        counts.update(
            cumulative_control_steps=config["previous_control_steps"]
            + prior["control_steps"]
            + counts["control_steps"],
            cumulative_native_steps=config["previous_native_steps"] + prior["native_steps"] + counts["native_steps"],
            cumulative_models=config["previous_models"] + prior["models"] + counts["models"],
            cumulative_contexts=config["previous_contexts"] + prior["contexts"] + counts["contexts"],
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
        worker_module=MODULE,
        branch_key=original.previous.branch_key,
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
