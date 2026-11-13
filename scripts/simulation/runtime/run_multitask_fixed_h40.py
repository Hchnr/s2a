"""Run the same saved-current-image H40 comparison for each development case."""

import argparse
import sys
from pathlib import Path

import numpy as np

from s2a.infrastructure.native_environment import restore_cv2_environment
from s2a.infrastructure.publication import publish_result as publish
from scripts.simulation.runtime import audit_reset_multitask_live_recovery as comparison
from scripts.simulation.runtime import managed_physical_worker as managed
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import reset_adaptive_audit as audit
from scripts.simulation.runtime import reset_multitask_adaptive_admission as admission
from scripts.simulation.runtime import run_reset_start_feedback as original
from scripts.simulation.runtime import run_segmented_reset_equivalence as isolation

SOURCE_ROOT = Path(__file__).resolve().parents[3]
CONFIG = SOURCE_ROOT / "configs/tracking/references/reset_multitask_fixed_h40_20261113.json"
PLAN = SOURCE_ROOT / "plans/v0.0.1/tracking/references/reset_multitask_fixed_h40_20261113.md"
CONFIG_SHA = "30cc4759d98e4de37b43b07efea76f3196c6998baeb6ece2ccf1c2642401555e"
PLAN_SHA = "cf38d5de5816bcd568d16e83d96be14f6aaddabd6a7960f6b3bde3214968a4e1"
OUTPUT = prefix.REPO / "outputs/reset_multitask_fixed_h40_20261113"
REPLAY = prefix.REPO / "outputs/reset_multitask_postflight_replay_20261113"
ORIGINAL = prefix.REPO / "outputs/reset_multitask_adaptive_600_20261113"
QA = prefix.REPO / "outputs/qa/multitask_fixed_h40_reuse_20261113/result.json"
INDICES = (0, 2, 4, 6, 8, 10)


def load_case(index):
    """Admit an independently replayed source and its first real WAM reference."""
    prefix.environment()
    config = prefix.read({"path": str(CONFIG), "sha256": CONFIG_SHA})
    prefix.checked({"path": str(PLAN), "sha256": PLAN_SHA})
    prefix.require(
        index in INDICES
        and config["case_indices"] == list(INDICES)
        and config["sampling_seeds"] == [42, 43, 42, 43, 42, 43]
        and config["prefix_steps"] == 8
        and config["generated_horizon"] == 40
        and config["segment_steps"] == 8
        and config["max_control_steps_per_phase"] == 48
        and config["max_native_steps_per_phase"] == 1200
        and config["max_models_per_phase"] == config["max_contexts_per_phase"] == 3
        and config["new_model_initializations"] == config["new_model_requests"] == config["fresh_cases"] == 0
        and config["automatic_retry"] is False,
        "The common bounded H40 protocol is required",
    )
    qa = prefix.read(prefix.reference(QA))
    prefix.require(qa["passed"] is qa["source_unchanged"] is True, "The common fixed-reference checks must pass")
    posthoc_ref = prefix.reference(REPLAY / f"case_{index:02d}" / "result.json")
    posthoc = prefix.read(posthoc_ref)
    prefix.require(
        posthoc["complete"] is posthoc["passed"] is True
        and posthoc["original_producer_remains_failed"] is True
        and posthoc["new_model_requests"] == 0
        and posthoc["decision_stop_independently_recomputed"] is False,
        "The original action sequence must have its separate independent replay",
    )
    old_dir = ORIGINAL / f"case_{index:02d}" / "producer"
    old_ref = prefix.reference(old_dir / "result.json")
    owner = prefix.read(old_ref)
    registration = prefix.read(owner["registration_ref"])
    workers = prefix.read(owner["workers_ref"])
    prefix.require(
        posthoc["original_failure_ref"] == old_ref
        and posthoc["producer_ref"] == owner["available_branch_ref"]
        and posthoc["model_ref"] == owner["available_model_ref"]
        and owner["complete"] is owner["passed"] is False
        and owner["gpu_released"] is owner["partial_source_match"] is True
        and workers["passed"] is True
        and all(
            item["actual_wait"] is item["reaped"] is True and item["returncode"] == 0 for item in workers["workers"]
        ),
        "The unchanged original failed owner and independently replayed workers are required",
    )
    for ref in registration["source_refs"]:
        prefix.checked(ref)
    inputs = admission.load_inputs(index, verify_assets=False)
    trial = admission.make_trial(index)
    producer = prefix.read(owner["available_branch_ref"])
    model = prefix.read(owner["available_model_ref"])
    bundle = prefix.read(registration["model_bundle_ref"])
    prefix.require(
        registration["inputs"] == admission.input_receipt(inputs)
        and bundle == trial.model_bundle(inputs)
        and producer["case"]["case_index"] == index,
        "The exact reset and seed must match the original model bundle",
    )
    counts = audit.summarize_branch(producer, "producer", inputs["config"])
    exchange = audit.verify_model_exchange(producer, model, bundle)
    prefix.require(counts["model_requests"] == exchange["requests"] >= 1, "The actual step8 reference is required")
    runtime = producer["runtime"]
    capture, channel, row = (
        runtime["adaptive_captures"][0],
        runtime["adaptive_model_exchanges"][0],
        model["responses"][0],
    )
    prefix.require(capture["frame_index"] == channel["step"] == row["step"] == 8, "The reference must start at step8")
    response, envelope = prefix.read(row["actual_response_ref"]), prefix.read(channel["response_ref"])
    refs = dict(
        config=prefix.reference(CONFIG),
        plan=prefix.reference(PLAN),
        code=prefix.reference(Path(__file__).resolve()),
        qa=prefix.reference(QA),
        original_owner=old_ref,
        original_producer=owner["available_branch_ref"],
        original_model=owner["available_model_ref"],
        original_replay=posthoc_ref,
        response=row["actual_response_ref"],
        envelope=channel["response_ref"],
        image=capture["canvas_png"],
    )
    for ref in refs.values():
        prefix.checked(ref)
    return config, inputs, producer, capture, response, envelope, registration, refs


def periods(branch, phase):
    stream = branch["runtime"]["continuation"]
    first = stream["first_window"]["periods"]
    if phase == "producer":
        later = [row for segment in stream["fixed"]["segments"] for row in segment["continuation"]["periods"]]
        stop = stream["fixed"]["stop"]
    else:
        later = [row for segment in stream["replays"] for row in segment["periods"]]
        stop = stream["reported_producer_stop"]
    return first + later, stop


def summarize(branch, phase, config):
    rows, stop = periods(branch, phase)
    runtime, counters = branch["runtime"], branch["runtime"]["counters"]
    count = len(rows)
    prefix.require(
        branch["complete"] is branch["passed"] is runtime["complete"] is runtime["passed"] is True
        and runtime["cleanup_complete"] is runtime["reset_observation_verified"] is True
        and [row["step"] for row in rows] == list(range(count))
        and 8 <= count <= config["max_control_steps_per_phase"]
        and runtime["control_steps_attempted"] == runtime["control_steps_completed"] == count
        and counters["mj_step"]["completed"] == 25 * count
        and counters["model_construct"]["completed"] == counters["context"]["completed"] == 3
        and all(len(row["actual_inner"]) == 25 for row in rows),
        "Every actual reset period and bounded physical inner step must be accounted for",
    )
    errors = np.asarray([row["tcp_error"] for row in rows], dtype=float)
    opening = [row["completed_feedback"]["score"] for row in rows[8:]]
    return dict(
        control_steps=count,
        native_steps=25 * count,
        generated_periods=count - 8,
        complete_h40=count == 48 and stop is None,
        stop=stop,
        pose_caps_satisfied=sum(bool(row["pose_caps_satisfied"]) for row in rows),
        position_rmse_m=float(np.sqrt(np.mean(np.sum(errors[:, :3] ** 2, axis=1)))),
        rotation_rmse_rad=float(np.sqrt(np.mean(np.sum(errors[:, 3:] ** 2, axis=1)))),
        minimum_inner_margin_rad=float(min(row["minimum_inner_margin_rad"] for row in rows)),
        minimum_reference_opening=min((row["opening_reference"] for row in opening), default=None),
        minimum_measured_opening=min((row["opening"] for row in opening), default=None),
        target_grasp_periods=sum(bool(row["target_grasp"]) for row in rows),
        task_success_periods=sum(bool(row["task_success"]) for row in rows),
        final_task_success=rows[-1]["task_success"],
        stable_last40=False,
    )


def run_case(index, phase):
    """Run the registered producer or replay only its actual dispatched actions."""
    prefix.require(phase in ("producer", "checker"), "One physical phase is required")
    config, inputs, original_producer, capture, response, envelope, registration, refs = load_case(index)
    prior = None
    if phase == "checker":
        isolation.checker_isolation()
        previous = prefix.read(prefix.reference(OUTPUT / f"case_{index:02d}" / "producer/result.json"))
        prefix.require(previous["complete"] is previous["passed"] is True, "The fixed-reference producer must close")
        prior = prefix.read(previous["branch_ref"])
    original.require_no_simulator_import()
    output = OUTPUT / f"case_{index:02d}" / phase
    output.mkdir(parents=True, exist_ok=False)
    simulation = output / "simulation"
    simulation.mkdir()
    result = dict(complete=False, passed=False, phase=phase, case_index=index, automatic_retry=False)
    try:

        def execute():
            cv2 = inputs["base"]["cv2_admission"]
            cv2_receipt = restore_cv2_environment(cv2["cv2_probe_ref"], cv2["cv2_runtime_source_refs"])
            from s2a.research.native_runtime.reset_bounded_runtime import ResetBoundedRuntime

            case = inputs["case"]
            runtime = ResetBoundedRuntime(
                expected=case["expected"],
                expected_receipt=case["expected_receipt"],
                chunk=case["chunk"],
                controller=case["controller"],
                tool=case["tool"],
            )
            runtime_output = simulation / "runtime"
            if phase == "producer":
                from scripts.simulation.runtime import reset_bowl_fixed_h40 as fixed
            else:
                from scripts.simulation.runtime import reset_bowl_fixed_h40_replay as fixed

            def continuation(env, terminal, observers, receipt, *, progress=None):
                common = dict(
                    case=case,
                    expected_capture=capture,
                    load_reference=lambda: (response, envelope),
                    runtime_output=runtime_output,
                    progress=progress,
                )
                if phase == "producer":
                    return fixed.run_producer(
                        env, terminal, observers, receipt, baseline=inputs["baseline_periods"], **common
                    )
                return fixed.run_checker(env, terminal, observers, receipt, producer=prior, **common)

            actual = runtime.run_feedback(
                case["config"],
                case["record"],
                case["record_xml"],
                runtime_output,
                inputs["base"]["asset_refs"],
                registered_seed=inputs["config"]["seed"],
                continuation=continuation,
                max_control_steps=config["max_control_steps_per_phase"],
            )
            forbidden = tuple(isolation.FORBIDDEN) + (
                "s2a.research.joint_control.preview.causal_reference_replan",
                "s2a.research.wam.tracking.runtime.adaptive_stream_reference",
            )
            unloaded = all(name not in sys.modules for name in forbidden)
            prefix.require(phase != "checker" or unloaded, "The checker cannot load candidate policies")
            return dict(
                complete=True,
                passed=True,
                case=case["case"],
                runtime=actual,
                cv2_admission=cv2_receipt,
                candidate_modules_unloaded=unloaded,
                fresh_cases=0,
            )

        managed.run_managed_physical_worker(phase=phase, result_path=simulation / "result.json", execute=execute)
        branch_ref = prefix.reference(simulation / "result.json")
        branch = prefix.read(branch_ref)
        counts = summarize(branch, phase, config)
        if phase == "checker":
            p, _ = periods(prior, "producer")
            q, _ = periods(branch, "checker")
            result["independent_max_absolute_difference"] = comparison.compare_periods(p, q)
        for ref in (*refs.values(), *registration["source_refs"]):
            prefix.checked(ref)
        result.update(
            complete=True,
            passed=True,
            branch_ref=branch_ref,
            counts=counts,
            input_refs=refs,
            new_model_requests=0,
            fresh_cases=0,
        )
    except BaseException as error:
        result["error"] = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        print(publish(output / "result.json", result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-index", type=int, choices=INDICES, required=True)
    parser.add_argument("--phase", choices=("producer", "checker"))
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if args.preflight:
        config, inputs, *_rest = load_case(args.case_index)
        print(
            {
                "case_index": args.case_index,
                "seed": inputs["config"]["seed"],
                "max_steps": config["max_control_steps_per_phase"],
            }
        )
    elif args.phase:
        run_case(args.case_index, args.phase)
    else:
        parser.error("--phase or --preflight is required")


if __name__ == "__main__":
    main()
