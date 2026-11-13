"""Independently replay actual actions after a bounded postflight audit failure."""

import argparse
import copy
import sys
from pathlib import Path

from s2a.infrastructure.native_environment import restore_cv2_environment
from s2a.infrastructure.publication import publish_result as publish
from scripts.simulation.runtime import adaptive_live_bowl_worker as historical
from scripts.simulation.runtime import managed_physical_worker as managed
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import reset_adaptive_audit as audit
from scripts.simulation.runtime import reset_multitask_adaptive_admission as admission
from scripts.simulation.runtime import run_reset_start_feedback as original
from scripts.simulation.runtime import run_segmented_reset_equivalence as isolation

PRELIMINARY = {
    "path": str(prefix.REPO / "outputs/reset_multitask_adaptive_preliminary_20261113/result.json"),
    "sha256": "7b2caee3292ef3d2726e701c4ba198b1d5e4758067071536e5550c2c278b7125",
}
OUTPUT = prefix.REPO / "outputs/reset_multitask_postflight_replay_20261113"
INDICES = (0, 2, 4, 6)


def admit(index):
    """Recheck the failed owner, saved model exchange, inputs, and executed budget."""
    prefix.environment()
    prefix.require(index in INDICES, "One registered earlier case is required")
    previous = prefix.read(PRELIMINARY)
    prefix.require(
        previous["complete"] is previous["original_producer_results_remain_failed"] is True
        and previous["fixed_verifier_ref"] == prefix.reference(Path(audit.__file__).resolve())
        and [row["case_index"] for row in previous["cases"]] == list(INDICES),
        "The exact postflight-only analysis and corrected verifier are required",
    )
    row = next(item for item in previous["cases"] if item["case_index"] == index)
    owner = prefix.read(row["original_failure_ref"])
    registration = prefix.read(row["original_registration_ref"])
    lifecycle = prefix.read(row["workers_ref"])
    prefix.require(
        owner["complete"] is owner["passed"] is False
        and owner["partial_source_match"] is owner["gpu_released"] is True
        and owner["error"]
        == {"type": "ValueError", "message": "The model must receive the producer's exact bounded stop"}
        and owner["registration_ref"] == row["original_registration_ref"]
        and owner["workers_ref"] == row["workers_ref"]
        and owner["available_branch_ref"] == row["producer_ref"]
        and owner["available_model_ref"] == row["model_ref"]
        and registration["model_bundle_ref"] == row["bundle_ref"]
        and lifecycle["passed"] is True
        and lifecycle["launch_error"] is None
        and lifecycle["timed_out"] is lifecycle["interrupted"] is False
        and {item["name"] for item in lifecycle["workers"]} == {"model", "simulation"}
        and all(
            item["actual_wait"] is item["reaped"] is True and item["returncode"] == 0 and item["signal_sent"] is False
            for item in lifecycle["workers"]
        ),
        "The original postflight failure and both cleanly closed workers are required",
    )
    for ref in registration["source_refs"]:
        prefix.checked(ref)
    for item in lifecycle["workers"]:
        prefix.checked(item["log_ref"])
    inputs = admission.load_inputs(index, verify_assets=False)
    trial = admission.make_trial(index)
    producer, model, bundle = (prefix.read(row[key]) for key in ("producer_ref", "model_ref", "bundle_ref"))
    prefix.require(
        registration["inputs"] == admission.input_receipt(inputs)
        and bundle == trial.model_bundle(inputs)
        and producer["case"]["case_id"] == row["case_id"],
        "Reset inputs and the model bundle must equal their original registration",
    )
    counts = audit.summarize_branch(producer, "producer", inputs["config"])
    exchange = audit.verify_model_exchange(producer, model, bundle)
    prefix.require(
        all(counts[key] == value for key, value in row["counts"].items())
        and counts["stop"]["reason"] == row["reported_stop_reason"]
        and exchange == row["model_exchange"],
        "The corrected producer analysis must still match all saved evidence",
    )
    return inputs, producer, model, bundle, row


def replay(index):
    """Replay recorded actions only, then compare every actual endpoint and image."""
    inputs, producer, model, bundle, row = admit(index)
    responses = historical.load_recorded_responses(producer, model)
    isolation.checker_isolation()
    original.require_no_simulator_import()
    directory = OUTPUT / f"case_{index:02d}" / "simulation"
    directory.mkdir(parents=True, exist_ok=False)

    def execute():
        cv2 = inputs["base"]["cv2_admission"]
        cv2_receipt = restore_cv2_environment(cv2["cv2_probe_ref"], cv2["cv2_runtime_source_refs"])
        from s2a.research.native_runtime.reset_bounded_runtime import ResetBoundedRuntime
        from scripts.simulation.runtime import reset_adaptive_stream as stream

        case = inputs["case"]
        runtime = ResetBoundedRuntime(
            expected=case["expected"],
            expected_receipt=case["expected_receipt"],
            chunk=case["chunk"],
            controller=case["controller"],
            tool=case["tool"],
        )
        runtime_output = directory / "runtime"
        registration = copy.deepcopy(bundle["model_registration"])
        registration["image_root"] = str(directory.relative_to(prefix.REPO))

        def continuation(env, terminal, observers, receipt, *, progress=None):
            return stream.run_checker(
                env,
                terminal,
                observers,
                receipt,
                case=case,
                producer=producer,
                responses=responses,
                registration=registration,
                runtime_output=runtime_output,
                progress=progress,
            )

        actual = runtime.run_feedback(
            case["config"],
            case["record"],
            case["record_xml"],
            runtime_output,
            inputs["base"]["asset_refs"],
            registered_seed=inputs["config"]["seed"],
            continuation=continuation,
            max_control_steps=inputs["config"]["max_control_steps_per_phase"],
        )
        forbidden = tuple(isolation.FORBIDDEN) + (
            "s2a.research.joint_control.preview.causal_reference_replan",
            "s2a.research.wam.tracking.runtime.adaptive_stream_reference",
        )
        unloaded = all(name not in sys.modules for name in forbidden)
        prefix.require(unloaded, "The independent checker must not load candidate policies")
        return dict(
            complete=True,
            passed=True,
            case=case["case"],
            runtime=actual,
            cv2_admission=cv2_receipt,
            candidate_modules_unloaded=unloaded,
            fresh_cases=0,
        )

    managed.run_managed_physical_worker(phase="checker", result_path=directory / "result.json", execute=execute)
    checker_ref = prefix.reference(directory / "result.json")
    checker = prefix.read(checker_ref)
    counts = audit.summarize_branch(checker, "checker", inputs["config"])
    comparison = audit.compare_branches(producer, checker)
    prefix.require(counts["control_steps"] == row["counts"]["control_steps"], "Every actual command must replay")
    registration = prefix.read(row["original_registration_ref"])
    for ref in registration["source_refs"]:
        prefix.checked(ref)
    result = dict(
        complete=True,
        passed=True,
        control_qualified=False,
        original_producer_remains_failed=True,
        preliminary_ref=PRELIMINARY,
        original_failure_ref=row["original_failure_ref"],
        producer_ref=row["producer_ref"],
        model_ref=row["model_ref"],
        checker_ref=checker_ref,
        script_ref=prefix.reference(Path(__file__).resolve()),
        fixed_verifier_ref=prefix.reference(Path(audit.__file__).resolve()),
        checker_counts=counts,
        independent_max_absolute_difference=comparison,
        decision_stop_independently_recomputed=False,
        new_model_requests=0,
        fresh_cases=0,
        automatic_retry=False,
    )
    return publish(directory.parent / "result.json", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-index", type=int, choices=INDICES, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if args.preflight:
        inputs, _producer, _model, _bundle, row = admit(args.case_index)
        print(
            {"case_index": args.case_index, "steps": row["counts"]["control_steps"], "seed": inputs["config"]["seed"]}
        )
    else:
        print(replay(args.case_index), flush=True)


if __name__ == "__main__":
    main()
