"""Physically compare segmented reset feedback and replay actual actions."""

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from s2a.infrastructure.native_environment import restore_cv2_environment
from s2a.infrastructure.publication import publish_result as publish
from s2a.research.native_runtime.development_continuation import run_continuation
from scripts.simulation.runtime import registered_feedback_phase
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_reset_start_feedback as original

ROOT = prefix.REPO
CONFIG = ROOT / "configs/tracking/joint_control/segmented_reset_equivalence_20261111.json"
PLAN = ROOT / "plans/v0.0.1/tracking/joint_control/segmented_reset_equivalence_20261111.md"
QA = ROOT / "outputs/qa/segmented_reset_equivalence_20261111/result.json"
DRY = ROOT / "outputs/segmented_reset_equivalence_dry_20261111/result.json"
OUTPUT = ROOT / "outputs/segmented_reset_equivalence_20261111"
MODULE = "scripts.simulation.runtime.run_segmented_reset_equivalence"
BASELINE = {
    "path": str(ROOT / "outputs/reset_start_feedback_20261111/producer/result.json"),
    "sha256": "2b7fc9fa25241d938245e14bce7f485d9c887f21bb0110893dd212a128d2cc36",
}
BASELINE_AUDIT = {
    "path": str(ROOT / "outputs/reset_start_feedback_audit_20261111/result.json"),
    "sha256": "61606588d8298f915b5b043a9c1fc6d13f695c83dc96b51f64a63633ed339b97",
}
FORBIDDEN = original.CHECKER_FORBIDDEN + (
    "s2a.research.joint_control.preview.segmented_reset_policy",
    "s2a.research.joint_control.preview.segmented_reset_history",
    "s2a.research.joint_control.preview.streaming_feedback_boundary",
    "s2a.research.response_models.causal.segmented_inner_error",
)
BRANCHES = original.BRANCHES


def checker_isolation():
    prefix.require(all(name not in sys.modules for name in FORBIDDEN), "Checker must not import candidate policies")


def validate_config(config):
    expected = dict(
        schema="s2a.segmented-reset-equivalence.v1",
        case_indices=list(range(12)),
        baseline_owner_sha256=BASELINE["sha256"],
        continuation_steps=8,
        comparison_absolute_tolerance=1e-10,
        comparison_relative_tolerance=0.0,
        world_model_calls=0,
        fresh_cases=0,
        automatic_retry=False,
    )
    prefix.require(
        config == expected and config.get("automatic_retry") is False, "The fixed equivalence budget is required"
    )
    prefix.require(
        all(type(index) is int for index in config["case_indices"])
        and type(config["comparison_absolute_tolerance"]) is float
        and type(config["comparison_relative_tolerance"]) is float,
        "Indices and comparison tolerances must be explicit numerical types",
    )
    for key in ("continuation_steps", "world_model_calls", "fresh_cases"):
        prefix.require(type(config[key]) is int, "Budgets must be explicit integers")


def source_refs():
    return original.source_refs() + [
        prefix.reference(path)
        for path in (
            CONFIG,
            PLAN,
            Path(__file__).resolve(),
            ROOT / "src/s2a/research/joint_control/preview/segmented_reset_policy.py",
            ROOT / "src/s2a/research/joint_control/preview/segmented_reset_history.py",
            ROOT / "src/s2a/research/joint_control/preview/streaming_feedback_boundary.py",
            ROOT / "src/s2a/research/response_models/causal/segmented_inner_error.py",
            ROOT / "tests/research/native_runtime/test_segmented_reset_equivalence.py",
            ROOT / "tests/research/joint_control/preview/test_segmented_reset_policy.py",
            ROOT / "tests/research/joint_control/preview/test_segmented_reset_history.py",
        )
    ]


def load_inputs(*, verify_assets=True):
    inputs = original.load_inputs(verify_assets=verify_assets)
    config_ref = prefix.reference(CONFIG)
    config, baseline, audit = prefix.read(config_ref), prefix.read(BASELINE), prefix.read(BASELINE_AUDIT)
    validate_config(config)
    prefix.require(
        baseline["complete"] is baseline["passed"] is baseline["sources_unchanged"] is True
        and baseline["control_qualified"] is False
        and baseline["phase"] == "producer"
        and len(baseline["branch_refs"]) == 12
        and baseline["counts"]["totals"]["control_steps"] == 96
        and baseline["counts"]["totals"]["native_steps"] == 2400
        and baseline["counts"]["totals"]["short_window_passes"] == 12
        and audit["complete"] is audit["passed"] is audit["sources_unchanged"] is True
        and audit["producer_ref"] == BASELINE
        and audit["native_models"] == audit["physics_steps"] == 0
        and len(audit["cases"]) == 12,
        "The complete independently replayed original producer is required",
    )
    baseline_registration = prefix.read(baseline["registration_ref"])
    prefix.require(
        baseline_registration["inputs"] == original.input_receipt(inputs), "The same original reset cases are required"
    )
    for ref in baseline_registration["source_refs"]:
        prefix.checked(ref)
    for i, ref in enumerate(baseline["branch_refs"]):
        row = prefix.read(ref)
        prefix.require(
            row["complete"] is row["passed"] is True
            and row["case"] == inputs["cases"][i]["case"]
            and len(row["runtime"]["continuation"]["periods"]) == 8
            and row["runtime"]["continuation"]["stop"] is None,
            "Each original branch must contain eight completed periods",
        )
    for ref in (config_ref, BASELINE, BASELINE_AUDIT, baseline["registration_ref"], *baseline["branch_refs"]):
        inputs["source_sha256"][ref["path"]] = ref["sha256"]
    inputs.update(equivalence_config_ref=config_ref, original_branch_refs=baseline["branch_refs"])
    return inputs


def input_receipt(inputs):
    return dict(
        reset=original.input_receipt(inputs),
        config_ref=inputs["equivalence_config_ref"],
        baseline_ref=BASELINE,
        baseline_audit_ref=BASELINE_AUDIT,
        original_branch_refs=inputs["original_branch_refs"],
    )


def selected_command(decision, expected):
    """Refuse any changed command before an extra physical step is dispatched."""
    prefix.require(
        type(decision) is dict and "stop" not in decision and decision.get("newly_selected") is True,
        "The segmented candidate must select the original completed action",
    )
    np.testing.assert_array_equal(decision["command"], expected["command"])
    return decision


def summarize_phase(branches, phase, config):
    normalized = []
    for branch in branches:
        runtime = branch["runtime"]
        if phase == "producer":
            segments = runtime.get("segment_receipts")
            prefix.require(
                type(segments) is list
                and len(segments) == 1
                and runtime["segment_trace_starts"] == [dict(shared=0, causal=0)],
                "Exactly the first segmented policy must be executed",
            )
            runtime = dict(
                runtime,
                quality_refiner=segments[0]["quality_refiner"],
                gripper_proposals=segments[0]["gripper_proposals"],
            )
        normalized.append(dict(branch, runtime=runtime))
    totals = original.summarize_phase(normalized, phase, config)
    prefix.require(
        totals["totals"]["control_steps"] == 96 and totals["totals"]["short_window_passes"] == 12,
        "All original eight-period branches must be reproduced",
    )
    return totals


def dry_run():
    prefix.environment()
    inputs = load_inputs(verify_assets=True)
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
                world_model_calls=0,
                fresh_cases=0,
                automatic_retry=False,
            ),
        ),
        flush=True,
    )


def worker(phase, index, mode):
    directory = OUTPUT / phase / original.branch_key(index, mode)
    result = dict(
        complete=False,
        passed=False,
        control_qualified=False,
        phase=phase,
        case={"case_index": index},
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
        registration = prefix.read(prefix.reference(OUTPUT / phase / "registration.json"))
        inputs = load_inputs(verify_assets=False)
        prefix.require(
            registration["phase"] == phase
            and registration["inputs"] == input_receipt(inputs)
            and registration["source_refs"] == source_refs(),
            "Frozen branch input and source identity required",
        )
        case = inputs["cases"][index]
        baseline = prefix.read(inputs["original_branch_refs"][index])["runtime"]["continuation"]["periods"]
        if phase == "checker":
            checker_isolation()
            branch = prefix.read(registration["producer_branch_refs"][original.branch_key(index, mode)])
            prefix.require(branch["complete"] is branch["passed"] is True, "Closed segmented producer required")
            expected = branch["runtime"]["continuation"]["periods"]
        else:
            expected = baseline
        result["cv2_admission"] = restore_cv2_environment(
            inputs["cv2_admission"]["cv2_probe_ref"], inputs["cv2_admission"]["cv2_runtime_source_refs"]
        )
        original.require_no_simulator_import()
        from s2a.research.native_runtime.development_prefix_runtime import PrefixReplayFailure
        from s2a.research.native_runtime.reset_feedback_runtime import ResetFeedbackRuntime

        adapter = ResetFeedbackRuntime(
            expected=case["expected"],
            expected_receipt=case["expected_receipt"],
            chunk=case["chunk"],
            controller=case["controller"],
            tool=case["tool"],
        )

        def continuation(env, terminal, observers, receipt, *, progress=None):
            if phase == "producer":
                from s2a.research.joint_control.preview.reset_feedback_policy import policy_inputs
                from s2a.research.joint_control.preview.segmented_reset_policy import build_segmented_reset_policy

                policy, _owner, previous = build_segmented_reset_policy(env, policy_inputs(case, terminal), receipt)

                def action(q, limits, jacobian, prior, offset):
                    start = time.monotonic()
                    decision = policy.select(q, limits, jacobian, prior, offset)
                    decision["total_decision_seconds"] = time.monotonic() - start
                    return selected_command(decision, expected[offset])
            else:
                checker_isolation()
                policy, previous, action = None, np.zeros(7), original.checker_action(expected)

            def completed(actual_env, command, offset):
                period = receipt["continuation_partial"]["periods"][-1]
                history = (
                    None
                    if policy is None
                    else policy.complete(
                        actual_env, command, offset, [sample["q"] for sample in period["actual_inner"]]
                    )
                )
                score = original.measured_gripper(actual_env, case["chunk"]["openings"][offset])
                original.compare_score(score, expected[offset]["completed_feedback"]["score"])
                return dict(score=score, policy_history=history, policy_history_recomputed=policy is not None)

            return run_continuation(
                env,
                terminal,
                observers,
                receipt,
                start_step=0,
                previous_command=previous,
                targets=np.asarray(case["chunk"]["effective_targets_tcp"][:8]),
                tool=case["tool"],
                target_object=case["target_object"],
                choose_action=action,
                stop_on_outside=True,
                expected_periods=expected,
                on_completed=completed,
                progress=progress,
            )

        try:
            runtime = adapter.run_feedback(
                case["config"],
                case["record"],
                case["record_xml"],
                directory / "runtime",
                inputs["asset_refs"],
                registered_seed=case["case"]["sampling_seed"],
                continuation=continuation,
            )
        except PrefixReplayFailure as error:
            result["runtime"] = error.receipt
            raise
        result.update(
            complete=True,
            passed=True,
            case=case["case"],
            runtime=runtime,
            candidate_modules_unloaded=all(name not in sys.modules for name in FORBIDDEN),
        )
        prefix.require(phase != "checker" or result["candidate_modules_unloaded"] is True, "Checker isolation required")
    except BaseException as error:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        print(publish(directory / "result.json", result), flush=True)


def execute(phase):
    qa = prefix.read(prefix.reference(QA))
    prefix.require(
        qa["all_tests_passed"] is True
        and qa["configured_parallelism"] == 32
        and qa["skipped_count"] == qa["xfail_count"] == qa["xpass_count"] == 0
        and not qa["collection"]["collection_errors"],
        "Complete source-matched CPU QA required",
    )
    registered_feedback_phase.execute_phase(
        phase,
        root=OUTPUT,
        qa_path=QA,
        dry_path=DRY,
        branches=BRANCHES,
        worker_module=MODULE,
        branch_key=original.branch_key,
        load_inputs=load_inputs,
        input_receipt=input_receipt,
        source_refs=source_refs,
        summarize_phase=summarize_phase,
        config_key="feedback_config",
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
