"""Run one reset-origin correct-reference control and independently replay it."""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

from s2a.infrastructure.native_environment import restore_cv2_environment
from s2a.infrastructure.owned_processes import run_owned_workers
from s2a.infrastructure.publication import publish_result, result_json
from s2a.references.oracle_timeline import OracleTimeline
from s2a.research.native_runtime.correct_reference_continuation import run_correct_reference_continuation
from s2a.research.native_runtime.development_prefix_replay import SAMPLE_FIELDS
from scripts.maintenance.run_cpu_tests import snapshot
from scripts.simulation.runtime import audit_reset_multitask_live_recovery as comparison
from scripts.simulation.runtime import managed_physical_worker as managed
from scripts.simulation.runtime import prepare_ordered_correct_stream as prior
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_reset_start_feedback as original

ROOT = prefix.REPO
PLAN = ROOT / "plans/v0.0.1/tracking/joint_control/reset_correct_reference_20261116.md"
OUTPUT = ROOT / "outputs/reset_correct_reference_20261116"
DRY = ROOT / "outputs/reset_correct_reference_dry_20261116.json"
QA = ROOT / "outputs/qa/reset_correct_reference_final_20261116/result.json"
MODULE = "scripts.simulation.runtime.run_reset_correct_reference"
PERIODS = 182
TARGET_OBJECT = "cream_cheese_1"
BASELINE = {
    "path": str(ROOT / "outputs/tracking_correct_reference_ordered_20261027/case08_candidate/result.json"),
    "sha256": "885d02679fcdf75ab3067de9b1557cff4a5cb0c5671ca149aa9e0afef3da0da0",
}
BASELINE_CHECK = {
    "path": str(ROOT / "outputs/tracking_correct_reference_ordered_check_20261027/case08/verified.json"),
    "sha256": "575ac7b623416a00568057e36baa11f5ab898ce909950a263bd11f7363a264af",
}


def source_refs():
    return (
        prior.source_refs()
        + original.source_refs()
        + [
            prefix.reference(path)
            for path in (
                Path(__file__).resolve(),
                PLAN,
                ROOT / "src/s2a/references/oracle_timeline.py",
                ROOT / "src/s2a/research/native_runtime/correct_reference_reset.py",
                ROOT / "src/s2a/research/native_runtime/correct_reference_continuation.py",
                ROOT / "scripts/simulation/runtime/managed_physical_worker.py",
                ROOT / "tests/unit/references/test_oracle_timeline.py",
                ROOT / "tests/research/native_runtime/test_correct_reference_reset.py",
                ROOT / "tests/research/native_runtime/test_correct_reference_continuation.py",
            )
        ]
    )


def load_inputs():
    """Bind correct targets and reset only; recorded actions remain scoring data."""
    data = prior.load_inputs()
    old = data["previous"]
    baseline, checked = prefix.read(BASELINE), prefix.read(BASELINE_CHECK)
    prefix.require(
        baseline["complete"] is baseline["passed"] is baseline["performance_qualified"] is True
        and baseline["case_index"] == 8
        and baseline["physics_steps"] == PERIODS
        and checked["complete"] is checked["passed"] is checked["performance_qualified"] is True
        and checked["result_ref"] == BASELINE
        and checked["physics_steps"] == PERIODS,
        "The original correct-reference control and its independent replay must be closed",
    )
    rows, chunks = prefix.read(data["normal_refs"]["rows"]), prefix.read(data["normal_refs"]["chunks"])
    prefix.require(
        len(rows) == PERIODS
        and all(row["step"] == index for index, row in enumerate(rows))
        and all(row["success"] is True for row in rows[-40:])
        and old["runtime_config"]["task_id"] == 6
        and old["runtime_config"]["seed"] == 42,
        "The original complete case and its 40 terminal success labels are required",
    )
    timeline = OracleTimeline(chunks, periods=PERIODS)
    first_targets, first_openings, _metadata = timeline.window(0)
    expected = {name: np.asarray(old["saved"][name][0]).copy() for name in (*SAMPLE_FIELDS, "gripper_current_action")}
    anchor = np.asarray(chunks[0]["anchor_tcp"], dtype=float)
    np.testing.assert_allclose(np.asarray(expected["source_pose"]) @ old["tool"], anchor, atol=1e-10, rtol=0)
    np.testing.assert_allclose(float(expected["opening"]), rows[0]["opening_before"], atol=1e-10, rtol=0)
    prefix.require(
        old["receipts"][0]["sim_state_sha256"] == old["controller"]["actual_initial_physics_sha256"],
        "The correct-reference anchor and controller must use the recorded reset",
    )
    case = dict(
        chunk=dict(anchor_tcp=anchor, effective_targets_tcp=first_targets, openings=first_openings),
        controller=old["controller"],
        tool=old["tool"],
        gripper_parameters=data["parameters"],
    )
    reset_input = dict(
        expected=expected, expected_receipt=old["receipts"][0], anchor=anchor, opening=rows[0]["opening_before"]
    )
    identity = dict(
        source_receipt=prior.input_receipt(data),
        baseline_ref=BASELINE,
        baseline_checker_ref=BASELINE_CHECK,
        reference_ref=data["normal_refs"]["chunks"],
        scoring_rows_ref=data["normal_refs"]["rows"],
        reset_sha256=old["receipts"][0]["sim_state_sha256"],
        reset_input_sha256=hashlib.sha256(result_json(reset_input).encode()).hexdigest(),
        oracle_periods=PERIODS,
        oracle_windows=timeline.window_count,
        overlap_pose_max_difference=timeline.overlap_pose_max_difference,
        model_requests=0,
        historical_actions_as_input=False,
        future_actual_states_as_input=False,
    )
    return dict(base=data, old=old, case=case, timeline=timeline, reset_input=reset_input, identity=identity)


def dry_run():
    prefix.environment()
    original.require_no_simulator_import()
    inputs = load_inputs()
    original.checker_isolation()
    print(
        publish_result(
            DRY,
            dict(
                complete=True,
                passed=True,
                identity=inputs["identity"],
                source_refs=source_refs(),
                physics_steps=0,
                world_model_calls=0,
                fresh_cases=0,
            ),
        ),
        flush=True,
    )


def progress_event(phase, receipt):
    return dict(
        event="physical_progress",
        pid=os.getpid(),
        phase=phase,
        control_steps_attempted=receipt.get("control_steps_attempted"),
        control_steps_completed=receipt.get("control_steps_completed"),
        newly_selected_commands=receipt.get("newly_selected_commands"),
        native_steps=receipt.get("counters", {}).get("mj_step"),
        models=receipt.get("counters", {}).get("model_construct"),
        contexts=receipt.get("counters", {}).get("context"),
    )


def worker(phase):
    prefix.require(phase in ("producer", "checker"), "One registered physical phase is required")
    print(json.dumps(dict(event="worker_started", pid=os.getpid(), phase=phase)), flush=True)
    directory = OUTPUT / phase / "worker"
    directory.mkdir(parents=True, exist_ok=False)
    registration = prefix.read(prefix.reference(OUTPUT / phase / "registration.json"))

    def execute():
        inputs = load_inputs()
        prefix.require(
            registration["phase"] == phase
            and registration["identity"] == inputs["identity"]
            and registration["source_refs"] == source_refs(),
            "The registered physical inputs must remain fixed",
        )
        saved = None
        if phase == "checker":
            original.checker_isolation()
            produced = prefix.read(registration["producer_branch_ref"])
            prefix.require(produced["complete"] is produced["passed"] is True, "A closed producer is required")
            saved = produced["runtime"]["continuation"]
        old, case, reset_input = inputs["old"], inputs["case"], inputs["reset_input"]
        cv2 = old["cv2_admission"]
        cv2_receipt = restore_cv2_environment(cv2["cv2_probe_ref"], cv2["cv2_runtime_source_refs"])
        original.require_no_simulator_import()
        from s2a.research.native_runtime.correct_reference_reset import CorrectReferenceResetRuntime

        adapter = CorrectReferenceResetRuntime(
            expected=reset_input["expected"],
            expected_receipt=reset_input["expected_receipt"],
            controller=old["controller"],
            tool=old["tool"],
            anchor=reset_input["anchor"],
            opening=reset_input["opening"],
        )
        factory = None
        if phase == "producer":
            from s2a.research.joint_control.preview.reset_feedback_policy import policy_inputs
            from s2a.research.joint_control.preview.segmented_reset_policy import build_segmented_reset_policy

            def factory(env, terminal, receipt):
                return build_segmented_reset_policy(env, policy_inputs(case, terminal), receipt)

        def continuation(env, terminal, observers, receipt, *, progress=None):
            return run_correct_reference_continuation(
                env,
                terminal,
                observers,
                receipt,
                timeline=inputs["timeline"],
                tool=old["tool"],
                target_object=TARGET_OBJECT,
                phase=phase,
                score_gripper=original.measured_gripper,
                compare_score=original.compare_score,
                policy_factory=factory,
                recorded=saved,
                progress=progress,
            )

        def progress(receipt):
            print(json.dumps(progress_event(phase, receipt)), flush=True)

        runtime = adapter.run_feedback(
            old["runtime_config"],
            old["record"],
            old["record_xml"],
            directory / "runtime",
            old["asset_refs"],
            registered_seed=42,
            continuation=continuation,
            max_control_steps=PERIODS,
            progress=progress,
        )
        if phase == "checker":
            original.checker_isolation()
        return dict(
            complete=True,
            passed=True,
            runtime=runtime,
            cv2_admission=cv2_receipt,
            candidate_modules_unloaded=phase == "checker",
            fresh_cases=0,
        )

    managed.run_managed_physical_worker(phase=phase, result_path=directory / "result.json", execute=execute)


def periods(branch):
    return [
        period for row in branch["runtime"]["continuation"]["segments"] for period in row["continuation"]["periods"]
    ]


def validate_branch(branch, phase):
    runtime = branch["runtime"]
    rows = periods(branch)
    prefix.require(
        branch["complete"] is branch["passed"] is runtime["complete"] is runtime["passed"] is True
        and runtime["reset_observation_verified"] is True
        and runtime["prefix_control_steps"] == 0
        and runtime["world_model_calls"] == branch["fresh_cases"] == 0
        and runtime["control_steps_attempted"] == runtime["control_steps_completed"] == len(rows) <= PERIODS
        and runtime["counters"]["mj_step"]["attempted"] == runtime["counters"]["mj_step"]["completed"] == 25 * len(rows)
        and runtime["counters"]["model_construct"]["completed"] == runtime["counters"]["context"]["completed"] == 3
        and runtime["cleanup_complete"] is True
        and runtime["continuation"]["summary"]["completed_periods"] == len(rows),
        "The physical branch must close with its exact reset and period counts",
    )
    if phase == "producer":
        prefix.require(runtime["newly_selected_commands"] == len(rows), "Every producer action must be newly selected")
    else:
        prefix.require(
            branch["candidate_modules_unloaded"] is True and runtime["newly_selected_commands"] == 0,
            "The checker must only replay, without candidate modules",
        )
    return dict(control_steps=len(rows), native_steps=25 * len(rows), summary=runtime["continuation"]["summary"])


def phase_run(phase):
    prefix.environment()
    prefix.require(phase in ("producer", "checker"), "One physical phase is required")
    qa_ref, dry_ref = prefix.reference(QA), prefix.reference(DRY)
    qa, dry = prefix.read(qa_ref), prefix.read(dry_ref)
    prefix.require(
        qa["complete"] is qa["passed"] is qa["source_unchanged"] is qa["actual_wait"] is qa["reaped"] is True
        and dry["complete"] is dry["passed"] is True
        and dry["physics_steps"] == 0,
        "The full selected CPU QA and zero-physics preflight must pass",
    )
    before = snapshot(ROOT)["sources"]
    prefix.require(before == prefix.read(qa["source_before_ref"])["sources"], "CPU QA must bind current sources")
    data, sources = load_inputs(), source_refs()
    prefix.require(dry["identity"] == data["identity"] and dry["source_refs"] == sources, "Dry identities must match")
    directory = OUTPUT / phase
    prefix.require(not directory.exists(), "A fresh physical phase directory is required")
    producer_ref = None
    if phase == "checker":
        previous = prefix.read(prefix.reference(OUTPUT / "producer/result.json"))
        prefix.require(previous["complete"] is previous["passed"] is True, "A completed producer is required")
        producer_ref = previous["branch_ref"]
        prefix.checked(producer_ref)
    registration_ref = publish_result(
        directory / "registration.json",
        dict(
            phase=phase,
            owner_pid=os.getpid(),
            identity=data["identity"],
            source_refs=sources,
            qa_ref=qa_ref,
            dry_ref=dry_ref,
            producer_branch_ref=producer_ref,
            max_control_steps=PERIODS,
            max_native_steps=25 * PERIODS,
            automatic_retry=False,
        ),
    )
    specs = [
        dict(key="correct08", argv=[sys.executable, "-m", MODULE, "--worker", phase], log=directory / "worker.log")
    ]
    lifecycle = run_owned_workers(specs, cwd=ROOT, timeout_seconds=3600)
    for item in lifecycle["workers"]:
        item["log_ref"] = prefix.reference(item.pop("log"))
    result = dict(
        complete=False,
        passed=False,
        phase=phase,
        registration_ref=registration_ref,
        lifecycle=lifecycle,
        automatic_retry=False,
    )
    try:
        prefix.require(
            lifecycle["launch_error"] is None
            and len(lifecycle["workers"]) == 1
            and lifecycle["workers"][0]["returncode"] == 0
            and lifecycle["workers"][0]["actual_wait"] is lifecycle["workers"][0]["reaped"] is True,
            "The owned physical worker must exit and be reaped",
        )
        branch_ref = prefix.reference(directory / "worker/result.json")
        branch = prefix.read(branch_ref)
        counts = validate_branch(branch, phase)
        if phase == "checker":
            producer = prefix.read(producer_ref)
            counts["replay_max_difference"] = comparison.compare_periods(periods(producer), periods(branch))
            prefix.require(
                producer["runtime"]["continuation"]["stop"] == branch["runtime"]["continuation"]["stop"],
                "The reported producer stop must be retained",
            )
        prefix.require(
            snapshot(ROOT)["sources"] == before and load_inputs()["identity"] == data["identity"],
            "Sources and inputs changed during the physical phase",
        )
        result.update(
            complete=True,
            passed=True,
            sources_unchanged=True,
            branch_ref=branch_ref,
            counts=counts,
            world_model_calls=0,
            fresh_cases=0,
        )
    except BaseException as error:
        result["error"] = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        print(publish_result(directory / "result.json", result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry", action="store_true")
    group.add_argument("--phase", choices=("producer", "checker"))
    group.add_argument("--worker", choices=("producer", "checker"))
    args = parser.parse_args()
    if args.dry:
        dry_run()
    elif args.phase:
        phase_run(args.phase)
    else:
        worker(args.worker)


if __name__ == "__main__":
    main()
