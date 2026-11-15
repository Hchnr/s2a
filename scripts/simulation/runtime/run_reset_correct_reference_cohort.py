"""Run and independently replay the fixed reset-origin correct-reference cohort."""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

from s2a.infrastructure.native_environment import restore_cv2_environment
from s2a.infrastructure.owned_processes import run_owned_workers
from s2a.infrastructure.publication import publish_result
from s2a.research.native_runtime.correct_reference_continuation import run_correct_reference_continuation
from s2a.research.native_runtime.correct_reference_reset import CorrectReferenceResetRuntime
from scripts.maintenance.run_cpu_tests import snapshot
from scripts.simulation.runtime import audit_reset_multitask_live_recovery as comparison
from scripts.simulation.runtime import managed_physical_worker as managed
from scripts.simulation.runtime import prepare_reset_correct_reference_cohort as preparation
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_reset_correct_reference as previous
from scripts.simulation.runtime import run_reset_start_feedback as original

ROOT = prefix.REPO
OUTPUT = ROOT / "outputs/reset_correct_reference_cohort_20261116"
DRY = ROOT / "outputs/reset_correct_reference_cohort_dry_20261116.json"
QA = ROOT / "outputs/qa/reset_correct_reference_cohort_final_20261116/result.json"
MODULE = "scripts.simulation.runtime.run_reset_correct_reference_cohort"


def source_refs():
    return (
        previous.source_refs()
        + preparation.cohort.source_refs()
        + [
            prefix.reference(path)
            for path in (
                preparation.CONFIG,
                preparation.PLAN,
                Path(preparation.__file__).resolve(),
                Path(__file__).resolve(),
                ROOT / "scripts/verification/reference_ports/audit_reset_correct_reference.py",
                ROOT / "tests/research/native_runtime/test_reset_correct_reference_cohort_inputs.py",
                ROOT / "tests/research/native_runtime/test_reset_correct_reference_cohort.py",
            )
        ]
    )


def dry_run():
    prefix.environment()
    original.require_no_simulator_import()
    inputs = preparation.load_inputs()
    original.checker_isolation()
    print(
        publish_result(
            DRY,
            dict(
                complete=True,
                passed=True,
                inputs=preparation.input_receipt(inputs),
                source_refs=source_refs(),
                physics_steps=0,
                world_model_calls=0,
                fresh_cases=0,
            ),
        ),
        flush=True,
    )


def validate_branch(branch, phase, expected_periods):
    """Verify actual physical counts without treating an early stop as success."""
    runtime = branch["runtime"]
    rows = previous.periods(branch)
    summary = runtime["continuation"]["summary"]
    prefix.require(
        branch["complete"] is branch["passed"] is runtime["complete"] is runtime["passed"] is True
        and runtime["reset_observation_verified"] is True
        and runtime["prefix_control_steps"] == 0
        and runtime["world_model_calls"] == branch["fresh_cases"] == 0
        and runtime["control_steps_attempted"] == runtime["control_steps_completed"] == len(rows) <= expected_periods
        and runtime["counters"]["mj_step"]["attempted"] == runtime["counters"]["mj_step"]["completed"] == 25 * len(rows)
        and runtime["counters"]["model_construct"]["completed"] == runtime["counters"]["context"]["completed"] == 3
        and runtime["cleanup_complete"] is True
        and summary["completed_periods"] == len(rows)
        and summary["expected_periods"] == expected_periods,
        "A closed reset-origin branch with exact physical counts is required",
    )
    if phase == "producer":
        prefix.require(runtime["newly_selected_commands"] == len(rows), "Every producer action must be newly selected")
    else:
        prefix.require(
            branch["candidate_modules_unloaded"] is True and runtime["newly_selected_commands"] == 0,
            "The checker must replay without loading or selecting the candidate",
        )
    return dict(control_steps=len(rows), native_steps=25 * len(rows), summary=summary)


def qualifies(summary, replayed, config):
    """Apply fixed full-length, replay, success, error and joint-buffer gates."""
    if not replayed or not isinstance(summary, dict):
        return False
    values = (summary.get("position_rmse_m"), summary.get("rotation_rmse_rad"), summary.get("minimum_inner_margin_rad"))
    if any(value is None or not np.isfinite(value) for value in values):
        return False
    return bool(
        summary.get("full_trajectory") is True
        and summary.get("completed_periods") == summary.get("expected_periods")
        and summary.get("stable_last_40") is True
        and summary["position_rmse_m"] <= config["position_cap_m"]
        and summary["rotation_rmse_rad"] <= config["rotation_cap_rad"]
        and summary["minimum_inner_margin_rad"] >= config["joint_buffer_rad"]
        and summary.get("inner_below_buffer_samples") == 0
        and summary.get("inner_outside_samples") == 0
    )


def checker_eligible(producer, cases, reused):
    """Preserve missing/failed producers as failures while replaying closed peers."""
    return [
        item["spec"]["index"]
        for item in cases
        if item["spec"]["index"] != reused
        and producer["cases"].get(str(item["spec"]["index"]), {}).get("status") == "closed"
    ]


def worker(phase, index):
    prefix.require(
        phase in ("producer", "checker") and type(index) is int and 0 <= index < 18 and index != 8,
        "One new registered cohort case and phase are required",
    )
    print(json.dumps(dict(event="worker_started", pid=os.getpid(), phase=phase, case_index=index)), flush=True)
    directory = OUTPUT / phase / f"case{index:02d}" / "worker"
    directory.mkdir(parents=True, exist_ok=False)
    registration = prefix.read(prefix.reference(OUTPUT / phase / "registration.json"))

    def execute():
        prefix.require(
            registration["phase"] == phase
            and index in registration["eligible_cases"]
            and registration["source_refs"] == source_refs(),
            "The registered phase, case and source identities must remain fixed",
        )
        shared = preparation.shared_inputs()
        prefix.require(shared["receipt"] == registration["inputs"]["shared_receipt"], "Shared assets must match")
        identity = registration["inputs"]["cases"][index]
        inputs = preparation.case_inputs(identity, shared)
        prefix.require(
            inputs["reset_input_sha256"] == identity["reset_input_sha256"]
            and inputs["target_parity"] == identity["target_parity"],
            "The registered reset and reference must match",
        )
        recorded = None
        if phase == "checker":
            original.checker_isolation()
            produced = prefix.read(registration["producer_branch_refs"][str(index)])
            validate_branch(produced, "producer", inputs["spec"]["periods"])
            recorded = produced["runtime"]["continuation"]
        cv2 = shared["cv2_admission"]
        cv2_receipt = restore_cv2_environment(cv2["cv2_probe_ref"], cv2["cv2_runtime_source_refs"])
        original.require_no_simulator_import()
        reset, case = inputs["reset"], inputs["case"]
        adapter = CorrectReferenceResetRuntime(
            expected=reset["expected"],
            expected_receipt=reset["expected_receipt"],
            controller=case["controller"],
            tool=case["tool"],
            anchor=reset["anchor"],
            opening=reset["opening"],
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
                tool=case["tool"],
                target_object=inputs["target_object"],
                phase=phase,
                score_gripper=original.measured_gripper,
                compare_score=original.compare_score,
                policy_factory=factory,
                recorded=recorded,
                progress=progress,
            )

        def progress(receipt):
            print(json.dumps(dict(case_index=index, **previous.progress_event(phase, receipt))), flush=True)

        runtime = adapter.run_feedback(
            inputs["runtime_config"],
            inputs["record"],
            inputs["record_xml"],
            directory / "runtime",
            shared["asset_refs"],
            registered_seed=inputs["spec"]["seed"],
            continuation=continuation,
            max_control_steps=inputs["spec"]["periods"],
            progress=progress,
        )
        if phase == "checker":
            original.checker_isolation()
        return dict(
            complete=True,
            passed=True,
            case=inputs["spec"],
            runtime=runtime,
            cv2_admission=cv2_receipt,
            candidate_modules_unloaded=phase == "checker",
            fresh_cases=0,
        )

    managed.run_managed_physical_worker(phase=phase, result_path=directory / "result.json", execute=execute)


def phase_run(phase):
    prefix.environment()
    prefix.require(phase in ("producer", "checker"), "One registered cohort phase is required")
    qa_ref, dry_ref = prefix.reference(QA), prefix.reference(DRY)
    qa, dry = prefix.read(qa_ref), prefix.read(dry_ref)
    prefix.require(
        qa["complete"] is qa["passed"] is qa["source_unchanged"] is qa["actual_wait"] is qa["reaped"] is True
        and dry["complete"] is dry["passed"] is True
        and dry["physics_steps"] == 0,
        "Frozen passing CPU QA and zero-physics preflight are required",
    )
    before = snapshot(ROOT)["sources"]
    prefix.require(before == prefix.read(qa["source_before_ref"])["sources"], "QA must bind current sources")
    inputs, sources = preparation.load_inputs(), source_refs()
    identity = preparation.input_receipt(inputs)
    prefix.require(dry["inputs"] == identity and dry["source_refs"] == sources, "Dry identities must match")
    config, cases = inputs["config"], inputs["cases"]
    producer, producer_refs = None, {}
    if phase == "checker":
        producer = prefix.read(prefix.reference(OUTPUT / "producer/result.json"))
        prefix.require(
            producer["owner_closed"] is producer["sources_unchanged"] is True,
            "A closed producer owner and unchanged sources are required",
        )
        eligible = checker_eligible(producer, cases, config["reused_case"])
        producer_refs = {str(index): producer["cases"][str(index)]["branch_ref"] for index in eligible}
        for ref in producer_refs.values():
            prefix.checked(ref)
    else:
        eligible = [case["spec"]["index"] for case in cases if case["spec"]["index"] != config["reused_case"]]
    directory = OUTPUT / phase
    directory.mkdir(parents=True, exist_ok=False)
    registration = publish_result(
        directory / "registration.json",
        dict(
            phase=phase,
            owner_pid=os.getpid(),
            inputs=identity,
            source_refs=sources,
            qa_ref=qa_ref,
            dry_ref=dry_ref,
            eligible_cases=eligible,
            producer_branch_refs=producer_refs,
            automatic_retry=False,
        ),
    )
    specs = [
        dict(
            key=str(index),
            argv=[sys.executable, "-m", MODULE, "--worker", phase, str(index)],
            log=directory / f"case{index:02d}" / "worker.log",
        )
        for index in eligible
    ]
    lifecycle = (
        run_owned_workers(specs, cwd=ROOT, timeout_seconds=config["worker_timeout_seconds"])
        if specs
        else dict(workers=[], launch_error=None)
    )
    for item in lifecycle["workers"]:
        item["log_ref"] = prefix.reference(item.pop("log"))
    lifecycle_ref = publish_result(directory / "workers.json", lifecycle)
    workers = {int(item["key"]): item for item in lifecycle["workers"]}
    outcomes = {}
    result = dict(
        complete=False,
        passed=False,
        owner_closed=False,
        sources_unchanged=False,
        phase=phase,
        registration_ref=registration,
        lifecycle_ref=lifecycle_ref,
        cases=outcomes,
        eligible_cases=eligible,
        world_model_calls=0,
        fresh_cases=0,
        automatic_retry=False,
    )
    try:
        for case in cases:
            index = case["spec"]["index"]
            if index == config["reused_case"]:
                continue
            entry = dict(status="not_run", expected_periods=case["spec"]["periods"])
            outcomes[str(index)] = entry
            if index not in eligible:
                entry["reason"] = "producer_not_closed"
                continue
            worker_info = workers.get(index)
            if worker_info is None:
                entry["reason"] = "worker_not_launched"
                continue
            result_path = directory / f"case{index:02d}" / "worker/result.json"
            if result_path.is_file():
                entry["available_branch_ref"] = prefix.reference(result_path)
            try:
                prefix.require(
                    worker_info["returncode"] == 0
                    and worker_info["actual_wait"] is worker_info["reaped"] is True
                    and worker_info["signal_sent"] is False
                    and result_path.is_file(),
                    "The owned case worker must exit normally and be reaped",
                )
                branch_ref = prefix.reference(result_path)
                branch = prefix.read(branch_ref)
                prefix.require(branch["case"] == case["spec"], "Worker case identity must match")
                counts = validate_branch(branch, phase, case["spec"]["periods"])
                if phase == "checker":
                    candidate = prefix.read(producer_refs[str(index)])
                    counts["replay_max_difference"] = comparison.compare_periods(
                        previous.periods(candidate), previous.periods(branch)
                    )
                    prefix.require(
                        candidate["runtime"]["continuation"]["stop"] == branch["runtime"]["continuation"]["stop"],
                        "The recorded producer stop must be retained",
                    )
                entry.update(status="closed", branch_ref=branch_ref, counts=counts)
            except Exception as error:
                entry.update(status="failed", error=dict(type=type(error).__name__, message=str(error)))
        owner_closed = all(item["actual_wait"] is item["reaped"] is True for item in lifecycle["workers"])
        source_unchanged = (
            snapshot(ROOT)["sources"] == before and preparation.input_receipt(preparation.load_inputs()) == identity
        )
        result.update(
            owner_closed=owner_closed,
            sources_unchanged=source_unchanged,
            complete=owner_closed and source_unchanged and lifecycle["launch_error"] is None,
            passed=owner_closed
            and source_unchanged
            and lifecycle["launch_error"] is None
            and len(outcomes) == 17
            and all(item["status"] == "closed" for item in outcomes.values()),
        )
    except BaseException as error:
        result["error"] = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        print(publish_result(directory / "result.json", result), flush=True)
    if not result["passed"]:
        raise SystemExit("One or more registered cohort workers did not close successfully")


def aggregate():
    """Report every registered denominator; incomplete cases never qualify."""
    prefix.environment()
    inputs = preparation.load_inputs()
    producer_ref, checker_ref = (
        prefix.reference(OUTPUT / "producer/result.json"),
        prefix.reference(OUTPUT / "checker/result.json"),
    )
    producer, checker = prefix.read(producer_ref), prefix.read(checker_ref)
    prefix.require(
        producer["owner_closed"]
        is checker["owner_closed"]
        is producer["sources_unchanged"]
        is checker["sources_unchanged"]
        is True,
        "Both phase owners must be closed with unchanged sources",
    )
    old_p, old_c = (prefix.read(inputs["configured"][key]) for key in ("reused_producer", "reused_checker"))
    registrations = [prefix.read(group["registration_ref"]) for group in (producer, checker)]
    identity = preparation.input_receipt(inputs)
    prefix.require(
        all(registration["inputs"] == identity for registration in registrations)
        and registrations[0]["source_refs"] == registrations[1]["source_refs"] == source_refs(),
        "The producer and checker must bind the current fixed cohort and sources",
    )
    results = []
    for item in inputs["cases"]:
        spec, index = item["spec"], item["spec"]["index"]
        if index == inputs["config"]["reused_case"]:
            p = dict(status="closed", branch_ref=old_p["branch_ref"], counts=old_p["counts"])
            c = dict(status="closed", branch_ref=old_c["branch_ref"], counts=old_c["counts"])
        else:
            p, c = producer["cases"][str(index)], checker["cases"][str(index)]
        replayed = p["status"] == c["status"] == "closed"
        summary = p.get("counts", {}).get("summary")
        replay_difference = None
        if replayed:
            produced, checked = prefix.read(p["branch_ref"]), prefix.read(c["branch_ref"])
            producer_counts = validate_branch(produced, "producer", spec["periods"])
            checker_counts = validate_branch(checked, "checker", spec["periods"])
            prefix.require(
                producer_counts["summary"] == summary
                and checker_counts["summary"] == c["counts"]["summary"]
                and produced["runtime"]["continuation"]["stop"] == checked["runtime"]["continuation"]["stop"],
                "Each stored case summary and stop must match its independently replayed branch",
            )
            replay_difference = comparison.compare_periods(previous.periods(produced), previous.periods(checked))
            prefix.require(
                replay_difference == c["counts"]["replay_max_difference"],
                "The replay maxima must also match the completed checker receipt",
            )
        results.append(
            dict(
                case=spec,
                producer_status=p["status"],
                checker_status=c["status"],
                producer_ref=p.get("branch_ref"),
                checker_ref=c.get("branch_ref"),
                summary=summary,
                replayed=replayed,
                replay_max_difference=replay_difference,
                qualified=qualifies(summary, replayed, inputs["config"]),
            )
        )
    passed = sum(item["qualified"] for item in results)
    print(
        publish_result(
            OUTPUT / "audit.json",
            dict(
                complete=True,
                passed=True,
                control_qualified=passed == len(results) == 18,
                registered_cases=18,
                qualified_cases=passed,
                cases=results,
                producer_ref=producer_ref,
                checker_ref=checker_ref,
                reused_audit_ref=inputs["configured"]["reused_audit"],
                world_model_calls=0,
                fresh_cases=0,
                physics_steps=0,
            ),
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry", action="store_true")
    group.add_argument("--phase", choices=("producer", "checker"))
    group.add_argument("--worker", nargs=2, metavar=("PHASE", "CASE"))
    group.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    if args.dry:
        dry_run()
    elif args.phase:
        phase_run(args.phase)
    elif args.worker:
        worker(args.worker[0], int(args.worker[1]))
    else:
        aggregate()


if __name__ == "__main__":
    main()
