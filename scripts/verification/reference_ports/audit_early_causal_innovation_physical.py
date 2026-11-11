"""Audit the bounded causal-innovation action against its independent replay."""

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from s2a.infrastructure.publication import publish_result as publish
from scripts.maintenance.run_cpu_tests import snapshot
from scripts.simulation.runtime import prepare_early_causal_innovation_physical as admission
from scripts.simulation.runtime import prepare_quality_rate_repair_recorded_feedback as prior_admission
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_causal_wam_two_window_feedback as historical
from scripts.verification.reference_ports import audit_quality_rate_repair_recorded_feedback as prior_audit

ROOT, RUN = admission.ROOT, admission.OUTPUT
OUTPUT = ROOT / "outputs/early_causal_innovation_physical_audit_20261111/result.json"
QA = ROOT / "outputs/qa/early_causal_innovation_physical_audit_20261111/result.json"
PRODUCER_SHA = "d8da5c619ec6ba2eddfa6ab4defad1f277f3883d68fe4e20650777dac02c896b"
CHECKER_SHA = "beea8cd09b39e39a65585c008b74c33657b929c9a9b958ab0d81f618067a4137"


def source_refs():
    return [
        prefix.reference(path)
        for path in (
            Path(__file__).resolve(),
            ROOT / "scripts/simulation/runtime/prepare_early_causal_innovation_physical.py",
            ROOT / "scripts/simulation/runtime/run_early_causal_innovation_physical.py",
            ROOT / "scripts/verification/reference_ports/audit_quality_rate_repair_recorded_feedback.py",
            ROOT / "tests/research/native_runtime/test_early_causal_innovation_physical_audit.py",
        )
    ]


def phase(name, digest, consumed):
    summary = prefix.read({"path": str(RUN / name / "result.json"), "sha256": digest}, consumed)
    prefix.require(
        summary["phase"] == name
        and summary["complete"] is summary["passed"] is summary["sources_unchanged"] is True
        and summary["control_qualified"] is False
        and summary["new_model_initializations"] == summary["new_model_requests"] == summary["new_denoise_calls"] == 0
        and summary["counts"]["control_steps"] == 123
        and summary["counts"]["native_steps"] == 3075
        and summary["counts"]["continuation_steps"] == 43,
        "The completed registered physical phase is required",
    )
    workers = prefix.read(summary["workers_ref"], consumed)
    prefix.require(
        workers["launch_error"] is None
        and len(workers["workers"]) == (2 if name == "producer" else 1)
        and all(
            row["actual_wait"] is row["reaped"] is True and row["signal_sent"] is False and row["returncode"] == 0
            for row in workers["workers"]
        ),
        "Every physical or relay child must exit and actually be reaped",
    )
    registration = prefix.read(summary["registration_ref"], consumed)
    for ref in registration["source_refs"]:
        prefix.checked(ref, consumed)
    branch = prefix.read(summary["branch_ref"], consumed)
    prefix.require(branch["complete"] is branch["passed"] is True, "A closed physical branch is required")
    return summary, branch


def compare_periods(produced, checked, previous):
    prefix.require(len(produced) == len(checked) == 43 and len(previous) >= 43, "Exact physical periods required")
    replay = {key: 0.0 for key in ("state", "q", "qvel", "time")}
    prior = {key: 0.0 for key in ("state", "command", "q", "qvel", "time")}
    for index, (actual, copy) in enumerate(zip(produced, checked, strict=True)):
        historical.compare_first_period(actual, copy)
        replay["state"] = max(
            replay["state"], float(np.max(np.abs(np.asarray(actual["actual_state"]) - copy["actual_state"])))
        )
        for key in ("q", "qvel", "time"):
            a = np.asarray([sample[key] for sample in actual["actual_inner"]])
            b = np.asarray([sample[key] for sample in copy["actual_inner"]])
            replay[key] = max(replay[key], float(np.max(np.abs(a - b))))
        if index < 42:
            old = previous[index]
            prior_admission.matched_prior_period(actual, old)
            for key, field in (("state", "actual_state"), ("command", "command")):
                prior[key] = max(prior[key], float(np.max(np.abs(np.asarray(actual[field]) - old[field]))))
            for key in ("q", "qvel", "time"):
                a = np.asarray([sample[key] for sample in actual["actual_inner"]])
                b = np.asarray([sample[key] for sample in old["actual_inner"]])
                prior[key] = max(prior[key], float(np.max(np.abs(a - b))))
    return replay, prior


def verify_causal_action(produced, previous, runtime):
    completed, action, old_action = produced[-2], produced[-1], previous[42]
    binding, quality = runtime["early_innovation_binding"], action["decision"]["quality"]
    stop = runtime["continuation"]["stop"]
    prefix.require(
        completed["step"] == 121
        and action["step"] == old_action["step"] == 122
        and quality["status"] == "innovation_model_witness"
        and quality["verification"]["passed"] is quality["verification"]["innovation_gate_passed"] is True
        and binding["last_completed_step"] == 121
        and binding["new_action_budget"] == 1
        and binding["solver_calls"] == binding["limit_stops"] == 1
        and binding["independent_audit_attempts"] == binding["independent_audit_passes"] == 1
        and binding["task_labels_used"] is False
        and stop["at_step"] == 123
        and stop["next_action_dispatched"] is False
        and stop["solver_status"] == "registered_one_period_limit",
        "One causal, audited action followed by the registered budget stop is required",
    )
    last_predicted = np.asarray(completed["decision"]["quality"]["first_tcp_error"])
    last_actual = np.asarray(completed["tcp_error"])
    translation = last_actual[:3] - last_predicted[:3]
    rotation = Rotation.from_rotvec(last_actual[3:]).inv() * Rotation.from_rotvec(last_predicted[3:])
    np.testing.assert_allclose(binding["translation_innovation_m"], translation, atol=1e-12, rtol=0)
    np.testing.assert_allclose(binding["rotation_discrepancy"], rotation.as_matrix(), atol=1e-12, rtol=0)
    predicted = np.asarray(quality["first_tcp_error"])
    corrected = np.r_[predicted[:3] + translation, (Rotation.from_rotvec(predicted[3:]) * rotation.inv()).as_rotvec()]
    np.testing.assert_allclose(
        corrected, quality["verification"]["innovation_corrected_first_tcp_error"], atol=1e-10, rtol=0
    )
    prefix.require(
        np.linalg.norm(corrected[:3]) <= 0.01 + 1e-10 and np.linalg.norm(corrected[3:]) <= 0.05 + 1e-10,
        "The independently reconstructed innovation gate must pass",
    )
    actual, old = np.asarray(action["tcp_error"]), np.asarray(old_action["tcp_error"])

    def norms(value):
        return {"position_m": float(np.linalg.norm(value[:3])), "rotation_rad": float(np.linalg.norm(value[3:]))}

    return {
        "step": 122,
        "raw_model_error": norms(predicted),
        "innovation_corrected_model_error": norms(corrected),
        "actual_error": norms(actual),
        "prior_actual_error": norms(old),
        "arm_command_max_difference_from_prior": float(
            np.max(np.abs(np.asarray(action["command"])[:6] - old_action["command"][:6]))
        ),
        "actual_pose_caps_satisfied": action["pose_caps_satisfied"],
        "task_success": action["task_success"],
        "quality_model_responses": quality["response_evaluations"],
    }


def audit():
    inputs = admission.load_inputs()
    consumed = dict(inputs["source_sha256"])
    producer_summary, producer = phase("producer", PRODUCER_SHA, consumed)
    checker_summary, checker = phase("checker", CHECKER_SHA, consumed)
    isolation = prefix.read(checker_summary["policy_isolation_ref"], consumed)
    prefix.require(
        isolation["before"] is isolation["after"] is True and checker["candidate_modules_unloaded"] is True,
        "The checker must remain isolated from candidate policies",
    )
    prior_audit.verify_references(inputs, producer, prefix.read(producer_summary["relay_ref"], consumed), consumed)
    produced, checked = prior_admission.periods(producer), prior_admission.periods(checker)
    replay, prior = compare_periods(produced, checked, inputs["prior_periods"])
    detail = verify_causal_action(produced, inputs["prior_periods"], producer["runtime"])
    prefix.require(
        all(value <= 1e-10 for value in replay.values())
        and checker["runtime"]["continuation"]["reported_producer_stop"] == producer["runtime"]["continuation"]["stop"],
        "Independent replay and reported budget stop must match",
    )
    tail = 0
    for row in reversed(produced):
        if not row["task_success"]:
            break
        tail += 1
    return {
        "complete": True,
        "passed": True,
        "source_refs": source_refs(),
        "producer_ref": {"path": str(RUN / "producer/result.json"), "sha256": PRODUCER_SHA},
        "checker_ref": {"path": str(RUN / "checker/result.json"), "sha256": CHECKER_SHA},
        "historical_responses": 6,
        "historical_denoise_calls": 210,
        "new_model_requests": 0,
        "new_denoise_calls": 0,
        "physical_periods_per_phase": 123,
        "native_steps_per_phase": 3075,
        "continuation_periods": 43,
        "prior_periods_matched": 42,
        "replayed_continuation_inner_samples": 1075,
        "prior_period_max_difference": prior,
        "independent_replay_max_difference": replay,
        "checker_policy_isolated": True,
        "step122": detail,
        "stop_kind": "registered_one_period_limit",
        "stop_reason_independently_verified": False,
        "terminal_consecutive_success_periods": tail,
        "minimum_joint_margin_rad": min(row["minimum_inner_margin_rad"] for row in produced),
        "inner_below_buffer_samples": sum(row["inner_below_buffer_samples"] for row in produced),
        "inner_outside_samples": sum(row["inner_outside_samples"] for row in produced),
        "qualified_stable_success": False,
        "control_qualified": False,
        "source_sha256": consumed,
        "automatic_retry": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", required=True)
    parser.parse_args()
    qa = prefix.read(prefix.reference(QA))
    prefix.require(
        qa["complete"] is qa["passed"] is qa["all_tests_passed"] is qa["source_unchanged"] is True
        and qa["actual_wait"] is qa["reaped"] is True
        and qa["signal_sent"] is False
        and qa["configured_parallelism"] == 32
        and qa["skipped_count"] == qa["xfail_count"] == qa["xpass_count"] == 0
        and not qa["collection"]["collection_errors"],
        "Completed 32-worker audit QA required",
    )
    prefix.require(
        snapshot(ROOT)["sources"] == prefix.read(qa["source_before_ref"])["sources"],
        "Audit QA and source identity must match",
    )
    print(publish(OUTPUT, audit()), flush=True)


if __name__ == "__main__":
    main()
