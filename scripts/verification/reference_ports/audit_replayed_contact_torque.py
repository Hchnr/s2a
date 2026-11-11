"""Independently verify replayed contact and torque evidence against saved actions."""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from s2a.infrastructure.publication import publish_result as publish
from scripts.maintenance.run_cpu_tests import snapshot
from scripts.simulation.runtime import prepare_quality_rate_repair_recorded_feedback as recorded
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_causal_wam_two_window_feedback as historical
from scripts.tracking.response_models import replay_contact_torque as experiment

ROOT, RUN = prefix.REPO, experiment.OUTPUT
OUTPUT = ROOT / "outputs/replayed_contact_torque_audit_20261111/result.json"
QA = ROOT / "outputs/qa/replayed_contact_torque_audit_20261111/result.json"
OWNER_SHA = "94bacfab9d7e65950e786eeef968c929ded6a9b02215055e48cb0b90751db3f8"


def source_refs():
    return [
        prefix.reference(path)
        for path in (
            Path(__file__).resolve(),
            experiment.CONFIG,
            experiment.PLAN,
            Path(experiment.__file__).resolve(),
            ROOT / "tests/research/response_models/test_replayed_contact_torque_audit.py",
        )
    ]


def check_branch(producer, replay, expected_periods):
    """Recheck commands, full endpoints, inner samples and observation clocks."""
    actual, checked = recorded.periods(producer), recorded.periods(replay)
    dynamics = replay["runtime"]["replay_dynamics"]
    samples = dynamics["samples"]
    prefix.require(
        len(actual) == len(checked) == expected_periods
        and len(samples) == 25 * expected_periods
        and replay["complete"] is replay["passed"] is replay["candidate_modules_unloaded"] is True
        and dynamics["actual_policy_inputs_changed"] is False
        and dynamics["controller_sample_phase"] == "after_control_before_integration"
        and dynamics["constraint_sample_phase"] == "last_inner_step_solver_cache_after_integration",
        "A complete isolated replay with correctly timed force observations is required",
    )
    maximum = {key: 0.0 for key in ("command", "state", "q", "qvel", "time", "observer_time")}
    for index, (source, copied) in enumerate(zip(actual, checked, strict=True)):
        prefix.require(source["step"] == copied["step"] == 80 + index, "Consecutive saved actions required")
        historical.compare_first_period(source, copied)
        for key, field in (("command", "command"), ("state", "actual_state")):
            maximum[key] = max(maximum[key], float(np.max(np.abs(np.asarray(source[field]) - copied[field]))))
        for key in ("q", "qvel", "time"):
            left = np.asarray([row[key] for row in source["actual_inner"]])
            right = np.asarray([row[key] for row in copied["actual_inner"]])
            maximum[key] = max(maximum[key], float(np.max(np.abs(left - right))))
        for inner, (sample, measured) in enumerate(
            zip(samples[25 * index : 25 * (index + 1)], copied["actual_inner"], strict=True)
        ):
            prefix.require(
                sample["step"] == source["step"] and sample["inner_index"] == inner, "Force sample alignment"
            )
            maximum["observer_time"] = max(maximum["observer_time"], abs(sample["time_after"] - measured["time"]))
            raw, clipped = np.asarray(sample["controller_raw_torque_nm"]), np.asarray(sample["commanded_torque_nm"])
            np.testing.assert_allclose(
                clipped, np.clip(raw, dynamics["torque_lower_nm"], dynamics["torque_upper_nm"]), atol=1e-12, rtol=0
            )
            np.testing.assert_array_equal(clipped, sample["actuator_control"])
    prefix.require(all(value <= 1e-10 for value in maximum.values()), "Recorded dynamics must align with actual replay")
    return maximum


def check_shared_prefix(historical_samples, innovation_samples):
    """Compare the 42 identical completed periods before policy divergence."""
    fields = (
        "time_before",
        "time_after",
        "controller_raw_torque_nm",
        "commanded_torque_nm",
        "actuator_control",
        "joint_constraint_force_nm",
        "joint_actuator_force_nm",
        "joint_bias_force_nm",
    )
    left, right = historical_samples[:1050], innovation_samples[:1050]
    prefix.require(len(left) == len(right) == 1050, "Forty-two complete shared periods required")
    maximum = {key: 0.0 for key in fields}
    for first, second in zip(left, right, strict=True):
        prefix.require(
            first["step"] == second["step"] and first["inner_index"] == second["inner_index"], "Shared clock"
        )
        for key in fields:
            maximum[key] = max(maximum[key], float(np.max(np.abs(np.asarray(first[key]) - second[key]))))
        prefix.require(first["contacts"] == second["contacts"], "Shared-prefix contacts differ")
    prefix.require(all(value <= 1e-10 for value in maximum.values()), "Shared-prefix dynamics differ")
    return maximum


def summarize_samples(samples):
    periods = []
    for start in range(0, len(samples), 25):
        chunk = samples[start : start + 25]
        prefix.require(len(chunk) == 25 and len({row["step"] for row in chunk}) == 1, "Complete force period")
        clipped, active, pairs = Counter(), 0, {}
        for sample in chunk:
            raw, command = np.asarray(sample["controller_raw_torque_nm"]), np.asarray(sample["commanded_torque_nm"])
            for joint in np.flatnonzero(np.abs(raw - command) > 1e-9):
                clipped[int(joint) + 1] += 1
            contacts = [
                item for item in sample["contacts"] if item["kind"] == "robot_environment" and item["active_constraint"]
            ]
            active += bool(contacts)
            for contact in contacts:
                pair = tuple(sorted((contact["geom1"], contact["geom2"])))
                item = pairs.setdefault(pair, {"samples": 0, "maximum_force_n": 0.0})
                item["samples"] += 1
                item["maximum_force_n"] = max(
                    item["maximum_force_n"], float(np.linalg.norm(contact["wrench_contact_frame"][:3]))
                )
        periods.append(
            {
                "step": chunk[0]["step"],
                "clipped_samples_by_joint": dict(clipped),
                "max_torque_clip_nm": max(
                    float(np.max(np.abs(np.asarray(row["controller_raw_torque_nm"]) - row["commanded_torque_nm"])))
                    for row in chunk
                ),
                "active_robot_environment_contact_samples": active,
                "max_joint_constraint_force_nm": max(
                    float(np.max(np.abs(row["joint_constraint_force_nm"]))) for row in chunk
                ),
                "contact_pairs": [{"geometries": pair, **item} for pair, item in sorted(pairs.items())],
            }
        )
    return periods


def audit():
    inputs = experiment.load_inputs()
    consumed = dict(inputs["replayed_contact_torque"]["source_sha256"])
    owner = prefix.read({"path": str(RUN / "result.json"), "sha256": OWNER_SHA}, consumed)
    prefix.require(
        owner["complete"] is owner["passed"] is owner["sources_unchanged"] is True
        and owner["world_model_calls"] == owner["fresh_cases"] == 0
        and owner["control_qualified"] is False,
        "One completed source-stable instrumented replay is required",
    )
    workers = prefix.read(owner["workers_ref"], consumed)
    prefix.require(
        workers["launch_error"] is None
        and len(workers["workers"]) == 2
        and all(
            row["actual_wait"] is row["reaped"] is True and row["signal_sent"] is False and row["returncode"] == 0
            for row in workers["workers"]
        ),
        "Both instrumented workers must exit and actually be reaped",
    )
    registration = prefix.read(owner["registration_ref"], consumed)
    for ref in registration["source_refs"]:
        prefix.checked(ref, consumed)
    reports, samples = {}, {}
    for name, source in inputs["replayed_contact_torque"]["branches"].items():
        branch = prefix.read(owner["summaries"][name]["branch_ref"], consumed)
        maximum = check_branch(source["producer"], branch, source["config"]["continuation_periods"])
        samples[name] = branch["runtime"]["replay_dynamics"]["samples"]
        periods = summarize_samples(samples[name])
        reports[name] = {
            "replay_max_difference": maximum,
            "continuation_samples": len(samples[name]),
            "periods": periods,
        }
    shared = check_shared_prefix(samples["historical"], samples["innovation"])
    return {
        "complete": True,
        "passed": True,
        "source_refs": source_refs(),
        "owner_ref": {"path": str(RUN / "result.json"), "sha256": OWNER_SHA},
        "source_sha256": consumed,
        "branches": reports,
        "shared_prefix_samples": 1050,
        "shared_prefix_max_difference": shared,
        "shared_prefix_contacts_identical": True,
        "new_world_model_calls": 0,
        "fresh_cases": 0,
        "control_qualified": False,
        "causal_mechanism_proven": False,
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
        and qa["skipped_count"] == 0
        and not qa["collection"]["collection_errors"]
        and snapshot(ROOT)["sources"] == prefix.read(qa["source_before_ref"])["sources"],
        "Completed source-pinned 32-worker audit QA required",
    )
    print(publish(OUTPUT, audit()), flush=True)


if __name__ == "__main__":
    main()
