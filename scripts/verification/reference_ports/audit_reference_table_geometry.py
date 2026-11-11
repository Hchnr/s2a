"""Audit static reference geometry against completed physical TCP history."""

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from s2a.geometry import finite, inverse, pose_error, transform, validate_transform
from s2a.infrastructure.publication import publish_result as publish
from s2a.research.wam.training.native.native_training import state_sha256
from scripts.maintenance.run_cpu_tests import snapshot
from scripts.simulation.runtime import prepare_quality_rate_repair_recorded_feedback as recorded
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.tracking.response_models import reference_table_geometry as geometry

ROOT, RUN = prefix.REPO, geometry.OUTPUT
OUTPUT = ROOT / "outputs/reference_table_geometry_audit_20261111/result.json"
QA = ROOT / "outputs/qa/reference_table_geometry_audit_20261111/result.json"
OWNER_SHA = "da4bb4ff61a8d29cec0507e033b05612eb456d82413639f1348f02ce1c3162fa"
FORCE_OWNER = {
    "path": str(ROOT / "outputs/replayed_contact_torque_20261111/result.json"),
    "sha256": "94bacfab9d7e65950e786eeef968c929ded6a9b02215055e48cb0b90751db3f8",
}


def source_refs():
    return [
        prefix.reference(path)
        for path in (
            Path(__file__).resolve(),
            geometry.CONFIG,
            geometry.PLAN,
            Path(geometry.__file__).resolve(),
            ROOT / "tests/research/response_models/test_reference_table_geometry_audit.py",
        )
    ]


def reconstruct_measured_tcp(desired, error):
    """Recover an observed pose using a completed desired-minus-measured error."""
    desired = validate_transform(np.asarray(desired, dtype=float))
    error = finite(error, (6,))
    measured_rotation = Rotation.from_rotvec(error[3:]).as_matrix().T @ desired[:3, :3]
    return transform(desired[:3, 3] - error[:3], measured_rotation)


def compare_geometry(inputs, branch, force_branch):
    item = inputs["reference_table_geometry"]
    producer = inputs["replayed_contact_torque"]["branches"]["historical"]["producer"]
    previous = recorded.periods(producer)
    rows = branch["analysis"]["periods"]
    force_samples = force_branch["runtime"]["replay_dynamics"]["samples"]
    prefix.require(
        len(rows) == len(item["states"]) == len(item["targets"]) == 47
        and len(previous) == 46
        and len(force_samples) == 1150,
        "Every fixed decision state, generated target and force period is required",
    )
    old = inputs["old"]["old"]["base"]["previous"]
    site_from_tcp = inverse(np.asarray(old["controller"]["source_from_site"])) @ old["tool"]
    initial = old["saved"]["controller_site_pose"][-1] @ site_from_tcp
    maximum_error, mismatches = 0.0, []
    for index, (row, state, target) in enumerate(zip(rows, item["states"], item["targets"], strict=True)):
        step = 80 + index
        prefix.require(
            row["step"] == step
            and row["generated_step"] == 80 + 8 * (index // 8)
            and row["executed"] is (step < 126)
            and row["state_sha256"] == state_sha256(state)
            and row["target_proxy_finger_table_intersection"]
            is bool(row["target_proxy_finger_table_contact_distances_m"])
            and all(value <= 0 for value in row["current_finger_table_contact_distances_m"])
            and all(value <= 0 for value in row["target_proxy_finger_table_contact_distances_m"])
            and row["finger_pose_transform_max_difference"] <= 1e-9
            and row["qpos_max_difference"] <= 1e-12
            and row["table_position_max_difference"] <= 1e-12,
            "Causal state, generated reference, and rigid geometry binding must match",
        )
        measured = (
            initial
            if index == 0
            else reconstruct_measured_tcp(item["targets"][index - 1], previous[index - 1]["tcp_error"])
        )
        independently_computed = pose_error(target, measured)
        maximum_error = max(maximum_error, float(np.max(np.abs(independently_computed - row["current_tcp_error"]))))
        if index:
            sample = force_samples[(index - 1) * 25 + 24]
            prefix.require(sample["step"] == step - 1 and sample["inner_index"] == 24, "Previous physical solver clock")
            previous_contact = any(
                contact["kind"] == "robot_environment"
                and {contact["geom1"], contact["geom2"]} == {"table_collision", "gripper0_finger2_collision"}
                and contact["distance_m"] <= 0
                for contact in sample["contacts"]
            )
            current_contact = bool(row["current_finger_table_contact_distances_m"])
            if previous_contact != current_contact:
                mismatches.append(step)
    prefix.require(maximum_error <= 1e-9, "Current TCP errors must match independently reconstructed physical history")
    return {
        "independent_current_tcp_error_max_difference": maximum_error,
        "prestate_contact_disagreements_with_previous_solver_cache": mismatches,
        "target_intersection_steps": [row["step"] for row in rows if row["target_proxy_finger_table_intersection"]],
        "executed_target_intersections": sum(
            row["target_proxy_finger_table_intersection"] and row["executed"] for row in rows
        ),
        "stopped_target_intersection": rows[-1]["target_proxy_finger_table_intersection"],
        "max_finger_transform_difference": max(row["finger_pose_transform_max_difference"] for row in rows),
        "max_qpos_change": max(row["qpos_max_difference"] for row in rows),
        "max_table_position_change": max(row["table_position_max_difference"] for row in rows),
    }


def audit():
    inputs = geometry.load_inputs()
    consumed = dict(inputs["reference_table_geometry"]["source_sha256"])
    owner = prefix.read({"path": str(RUN / "result.json"), "sha256": OWNER_SHA}, consumed)
    prefix.require(
        owner["complete"] is owner["passed"] is owner["sources_unchanged"] is True
        and owner["world_model_calls"] == owner["fresh_cases"] == 0
        and owner["control_qualified"] is False,
        "A completed source-stable geometry study is required",
    )
    workers = prefix.read(owner["workers_ref"], consumed)
    prefix.require(
        workers["launch_error"] is None
        and len(workers["workers"]) == 1
        and workers["workers"][0]["actual_wait"] is workers["workers"][0]["reaped"] is True
        and workers["workers"][0]["returncode"] == 0
        and workers["workers"][0]["signal_sent"] is False,
        "The geometry worker must actually exit and be reaped",
    )
    registration = prefix.read(owner["registration_ref"], consumed)
    for ref in registration["source_refs"]:
        prefix.checked(ref, consumed)
    branch = prefix.read(owner["branch_ref"], consumed)
    stats = branch["stats"]
    prefix.require(
        branch["complete"] is branch["passed"] is True
        and stats["models_attempted"] == stats["models_completed"] == 1
        and stats["forward_queries_attempted"] == stats["forward_queries_completed"] == 94
        and stats["physics_steps_attempted"] == branch["world_model_calls"] == branch["fresh_cases"] == 0
        and branch["analysis"]["max_object_rotation_change"] <= 1e-12
        and branch["analysis"]["max_qpos_change_during_placement"] <= 1e-12,
        "The registered static-only geometry budget and object invariants must hold",
    )
    force_owner = prefix.read(FORCE_OWNER, consumed)
    prefix.require(force_owner["complete"] is force_owner["passed"] is True, "Closed force replay required")
    force_branch = prefix.read(force_owner["summaries"]["historical"]["branch_ref"], consumed)
    summary = compare_geometry(inputs, branch, force_branch)
    return {
        "complete": True,
        "passed": True,
        "source_refs": source_refs(),
        "owner_ref": {"path": str(RUN / "result.json"), "sha256": OWNER_SHA},
        "source_sha256": consumed,
        "summary": summary,
        "worker_physics_steps": 0,
        "worker_forward_queries": 94,
        "audit_forward_queries": 0,
        "world_model_calls": 0,
        "fresh_cases": 0,
        "inverse_kinematics_verified": False,
        "continuous_path_verified": False,
        "control_qualified": False,
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
        "Completed source-pinned 32-worker geometry audit QA required",
    )
    print(publish(OUTPUT, audit()), flush=True)


if __name__ == "__main__":
    main()
