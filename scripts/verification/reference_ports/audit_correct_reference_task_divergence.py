"""Compare saved object and robot trajectories across the full oracle cohort."""

import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from s2a.infrastructure.publication import publish_result
from s2a.research.wam.training.native.native_training import state_sha256
from scripts.maintenance.run_cpu_tests import snapshot
from scripts.simulation.runtime import prepare_reset_correct_reference_cohort as preparation
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_reset_correct_reference as experiment

ROOT = prefix.REPO
PLAN = ROOT / "plans/v0.0.1/tracking/joint_control/correct_reference_failure_diagnosis_20261116.md"
OUTPUT = ROOT / "outputs/correct_reference_failure_diagnosis_20261116/result.json"
AUDIT = {
    "path": str(ROOT / "outputs/reset_correct_reference_cohort_20261116/audit.json"),
    "sha256": "dbe7d296758cc7cb9c9f66168ed140b13ed2967dbb84e5879044c3e849ec704a",
}
DRY = {
    "path": str(ROOT / "outputs/reset_correct_reference_cohort_dry_20261116.json"),
    "sha256": "b1fa5eadf94aa475386d0fc1762da32b2df0152d4da8c5e75ee44b5099874cd5",
}


def joint_layout(xml):
    """Record named joints and their body paths in XML traversal order."""
    world = ET.fromstring(xml).find("worldbody")
    if world is None:
        raise ValueError("A model worldbody is required")
    layout = []

    def visit(node, bodies):
        current = (*bodies, node.get("name")) if node.tag == "body" else bodies
        if node.tag in ("joint", "freejoint"):
            layout.append((node.tag, node.get("name"), node.get("type", "hinge"), current))
        for child in node:
            visit(child, current)

    visit(world, ())
    names = [row[1] for row in layout]
    if not layout or any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError("An ordered set of uniquely named model joints is required")
    return layout


def state_position_columns(states, positions):
    """Prove one unique saved-state triple equals every audited object position."""
    states, positions = np.asarray(states, dtype=float), np.asarray(positions, dtype=float)
    if (
        states.ndim != 2
        or states.shape[0] < 1
        or states.shape[1] < 3
        or positions.shape != (len(states), 3)
        or not np.isfinite(states).all()
        or not np.isfinite(positions).all()
    ):
        raise ValueError("Finite aligned saved states and object positions are required")
    residuals = [
        float(np.max(np.abs(states[:, start : start + 3] - positions))) for start in range(states.shape[1] - 2)
    ]
    matches = [index for index, value in enumerate(residuals) if value <= 1e-10]
    if len(matches) != 1:
        raise ValueError("The original object position must match exactly one state triple")
    start = matches[0]
    return dict(columns=[start, start + 1, start + 2], maximum_residual=residuals[start])


def compare_case(rows, saved_states, periods, target_object):
    """Compare only the actually executed common prefix; labels remain offline."""
    if not rows or len(periods) == 0 or len(periods) > len(rows):
        raise ValueError("A nonempty original trajectory and its actual prefix are required")
    if any(row.get("step") != index for index, row in enumerate(rows)) or any(
        row.get("step") != index for index, row in enumerate(periods)
    ):
        raise ValueError("Both original and candidate periods must start at reset and be consecutive")
    saved = np.asarray(saved_states, dtype=float)
    if saved.ndim != 2 or len(saved) != len(rows) + 1:
        raise ValueError("Original saved states must include reset and every original endpoint")
    original_positions = np.asarray([row["objects"][target_object]["position"] for row in rows], dtype=float)
    mapping = state_position_columns(saved[1:], original_positions)
    current = np.asarray([row["actual_state"] for row in periods], dtype=float)
    if current.shape != (len(periods), saved.shape[1]) or not np.isfinite(current).all():
        raise ValueError("Every candidate endpoint must use the same finite state layout")
    count = len(periods)
    old = rows[:count]
    reference = np.asarray([row["gripper"]["preview"]["original_reference"] for row in old], dtype=float)
    measured_reference = np.asarray(
        [row["completed_feedback"]["score"]["opening_reference"] for row in periods], dtype=float
    )
    old_error = np.asarray([row["tcp_error"] for row in old], dtype=float)
    new_error = np.asarray([row["tcp_error"] for row in periods], dtype=float)
    old_commands = np.asarray([row["command"] for row in old], dtype=float)
    new_commands = np.asarray([row["command"] for row in periods], dtype=float)
    old_opening = np.asarray([row["opening"] for row in old], dtype=float)
    new_opening = np.asarray([row["completed_feedback"]["score"]["opening"] for row in periods], dtype=float)
    values = (reference, measured_reference, old_error, new_error, old_commands, new_commands, old_opening, new_opening)
    if (
        old_error.shape != (count, 6)
        or new_error.shape != (count, 6)
        or old_commands.shape != (count, 7)
        or new_commands.shape != (count, 7)
        or not all(np.isfinite(value).all() for value in values)
        or np.max(np.abs(reference - measured_reference)) > 1e-10
    ):
        raise ValueError("The aligned original and candidate reference and measurements must be finite")
    start = mapping["columns"][0]
    object_delta = np.linalg.norm(current[:, start : start + 3] - original_positions[:count], axis=1)
    tcp_delta = np.linalg.norm(old_error[:, :3] - new_error[:, :3], axis=1)
    arm_delta = np.max(np.abs(old_commands[:, :6] - new_commands[:, :6]), axis=1)
    grip_diff = old_commands[:, 6] != new_commands[:, 6]
    old_grasp = [row["objects"][target_object]["grasp_contact"] is True for row in old]
    old_success = [row["success"] is True for row in old]
    new_grasp = [row["target_grasp"] is True for row in periods]
    new_success = [row["task_success"] is True for row in periods]

    def first(indices):
        indices = np.flatnonzero(indices)
        return int(indices[0]) if len(indices) else None

    def rms(value):
        return float(np.sqrt(np.mean(value**2)))

    return dict(
        common_periods=count,
        original_periods=len(rows),
        state_position_mapping=mapping,
        object_position_difference_m=dict(
            maximum=float(np.max(object_delta)),
            final=float(object_delta[-1]),
            first_above_1mm=first(object_delta > 0.001),
            first_above_10mm=first(object_delta > 0.01),
        ),
        tcp_position_difference_m=dict(maximum=float(np.max(tcp_delta)), median=float(np.median(tcp_delta))),
        arm_command_different_periods=int(np.sum(arm_delta > 1e-12)),
        arm_command_maximum_absolute_difference=float(np.max(arm_delta)),
        gripper_command_different_periods=int(np.sum(grip_diff)),
        first_gripper_command_difference=first(grip_diff),
        original_opening_rmse=rms(old_opening - reference),
        candidate_opening_rmse=rms(new_opening - reference),
        original_grasp_periods=sum(old_grasp),
        candidate_grasp_periods=sum(new_grasp),
        original_success_periods=sum(old_success),
        candidate_success_periods=sum(new_success),
        original_first_success=first(old_success),
        candidate_first_success=first(new_success),
    )


def main():
    prefix.environment()
    before = snapshot(ROOT)["sources"]
    audit, dry = prefix.read(AUDIT), prefix.read(DRY)
    prefix.require(
        audit["complete"] is audit["passed"] is dry["complete"] is dry["passed"] is True
        and audit["control_qualified"] is False
        and audit["registered_cases"] == len(audit["cases"]) == len(dry["inputs"]["cases"]) == 18,
        "The closed fixed cohort and its full original denominator are required",
    )
    phase_results = [prefix.read(audit[key]) for key in ("producer_ref", "checker_ref")]
    registrations = [prefix.read(result["registration_ref"]) for result in phase_results]
    prefix.require(
        all(result["owner_closed"] is result["sources_unchanged"] is True for result in phase_results)
        and all(registration["inputs"] == dry["inputs"] for registration in registrations),
        "The complete producer and checker must bind the same original inputs",
    )
    config = prefix.read(dry["inputs"]["config_ref"])
    shared = preparation.shared_inputs()
    prefix.require(shared["receipt"] == dry["inputs"]["shared_receipt"], "The audited simulator assets must match")
    comparisons = []
    for item, identity in zip(audit["cases"], dry["inputs"]["cases"], strict=True):
        spec, refs = identity["spec"], identity["refs"]
        prefix.require(
            item["case"] == spec
            and item["replayed"] is True
            and item["producer_status"] == item["checker_status"] == "closed",
            "Each original case must have a closed and independently replayed candidate",
        )
        rows = prefix.read(refs["rows"])
        baseline = prefix.read(refs["result"])
        branch = prefix.read(item["producer_ref"])
        reconstructed = preparation.case_inputs(identity, shared)
        mapped_xml = reconstructed["record_xml"]
        mapped_sha = hashlib.sha256(mapped_xml.encode()).hexdigest()
        model_stage = branch["runtime"]["model_records"][2]
        original_layout = joint_layout(prefix.checked(baseline["model_ref"]).read_text())
        candidate_layout = joint_layout(mapped_xml)
        prefix.require(
            model_stage["stage"] == "record_xml"
            and model_stage["completed"] is True
            and model_stage["xml_sha256"] == mapped_sha
            and original_layout == candidate_layout,
            "Both executed models must have the same ordered joint and body layout",
        )
        layout_sha = hashlib.sha256(json.dumps(original_layout, separators=(",", ":")).encode()).hexdigest()
        with np.load(prefix.checked(refs["arrays"]), allow_pickle=False) as archive:
            saved = np.asarray(archive["sim_states"]).copy()
        recorded_sha = baseline["images"][0]["current_endpoint"]["sim_state_sha256"]
        prefix.require(
            len(rows) == spec["periods"]
            and state_sha256(saved[0]) == recorded_sha
            and branch["runtime"]["reset_endpoint_comparison"]["recorded_state_sha256"] == recorded_sha,
            "Original and candidate must share the exact saved reset",
        )
        periods = experiment.periods(branch)
        metrics = compare_case(rows, saved, periods, config["target_objects_by_task"][str(spec["task_id"])])
        metrics["model_layout"] = dict(
            joint_count=len(original_layout), joint_layout_sha256=layout_sha, candidate_record_xml_sha256=mapped_sha
        )
        prefix.require(
            metrics["common_periods"] == item["summary"]["completed_periods"],
            "Only actually executed and independently replayed periods may be compared",
        )
        comparisons.append(
            dict(
                case=spec,
                qualified=item["qualified"],
                baseline_refs=refs,
                producer_ref=item["producer_ref"],
                metrics=metrics,
            )
        )
    prefix.require(snapshot(ROOT)["sources"] == before, "Diagnostic sources changed during analysis")
    print(
        publish_result(
            OUTPUT,
            dict(
                complete=True,
                passed=True,
                source_unchanged=True,
                exploratory_after_outcome=True,
                cases=comparisons,
                registered_cases=18,
                cohort_audit_ref=AUDIT,
                cohort_dry_ref=DRY,
                plan_ref=prefix.reference(PLAN),
                source_ref=prefix.reference(Path(__file__).resolve()),
                physics_steps=0,
                world_model_calls=0,
                fresh_cases=0,
            ),
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
