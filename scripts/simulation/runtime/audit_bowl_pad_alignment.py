"""Recompute saved box projections from vertices without running MuJoCo."""

import itertools

import numpy as np

from s2a.infrastructure.publication import publish_result
from scripts.simulation.runtime import audit_bowl_h40_opening_lead as lead_audit
from scripts.simulation.runtime import audit_reset_bowl_saved_contacts as contacts_audit
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix

ROOT = prefix.REPO
SOURCE = {
    "path": str(ROOT / "outputs/bowl_pad_alignment_20261112/result.json"),
    "sha256": "19d2537aed166cda156c5c3152e00d679593425bf66e5f1d53cd62ce98f251be",
}
OUTPUT = ROOT / "outputs/bowl_pad_alignment_audit_20261112/result.json"
TARGET = [f"akita_black_bowl_1_g{i}" for i in range(1, 41)]
PADS = ["gripper0_finger1_pad_collision", "gripper0_finger2_pad_collision"]
CORNERS = np.asarray(list(itertools.product((-1.0, 1.0), repeat=3)))


def check_point(point):
    """Independently recompute each oriented box projection from eight vertices."""
    a = point["pad_alignment"]
    left, right = np.asarray(a["left_pad_world_m"]), np.asarray(a["right_pad_world_m"])
    mid = (left + right) / 2
    direction = right - left
    length = float(np.linalg.norm(direction))
    prefix.require(length > 1e-12, "Distinct pad centers are required")
    axis = direction / length
    body = np.asarray(a["body_origin_world_m"])
    delta = body - mid
    along = float(axis @ delta)
    np.testing.assert_allclose(a["pad_midpoint_world_m"], mid, rtol=0, atol=1e-12)
    np.testing.assert_allclose(a["jaw_axis_world"], axis, rtol=0, atol=1e-12)
    np.testing.assert_allclose(a["pad_center_separation_m"], length, rtol=0, atol=1e-12)
    np.testing.assert_allclose(a["body_from_pad_midpoint_world_m"], delta, rtol=0, atol=1e-12)
    np.testing.assert_allclose(a["body_axis_projection_m"], along, rtol=0, atol=1e-12)
    np.testing.assert_allclose(a["body_axis_perpendicular_m"], np.linalg.norm(delta - along * axis), rtol=0, atol=1e-12)
    rotation, position = np.asarray(a["site_rotation_world"]), np.asarray(a["site_position_world_m"])
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), rtol=0, atol=1e-10)
    np.testing.assert_allclose(np.linalg.det(rotation), 1, rtol=0, atol=1e-10)
    np.testing.assert_allclose(a["body_from_site_in_site_m"], rotation.T @ (body - position), rtol=0, atol=1e-12)
    boxes = a["box_projections"]
    prefix.require(
        [box["name"] for box in boxes] == PADS + TARGET, "All registered collision boxes must appear in order"
    )
    intervals = []
    for box in boxes:
        center, rotation, size = (
            np.asarray(box[key], dtype=float) for key in ("center_world_m", "rotation_world", "half_size_m")
        )
        prefix.require(
            center.shape == size.shape == (3,)
            and rotation.shape == (3, 3)
            and np.isfinite([*center, *size, *rotation.ravel()]).all()
            and np.all(size > 0),
            "Every box requires finite positive geometry",
        )
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), rtol=0, atol=1e-10)
        np.testing.assert_allclose(np.linalg.det(rotation), 1, rtol=0, atol=1e-10)
        vertices = center + (CORNERS * size) @ rotation.T
        projections = (vertices - mid) @ axis
        interval = [float(projections.min()), float(projections.max())]
        np.testing.assert_allclose(box["jaw_axis_interval_m"], interval, rtol=0, atol=1e-12)
        intervals.append(interval)
    np.testing.assert_allclose(boxes[0]["center_world_m"], left, rtol=0, atol=1e-12)
    np.testing.assert_allclose(boxes[1]["center_world_m"], right, rtol=0, atol=1e-12)
    target_interval = [min(row[0] for row in intervals[2:]), max(row[1] for row in intervals[2:])]
    np.testing.assert_allclose(a["target_axis_interval_m"], target_interval, rtol=0, atol=1e-12)
    contacts_audit.check_contacts(point["contacts"])
    right_interval = intervals[1]
    gap = max(0.0, right_interval[0] - target_interval[1], target_interval[0] - right_interval[1])
    return dict(
        target_interval_m=target_interval,
        left_pad_interval_m=intervals[0],
        right_pad_interval_m=right_interval,
        right_pad_projection_gap_m=gap,
        jaw_axis_world=axis.tolist(),
        body_axis_projection_m=along,
        body_axis_perpendicular_m=float(a["body_axis_perpendicular_m"]),
        pad_center_separation_m=length,
        body_from_site_in_site_m=a["body_from_site_in_site_m"],
    )


def main():
    prefix.environment()
    source = prefix.read(SOURCE)
    worker, lifecycle = prefix.read(source["worker_ref"]), prefix.read(source["workers_ref"])
    counts = worker["counters"]
    prefix.require(
        source["complete"] is source["passed"] is source["sources_unchanged"] is True
        and worker["complete"] is worker["passed"] is True
        and source["automatic_retry"] is False
        and len(lifecycle["workers"]) == 1
        and lifecycle["workers"][0]["actual_wait"] is lifecycle["workers"][0]["reaped"] is True
        and lifecycle["workers"][0]["returncode"] == 0
        and lifecycle["workers"][0]["signal_sent"] is False
        and counts["models"] == counts["data"] == 1
        and counts["forwards_attempted"] == counts["forwards_completed"] == 97
        and counts["physics_steps"] == counts["contexts"] == counts["model_requests"] == 0,
        "The bounded static result and reaped worker are required",
    )
    registration = prefix.read(source["registration_ref"])
    refs = registration["inputs"]["refs"]
    prior = prefix.read(refs["prior_contacts_worker"])["geometry"]
    lead = prefix.read(refs["lead_audit"])
    prefix.require(
        refs["lead_audit"]["sha256"] == "005cb8a5e91c8625f2bb7befee71cd9978d0d1fb1276b75629769ec4d67b8577"
        and lead["complete"] is lead["passed"] is True
        and lead["producer_branch_ref"] == refs["lead_producer"],
        "The registered lead audit is required",
    )
    candidate = prefix.read(refs["lead_producer"])
    first, rest = lead_audit.original.flatten_fixed(candidate)
    candidate_periods = first + rest
    value = worker["geometry"]
    check_point(value["reset"])
    prefix.require(value["reset"]["contacts"] == prior["reset"]["contacts"], "The reset contacts must match")
    summaries = {}
    for branch, rows in value["points"].items():
        prefix.require(
            branch in ("fixed", "lead")
            and len(rows) == 48
            and [row["endpoint_step"] for row in rows] == list(range(1, 49)),
            "Both complete ordered endpoint sets are required",
        )
        positions, mismatches = {}, []
        for point in rows:
            step = point["endpoint_step"]
            position = check_point(point)
            if step in (40, 43, 46, 48):
                positions[str(step)] = position
            if point["contacts"]["bilateral"] != point["target_grasp"]:
                mismatches.append(step)
            if branch == "fixed":
                old = prior["points"]["fixed"][step - 1]
                prefix.require(
                    point["contacts"] == old["contacts"] and point["state_sha256"] == old["state_sha256"],
                    "All original baseline contacts and endpoint identities must match",
                )
            else:
                old = candidate_periods[step - 1]
                prefix.require(
                    point["state_sha256"] == old["endpoint_receipt"]["sim_state_sha256"]
                    and point["target_grasp"] == old["target_grasp"]
                    and point["task_success"] == old["task_success"],
                    "Lead endpoint identities and native labels must match the physical run",
                )
        prefix.require(mismatches == value["contact_label_mismatches"][branch], "All label mismatches must be listed")
        summaries[branch] = dict(
            selected_endpoints=positions,
            left_pad_steps=[row["endpoint_step"] for row in rows if row["contacts"]["left_pad"]],
            right_pad_steps=[row["endpoint_step"] for row in rows if row["contacts"]["right_pad"]],
            bilateral_steps=[row["endpoint_step"] for row in rows if row["contacts"]["bilateral"]],
            grasp_label_mismatch_steps=mismatches,
        )
    prefix.require(set(summaries) == {"fixed", "lead"}, "Exactly the two registered branches are required")
    result = dict(
        complete=True,
        passed=True,
        control_qualified=False,
        source_ref=SOURCE,
        worker_ref=source["worker_ref"],
        summaries=summaries,
        audited_points=97,
        audited_box_projections=97 * 42,
        audit_models=0,
        audit_forwards=0,
        audit_physics_steps=0,
        original_forward_reexecuted=False,
        cumulative_static_models=4,
        cumulative_static_forwards=301,
        automatic_retry=False,
    )
    print(publish_result(OUTPUT, result), flush=True)


if __name__ == "__main__":
    main()
