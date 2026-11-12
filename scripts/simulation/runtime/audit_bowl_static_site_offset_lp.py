"""Independently check every static box-pair result with linear programming."""

import numpy as np
from scipy.optimize import linprog

from s2a.infrastructure.publication import publish_result
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix

ROOT = prefix.REPO
SOURCE = {
    "path": str(ROOT / "outputs/bowl_static_site_offset_20261112/result.json"),
    "sha256": "f49c4f6ca5a95b84492b657ca5fb50bf593411b487cee3f75e327c88de78fc69",
}
OUTPUT = ROOT / "outputs/bowl_static_site_offset_lp_audit_20261112/result.json"


def box_slack(first, second):
    """Maximize equal signed clearance from all twelve OBB faces in world metres."""
    blocks, bounds = [], []
    for box in (first, second):
        center, rotation, size = (
            np.asarray(box[key], dtype=float) for key in ("center_world_m", "rotation_world", "half_size_m")
        )
        if (
            center.shape != (3,)
            or rotation.shape != (3, 3)
            or size.shape != (3,)
            or not np.isfinite([*center, *rotation.ravel(), *size]).all()
            or np.any(size <= 0)
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-10, rtol=0)
        ):
            raise ValueError("Finite oriented boxes with proper orthogonal axes are required")
        local_center = rotation.T @ center
        blocks.extend((rotation.T, -rotation.T))
        bounds.extend((size + local_center, size - local_center))
    normals = np.vstack(blocks)
    rhs = np.concatenate(bounds)
    result = linprog(
        [0, 0, 0, -1], A_ub=np.column_stack((normals, np.ones(12))), b_ub=rhs, bounds=[(None, None)] * 4, method="highs"
    )
    prefix.require(
        result.success
        and np.isfinite(result.x).all()
        and np.max(np.column_stack((normals, np.ones(12))) @ result.x - rhs) <= 1e-8,
        "The independent LP must return a finite, constraint-verified optimum",
    )
    return float(result.x[3])


def main():
    prefix.environment()
    source = prefix.read(SOURCE)
    audit = prefix.read(source["source_audit_ref"])
    worker = prefix.read(audit["worker_ref"])
    prefix.require(
        source["complete"] is source["passed"] is audit["complete"] is audit["passed"] is True
        and source["control_qualified"] is audit["control_qualified"] is False
        and source["new_models"] == source["new_forwards"] == source["new_physics_steps"] == 0,
        "The registered saved-box analysis and prior audit are required",
    )
    checked, mismatches, summaries = 0, [], {}
    for branch, steps in source["results"].items():
        summaries[branch] = {}
        for step_text, value in steps.items():
            point = worker["geometry"]["points"][branch][int(step_text) - 1]
            boxes = point["pad_alignment"]["box_projections"]
            summaries[branch][step_text] = {}
            for scenario in ("recorded", "offset"):
                delta = np.asarray(value[scenario]["pad_translation_world_m"], dtype=float)
                summaries[branch][step_text][scenario] = {}
                for pad in boxes[:2]:
                    pad_copy = dict(pad, center_world_m=np.asarray(pad["center_world_m"]) + delta)
                    reported = value[scenario]["pads"][pad["name"]]["pairs"]
                    prefix.require(
                        [row["target_geom"] for row in reported] == [box["name"] for box in boxes[2:]],
                        "Every target box must be independently checked in order",
                    )
                    overlap, near, separated = [], [], []
                    for row, target in zip(reported, boxes[2:], strict=True):
                        slack = box_slack(pad_copy, target)
                        actual = "overlap" if slack > 1e-10 else "separated" if slack < -1e-10 else "near"
                        if row["classification"] != "near" and actual != "near" and row["classification"] != actual:
                            mismatches.append(
                                dict(
                                    branch=branch,
                                    step=int(step_text),
                                    scenario=scenario,
                                    pad=pad["name"],
                                    target=target["name"],
                                    separation_result=row["classification"],
                                    lp_result=actual,
                                    lp_slack_m=slack,
                                )
                            )
                        {"overlap": overlap, "near": near, "separated": separated}[actual].append(target["name"])
                        checked += 1
                    summaries[branch][step_text][scenario][pad["name"]] = dict(
                        overlapping_targets=overlap, near_targets=near, separated_count=len(separated)
                    )
    prefix.require(
        checked == 2 * 4 * 2 * 2 * 40 and not mismatches, "Every registered pair must agree with the independent LP"
    )
    result = dict(
        complete=True,
        passed=True,
        control_qualified=False,
        source_ref=SOURCE,
        checked_pairs=checked,
        robust_sign_mismatches=mismatches,
        summaries=summaries,
        audit_models=0,
        audit_forwards=0,
        audit_physics_steps=0,
        audit_model_requests=0,
        automatic_retry=False,
    )
    print(publish_result(OUTPUT, result), flush=True)


if __name__ == "__main__":
    main()
