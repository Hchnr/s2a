"""Summarize measured task interactions separately from generated-reference tracking."""

import json
from collections import Counter
from pathlib import Path

import numpy as np

from s2a.artifacts import REPO
from s2a.clean_training_runtime import publish, read, reference, require


def main():
    root = REPO / "outputs/serialized_preview_wam_development_20261026/development/cases"
    records = []
    for worker_path in sorted(root.glob("*/worker_result.json")):
        worker_ref = reference(worker_path)
        worker = read(worker_ref)
        if not worker["complete"]:
            continue
        folder = next(worker_path.parent.glob("simulation/*/*/steps.jsonl")).parent
        manifest_ref, rows_ref = reference(folder / "manifest.json"), reference(folder / "steps.jsonl")
        manifest = read(manifest_ref)
        rows = [json.loads(line) for line in Path(rows_ref["path"]).read_text().splitlines()]
        require(len(rows) == 600, "Complete600 case only")
        target = {8: "akita_black_bowl_1", 9: "wine_bottle_1", 6: "cream_cheese_1"}[manifest["record"]["task_id"]]
        targets = np.asarray([row["opening_target"] for row in rows])
        actual = np.asarray([row["opening_after"] for row in rows])
        errors = np.asarray([row["tcp_error"] for row in rows])
        objects = [row["objects"][target] for row in rows]
        xyz = np.asarray([obj["position"] for obj in objects])
        require(all(type(obj["grasp_contact"]) is bool for obj in objects), "Actual native grasp flags required")
        record = dict(
            case=worker["case"],
            task=manifest["record"]["task"],
            target=target,
            target_grasp_contact_steps=sum(obj["grasp_contact"] for obj in objects),
            target_max_height_gain_m=max(obj["height_gain_m"] for obj in objects),
            target_final_displacement_from_first_endpoint_m=float(np.linalg.norm(xyz[-1] - xyz[0])),
            opening_target_range=[float(targets.min()), float(targets.max())],
            opening_target_below_point2_steps=int(np.sum(targets < 0.2)),
            opening_target_above_point8_steps=int(np.sum(targets > 0.8)),
            opening_actual_range=[float(actual.min()), float(actual.max())],
            opening_mean_absolute_tracking_error=float(np.mean(np.abs(targets - actual))),
            tcp_position_rmse_m=float(np.sqrt(np.mean(np.sum(errors[:, :3] ** 2, axis=1)))),
            rotation_rmse_rad=float(np.sqrt(np.mean(np.sum(errors[:, 3:] ** 2, axis=1)))),
            gripper_routes=dict(Counter(row["gripper"]["route"] for row in rows)),
            success_steps=sum(row["success"] for row in rows),
            worker_ref=worker_ref,
            manifest_ref=manifest_ref,
            rows_ref=rows_ref,
        )
        records.append(record)
    require(len(records) == 8, "Expected exactly8 completed cases")
    result = dict(
        complete=True,
        passed=True,
        cases=records,
        source_ref=reference(Path(__file__)),
        scope="saved observation and reference descriptors; no causal intervention or fresh qualification",
        opening_thresholds_are_descriptive_only=True,
        new_model_calls=0,
        physics_steps=0,
    )
    print(json.dumps(publish(REPO / "outputs/preview_task_interaction_20261027/verified.json", result)))


if __name__ == "__main__":
    main()
