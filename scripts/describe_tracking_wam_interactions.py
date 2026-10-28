"""Describe closed-case task interactions, separately from TCP reference errors."""

import argparse
import json
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import numpy as np

from s2a.artifacts import REPO
from s2a.clean_training_runtime import publish, read, reference, require
from s2a.terminal_development_policy import case_order


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case-indices", type=int, nargs="+", required=True, choices=range(12))
    p.add_argument("--run-id", required=True, choices=["first_three", "task0_complete"])
    args = p.parse_args()
    require(len(set(args.case_indices)) == len(args.case_indices), "Unique explicit cases")
    root = REPO / "outputs/tracking_preview_wam_development_v3_20261027/development/cases"
    records = []
    for index in args.case_indices:
        case = case_order()[index]
        worker_ref = reference(root / case["case_id"] / "worker_result.json")
        worker = read(worker_ref)
        require(worker["complete"] is worker["passed"] is True and worker["physics_steps"] == 600, "Closed600 case")
        folder = Path(worker["physical_case_dir"])
        manifest_ref = reference(folder / "manifest.json")
        manifest = read(manifest_ref)
        rows_ref = reference(folder / "steps.jsonl")
        rows = [json.loads(x) for x in Path(rows_ref["path"]).read_text().splitlines()]
        require(len(rows) == 600, "Full control rows")
        target = {8: "akita_black_bowl_1", 9: "wine_bottle_1", 6: "cream_cheese_1"}[manifest["record"]["task_id"]]
        objects = [r["objects"][target] for r in rows]
        desired = np.asarray([r["opening_target"] for r in rows])
        actual = np.asarray([r["opening_after"] for r in rows])
        errors = np.asarray([r["tcp_error"] for r in rows])
        q = np.asarray([r["coupled_dynamics"]["robot_joint_position"] for r in rows])
        xml = ET.fromstring((folder / "model.xml").read_bytes())
        require(xml.find("compiler").get("angle") == "radian", "Explicit radians")
        joints = {n.get("name"): n for n in xml.iter("joint")}
        limits = np.asarray([[float(x) for x in joints[f"robot0_joint{k}"].get("range").split()] for k in range(1, 8)])
        margin = np.minimum(q - limits[:, 0], limits[:, 1] - q)
        crossing = np.flatnonzero(margin[:, 6] < 0)
        records.append(
            dict(
                case=case,
                target=target,
                success_steps=sum(r["success"] for r in rows),
                target_grasp_contact_steps=sum(o["grasp_contact"] for o in objects),
                target_max_height_gain_m=max(o["height_gain_m"] for o in objects),
                opening_target_range=[float(desired.min()), float(desired.max())],
                opening_actual_range=[float(actual.min()), float(actual.max())],
                opening_target_below_point2_steps=int(np.sum(desired < 0.2)),
                opening_target_above_point8_steps=int(np.sum(desired > 0.8)),
                opening_mae=float(np.mean(abs(desired - actual))),
                position_rmse_m=float(np.sqrt(np.mean(np.sum(errors[:, :3] ** 2, axis=1)))),
                so3_rmse_rad=float(np.sqrt(np.mean(np.sum(errors[:, 3:] ** 2, axis=1)))),
                routes=dict(Counter(r["tracking_refinement"]["route"] for r in rows)),
                minimum_actual_q7_margin_before_period_rad=float(margin[:, 6].min()),
                q7_outside_before_period_count=int(np.sum(margin[:, 6] < 0)),
                first_q7_outside_before_step_index=None if not len(crossing) else int(crossing[0]),
                worker_ref=worker_ref,
                rows_ref=rows_ref,
                manifest_ref=manifest_ref,
                xml_ref=reference(folder / "model.xml"),
            )
        )
    ref = publish(
        REPO / f"outputs/tracking_wam_interactions_20261027/{args.run_id}.json",
        dict(
            complete=True,
            passed=True,
            source_ref=reference(Path(__file__)),
            cases=records,
            physics_steps=0,
            world_model_calls=0,
            fresh_cases=0,
            opening_thresholds_descriptive_only=True,
            scope="Saved producer-log descriptors; not independent task verification, causal attribution, or development qualification",
        ),
    )
    print(json.dumps(ref), flush=True)


if __name__ == "__main__":
    main()
