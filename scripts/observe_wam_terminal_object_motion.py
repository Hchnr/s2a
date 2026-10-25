"""Describe sealed 061 object motion offline; no online object-truth features."""

import json
import math
from pathlib import Path

from analyze_wam_release_horizon import reference

ROOT = Path("outputs/terminal_development_20261025")
TARGETS = ("akita_black_bowl_1", "wine_bottle_1", "cream_cheese_1")


def main():
    audit_path = ROOT / "review/DEVELOPMENT_verified.json"
    audit_ref = reference(audit_path)
    audit = json.loads(audit_path.read_text())
    if not (audit["complete"] and audit["passed"] and audit["physics_steps"] == 7200):
        raise ValueError("Requires full independent 061 audit")
    cases = []
    for worker_path in sorted((ROOT / "development/cases").glob("*/worker_result.json")):
        worker = json.loads(worker_path.read_text())
        steps_path = Path(worker["physical_case_dir"]) / "steps.jsonl"
        before = [reference(p) for p in (worker_path, steps_path)]
        rows = [json.loads(line) for line in steps_path.read_text().splitlines()]
        if not worker["complete"] or worker["actual_requests"] != 75 or len(rows) != 600:
            raise ValueError("Incomplete fixed case")
        if any(type(row["success"]) is not bool for row in rows):
            raise ValueError("Native success must be boolean")
        target = TARGETS[worker["case"]["task_index"]]
        objects = [row["objects"][target] for row in rows]
        origin = objects[0]["position"]
        cases.append(
            {
                "case": worker["case"]["case_id"],
                "target": target,
                "input_refs": before,
                "success_steps": sum(row["success"] for row in rows),
                "max_logged_height_gain_m": max(obj["height_gain_m"] for obj in objects),
                "max_displacement_from_first_logged_position_m": max(
                    math.dist(origin, obj["position"]) for obj in objects
                ),
                "terminal_displacement_from_first_logged_position_m": math.dist(origin, objects[-1]["position"]),
                "logged_grasp_contact_steps": sum(obj["grasp_contact"] for obj in objects),
                "saturated_steps": sum(row["saturated"] for row in rows),
                "last40_measured_opening_range": [
                    min(row["opening_after"] for row in rows[-40:]),
                    max(row["opening_after"] for row in rows[-40:]),
                ],
            }
        )
        if before != [reference(p) for p in (worker_path, steps_path)]:
            raise RuntimeError("Sealed inputs changed")
    if len(cases) != 12 or audit_ref != reference(audit_path):
        raise ValueError("Incomplete cohort or changed audit")
    result = {
        "scope": "Offline object motion description, not native predicate replacement or causal stage certification",
        "script_ref": reference(Path(__file__)),
        "audit_ref": audit_ref,
        "new_model_calls": 0,
        "new_physics_steps": 0,
        "fresh_cases": 0,
        "cases": cases,
    }
    with (ROOT / "full_object_motion_description.json").open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    for case in cases:
        print(json.dumps({k: v for k, v in case.items() if k != "input_refs"}))


if __name__ == "__main__":
    main()
